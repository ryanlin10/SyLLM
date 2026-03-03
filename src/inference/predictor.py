"""Inference using vLLM for efficient text generation."""

from typing import Optional, List
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest


def _patch_mistral_tokenizer():
    """Patch MistralCommonTokenizer to fix two compatibility bugs with vLLM on
    Mistral 3.x multimodal models:

    1. from_pretrained rejects _commit_hash kwargs passed by AutoTokenizer.
    2. _piece_to_id encodes special tokens (e.g. [IMG]) via tiktoken which has
       no special tokens registered, returning 3 pieces instead of 1. The correct
       ID is available in tokenizer._special_tokens_reverse_vocab.
    """
    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
        import functools

        # --- Patch 1: strip unknown private kwargs from from_pretrained ---
        original_fp = MistralCommonTokenizer.from_pretrained.__func__

        @classmethod
        @functools.wraps(original_fp)
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
            allowed_private = {"_from_auto", "trust_remote_code"}
            for k in list(kwargs.keys()):
                if k.startswith("_") and k not in allowed_private:
                    kwargs.pop(k)
            return original_fp(cls, pretrained_model_name_or_path, *init_inputs, **kwargs)

        MistralCommonTokenizer.from_pretrained = patched_from_pretrained

        # --- Patch 2: look up special tokens directly before tiktoken encode ---
        original_p2id = MistralCommonTokenizer._piece_to_id

        def patched_piece_to_id(self, piece: str) -> int:
            try:
                inner = self.tokenizer.instruct_tokenizer.tokenizer
                rev = getattr(inner, "_special_tokens_reverse_vocab", {})
                if piece in rev:
                    return rev[piece]
            except Exception:
                pass
            return original_p2id(self, piece)

        MistralCommonTokenizer._piece_to_id = patched_piece_to_id

    except Exception:
        pass  # If patching fails, proceed and let vLLM report the real error


def _safe_gpu_memory_utilization(requested: float = 0.9, headroom: float = 0.05) -> float:
    """Return a gpu_memory_utilization that fits within current free VRAM.

    If the requested value would exceed free memory, scales it down to
    (free / total) * (1 - headroom), capped at the requested value.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            max_safe = (free / total) * (1.0 - headroom)
            if requested > max_safe:
                adjusted = round(max_safe, 2)
                print(
                    f"[VLLMPredictor] Requested gpu_memory_utilization={requested} exceeds "
                    f"available VRAM ({free/2**30:.1f}/{total/2**30:.1f} GiB free). "
                    f"Lowering to {adjusted}."
                )
                return adjusted
    except Exception:
        pass
    return requested


class VLLMPredictor:
    """Simple predictor using vLLM for efficient inference."""

    def __init__(
        self,
        model_path: str,
        lora_adapter_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        download_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        enforce_eager: bool = False,
    ):
        """
        Initialize the vLLM predictor.

        Args:
            model_path: HuggingFace model path or local path to base model
            lora_adapter_path: Optional path to LoRA adapter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None for model default).
                Reducing this (e.g. 4096–8192) shrinks the KV-cache and allows
                more concurrent sequences, improving throughput significantly.
            download_dir: Directory for vLLM to cache model weights (passed as
                cache_dir to snapshot_download; use an existing cache to avoid
                re-downloading)
            hf_token: HuggingFace token for gated models (also read from HF_TOKEN
                env var); set before spawning EngineCore subprocess
            enforce_eager: Disable CUDA graph compilation (default False). Set
                True only when the compilation cache disk is full or for
                debugging; it causes a 2–4× throughput regression.
        """
        import os
        # Set HF_TOKEN before vLLM spawns its EngineCore subprocess so the child
        # inherits it and can authenticate with HuggingFace.
        resolved_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if resolved_token:
            os.environ["HF_TOKEN"] = resolved_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved_token

        # Redirect vLLM compilation/model caches away from any full disks.
        # Override unconditionally — VLLM_CACHE_ROOT may already point at a
        # full disk (e.g. /workspace/.vllm_cache) in the inherited environment.
        os.environ["VLLM_CACHE_ROOT"] = "/tmp/vllm_cache"
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

        # Apply compatibility patch before vLLM initialises the processor
        _patch_mistral_tokenizer()

        self.model_path = model_path
        self.lora_adapter_path = lora_adapter_path
        self.lora_request = None

        llm_kwargs = {
            "model": model_path,
            "tokenizer_mode": "mistral",
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": _safe_gpu_memory_utilization(gpu_memory_utilization),
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "download_dir": download_dir,
            "limit_mm_per_prompt": {"image": 0},
            "enforce_eager": enforce_eager,
        }

        if lora_adapter_path:
            llm_kwargs["enable_lora"] = True
            self.lora_request = LoRARequest("finetuned", 1, lora_adapter_path)

        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_lora: Optional[bool] = None,
    ) -> str:
        """
        Generate a response for a single message.

        Args:
            message: User message/prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_lora: Override LoRA usage. None=default behavior,
                      True=force LoRA, False=force base model only.

        Returns:
            Raw generated text
        """
        messages = self._build_messages(message, system_prompt)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_kwargs = {"sampling_params": sampling_params}
        lora = self._resolve_lora(use_lora)
        if lora:
            chat_kwargs["lora_request"] = lora

        outputs = self.llm.chat(messages, **chat_kwargs)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_lora: Optional[bool] = None,
    ) -> List[str]:
        """
        Generate responses for multiple messages.

        Args:
            messages: List of user messages
            system_prompt: Optional system prompt (applied to all)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_lora: Override LoRA usage. None=default behavior,
                      True=force LoRA, False=force base model only.

        Returns:
            List of raw generated texts
        """
        conversations = [self._build_messages(msg, system_prompt) for msg in messages]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_kwargs = {"sampling_params": sampling_params}
        lora = self._resolve_lora(use_lora)
        if lora:
            chat_kwargs["lora_request"] = lora

        outputs = self.llm.chat(conversations, **chat_kwargs)
        return [output.outputs[0].text for output in outputs]

    def _resolve_lora(self, use_lora: Optional[bool] = None) -> Optional[LoRARequest]:
        """Resolve whether to use LoRA for this request.

        Args:
            use_lora: None=default (use if configured), True=force LoRA,
                      False=force base model only.

        Returns:
            LoRARequest if LoRA should be used, None otherwise.
        """
        if use_lora is None:
            return self.lora_request
        if use_lora:
            return self.lora_request
        return None

    def _build_messages(
        self, message: str, system_prompt: Optional[str] = None
    ) -> List[dict]:
        """Build message list for chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return messages
