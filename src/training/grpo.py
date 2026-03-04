"""
Group Relative Policy Optimization (GRPO) trainer for reinforcement learning
on reasoning tasks.

GRPO generates multiple responses per prompt, scores them with a reward function,
and computes group-relative advantages to update the policy. This avoids the need
for a learned value function (as in PPO) by using the group mean/std as a baseline.

Algorithm:
    1. For each prompt, sample G responses from the current policy.
    2. Score each response with SoundnessReward.
    3. Normalize rewards within each group: adv = (r - mean(r)) / (std(r) + eps).
    4. Compute importance-sampling ratio: ratio = exp(log_pi_new - log_pi_old).
    5. Clipped surrogate loss: L = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv).
    6. KL penalty against frozen reference model: kl_loss = kl_coeff * KL(pi || pi_ref).
    7. Total loss = L + kl_loss.

Reference:
    Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
    in Open Language Models", 2024.
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional heavy imports -- guarded so the module can be imported without
# GPU libraries installed.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root so sibling imports work when running as a script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry -- import from lora_finetune when available, otherwise
# provide a minimal fallback so that the module remains self-contained.
# ---------------------------------------------------------------------------

def _get_model_info(model_key: str) -> Dict[str, Any]:
    """Resolve a model key to its registry entry.

    Attempts to reuse ``scripts.lora_finetune.get_model_info`` if importable,
    otherwise falls back to a sensible default.
    """
    try:
        from scripts.lora_finetune import get_model_info  # type: ignore[import]
        return get_model_info(model_key)
    except Exception:
        pass

    # Minimal fallback -- covers the most common case.
    return {
        "name": model_key,
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_length": 4096,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    lora_adapter_path: Optional[str] = None  # path to SFT adapter to start from
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # GRPO hyper-parameters
    group_size: int = 4            # number of responses per prompt (G)
    clip_epsilon: float = 0.2      # PPO-style clipping
    kl_coeff: float = 0.05         # KL penalty coefficient
    advantage_eps: float = 1e-8    # epsilon for advantage normalization

    # Generation
    max_gen_length: int = 512
    max_prompt_length: int = 1024
    temperature: float = 0.7       # sampling temperature for generation
    top_p: float = 1.0

    # Optimization
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 2            # prompts per optimization step
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # Precision / memory
    fp16: bool = True
    bf16: bool = False
    use_4bit: bool = False
    use_8bit: bool = False
    gradient_checkpointing: bool = True

    # Logging / saving
    output_dir: str = "./outputs/grpo"
    logging_steps: int = 10
    save_steps: int = 200
    sample_steps: int = 50       # log qualitative sample completions every N steps (0 = off)
    n_sample_prompts: int = 2    # number of prompts to sample per logging event
    seed: int = 42

    # Reward
    task_type: str = "free_form"   # for SoundnessReward.score
    verifier_weight: float = 0.3   # weight for process/verifier reward component
    outcome_weight: float = 0.7    # weight for outcome (correct answer) reward component
    correct_reward: float = 1.0    # outcome signal for a correct answer
    wrong_reward: float = -1.0     # outcome signal for a wrong answer (negative = penalty)
    skip_z3_verify: bool = False   # skip Z3 per-step verification (use optimistic process reward)

    # Attention / compute
    use_flash_attention: bool = True  # use flash_attention_2 if available

    # Parallel Z3 verification
    # Number of worker threads for concurrent Z3 segment verification.
    # Z3's C extension releases the GIL during solver.check(), so threads give
    # true parallelism.  At B*G=16 responses x ~5 segments = 80 calls: with 16
    # workers this reduces sequential 400 s worst-case to ~25 s.
    # Set to 0 to disable (fall back to sequential per-response verification).
    n_verify_workers: int = 16


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    GRPO eliminates the need for a learned value function by computing
    advantages relative to a group of sampled responses for each prompt.
    A frozen copy of the initial policy serves as the KL reference.

    Typical usage::

        config = GRPOConfig(model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
        trainer = GRPOTrainer(config)
        trainer.train(train_data)
    """

    def __init__(self, config: GRPOConfig) -> None:
        self._check_deps()
        self.config = config
        self.reward_fn = None       # SoundnessReward, created in train()
        self.model = None           # policy model (with LoRA)
        self.ref_model = None       # frozen reference model (base + optional SFT adapter)
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.global_step: int = 0

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_deps() -> None:
        """Ensure heavy dependencies are available."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GRPOTrainer. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for GRPOTrainer. Install with: pip install transformers")
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for GRPOTrainer. Install with: pip install peft")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_data: List[Dict[str, str]]) -> str:
        """Run the full GRPO training loop.

        Args:
            train_data: List of dicts, each with ``"prompt"`` and ``"target"`` keys.

        Returns:
            Path to the final saved model directory.
        """
        cfg = self.config
        set_seed(cfg.seed)

        # 1. Setup model, tokenizer, LoRA, reference model.
        logger.info("Setting up model and tokenizer ...")
        self._setup_model()

        # 2. Initialise reward function (lazy import to avoid circular deps).
        logger.info("Initialising SoundnessReward ...")
        self._setup_reward()

        # 3. Optimiser and scheduler.
        self._setup_optimizer(len(train_data))

        # 4. Output directory.
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. Training loop.
        logger.info("Starting GRPO training  --  %d examples, %d epochs", len(train_data), cfg.num_epochs)
        logger.info("  group_size=%d  clip_eps=%.3f  kl_coeff=%.4f", cfg.group_size, cfg.clip_epsilon, cfg.kl_coeff)
        logger.info("  batch_size=%d  grad_accum=%d  lr=%.2e", cfg.batch_size, cfg.gradient_accumulation_steps, cfg.learning_rate)

        self._train_data = train_data   # stored for use by _log_samples
        total_prompts = len(train_data)
        self.global_step = 0
        running_stats: Dict[str, float] = {
            "loss": 0.0, "reward": 0.0, "kl": 0.0, "adv": 0.0,
            "process_reward": 0.0, "outcome_reward": 0.0,
            "total_segs": 0.0, "sound_segs": 0.0,
        }

        for epoch in range(cfg.num_epochs):
            # Shuffle data each epoch.
            indices = np.random.permutation(total_prompts).tolist() if NUMPY_AVAILABLE else list(range(total_prompts))
            epoch_rewards: List[float] = []

            for batch_start in range(0, total_prompts, cfg.batch_size):
                batch_indices = indices[batch_start : batch_start + cfg.batch_size]
                batch = [train_data[i] for i in batch_indices]

                prompts = [ex["prompt"] for ex in batch]
                targets = [ex["target"] for ex in batch]

                step_metrics = self._train_step(prompts, targets)

                # Accumulate stats.
                for k in running_stats:
                    running_stats[k] += step_metrics.get(k, 0.0)
                epoch_rewards.extend(step_metrics.get("batch_rewards", []))

                self.global_step += 1

                # Logging.
                if self.global_step % cfg.logging_steps == 0:
                    avg = {k: v / cfg.logging_steps for k, v in running_stats.items()}
                    logger.info(
                        "step=%d  epoch=%d  loss=%.4f  reward=%.4f  "
                        "verifier=%.4f  outcome=%.4f  kl=%.4f  adv_std=%.4f  "
                        "segs=%d  sound=%d",
                        self.global_step, epoch + 1,
                        avg["loss"], avg["reward"],
                        avg["process_reward"], avg["outcome_reward"],
                        avg["kl"], avg["adv"],
                        int(avg["total_segs"]), int(avg["sound_segs"]),
                    )
                    if WANDB_AVAILABLE and wandb.run is not None:
                        log_dict = {f"grpo/{k}": v for k, v in avg.items() if k != "batch_rewards"}
                        log_dict["grpo/lr"] = self.scheduler.get_last_lr()[0]
                        wandb.log(log_dict, step=self.global_step)
                    running_stats = {k: 0.0 for k in running_stats}

                # Qualitative sample logging.
                if cfg.sample_steps > 0 and self.global_step % cfg.sample_steps == 0:
                    self._log_samples(self.global_step)

                # Save checkpoint.
                if cfg.save_steps > 0 and self.global_step % cfg.save_steps == 0:
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

            # Flush any uncommitted accumulated gradients at epoch boundary.
            if cfg.gradient_accumulation_steps > 1 and \
                    self.global_step % cfg.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            mean_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            logger.info("Epoch %d/%d complete  --  mean reward = %.4f", epoch + 1, cfg.num_epochs, mean_reward)
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"grpo/epoch_mean_reward": mean_reward, "epoch": epoch + 1}, step=self.global_step)

        # Save final model.
        final_dir = output_dir / "final"
        self._save_checkpoint(final_dir)
        logger.info("Training complete. Final model saved to %s", final_dir)
        return str(final_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        """Load the base model with LoRA and prepare a frozen reference."""
        cfg = self.config
        model_info = _get_model_info(cfg.model_name)
        hf_name = model_info.get("name", cfg.model_name)
        target_modules = cfg.lora_target_modules or model_info.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        # Device / dtype.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.bf16:
            dtype = torch.bfloat16
        elif cfg.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Quantisation config.
        quant_config = None
        if cfg.use_4bit or cfg.use_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=cfg.use_4bit,
                load_in_8bit=cfg.use_8bit,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Determine attention implementation.
        attn_impl = None
        if cfg.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 enabled")
            except ImportError:
                logger.info("flash_attn not installed; using SDPA (PyTorch built-in)")
                attn_impl = "sdpa"

        def _load_kwargs():
            kw = dict(
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quant_config,
            )
            if attn_impl:
                kw["attn_implementation"] = attn_impl
            return kw

        logger.info("Loading base model: %s", hf_name)
        model_class_name = model_info.get("model_class")
        if model_class_name == "Mistral3ForConditionalGeneration":
            from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
            logger.info("Using special model class: %s", model_class_name)
            base_model = Mistral3ForConditionalGeneration.from_pretrained(
                hf_name, **_load_kwargs()
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                hf_name, **_load_kwargs()
            )

        # Tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for generation

        # Prepare for kbit if quantised.
        if cfg.use_4bit or cfg.use_8bit:
            base_model = prepare_model_for_kbit_training(base_model)

        # Enable gradient checkpointing.
        if cfg.gradient_checkpointing:
            base_model.config.use_cache = False
            base_model.gradient_checkpointing_enable()

        # Load existing SFT adapter or create a new LoRA adapter.
        if cfg.lora_adapter_path and Path(cfg.lora_adapter_path).exists():
            logger.info("Loading SFT adapter from %s", cfg.lora_adapter_path)
            self.model = PeftModel.from_pretrained(base_model, cfg.lora_adapter_path, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_cfg)

        self.model.print_trainable_parameters()
        self.model.train()

        # Reference model -- we reuse the *same* base weights but run them in
        # eval / no-grad mode. For LoRA models ``model.disable_adapter_layers()``
        # gives us the frozen base, but we also want the SFT adapter weights
        # frozen as the reference. The simplest approach: merge + keep a copy
        # of the adapter weights, then re-apply. Because that doubles memory,
        # we instead just evaluate under ``torch.no_grad`` with the current
        # adapter frozen (we detach log probs). This is equivalent at step 0.
        # During training we compute reference log probs by temporarily
        # disabling adapter layers.
        self.ref_model = None  # sentinel -- we use adapter toggling instead

    def _setup_reward(self) -> None:
        """Initialise the SoundnessReward scorer."""
        try:
            from src.training.soundness_reward import SoundnessReward
        except ImportError:
            try:
                from .soundness_reward import SoundnessReward
            except ImportError:
                logger.warning(
                    "SoundnessReward not available. Using a simple string-match fallback."
                )
                self.reward_fn = None
                return

        self.reward_fn = SoundnessReward(
            verifier_weight=self.config.verifier_weight,
            outcome_weight=self.config.outcome_weight,
            correct_reward=self.config.correct_reward,
            wrong_reward=self.config.wrong_reward,
            skip_verify=self.config.skip_z3_verify,
            n_verify_workers=self.config.n_verify_workers,
        )

    def _setup_optimizer(self, num_train_examples: int) -> None:
        """Create the AdamW optimizer and a linear warmup scheduler."""
        cfg = self.config
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        batches_per_epoch = math.ceil(num_train_examples / cfg.batch_size)
        total_steps = (batches_per_epoch * cfg.num_epochs) // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (
                min(1.0, step / max(warmup_steps, 1))
                if step < warmup_steps
                else max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
            ),
        )

    # ------------------------------------------------------------------
    # Core GRPO logic
    # ------------------------------------------------------------------

    def _train_step(
        self, prompts: List[str], targets: List[str]
    ) -> Dict[str, Any]:
        """Execute a single GRPO optimisation step for a batch of prompts.

        All B prompts are tiled to B*G sequences and passed to model.generate()
        in one call, then the policy and reference forward passes each process
        all B*G sequences at once.  This keeps the GPU batch large and avoids
        the per-prompt sequential loop that leaves the GPU underutilised.

        Returns:
            Dictionary of scalar metrics for this step.
        """
        cfg = self.config
        B   = len(prompts)
        G   = cfg.group_size

        # ------------------------------------------------------------------ #
        # 1. Tokenise all B prompts.                                          #
        #    The tokeniser left-pads shorter prompts to the longest in the    #
        #    batch, so every row shares the same padded prefix length P.      #
        # ------------------------------------------------------------------ #
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        ).to(self.device)
        prompt_ids  = enc["input_ids"]        # (B, P)
        prompt_mask = enc["attention_mask"]   # (B, P)
        P = prompt_ids.shape[1]               # padded prompt length, same for all

        # ------------------------------------------------------------------ #
        # 2. Tile to (B*G, P) and generate all responses in one call.        #
        #    repeat_interleave groups responses by prompt:                    #
        #      rows [0 .. G-1]    → prompt 0                                 #
        #      rows [G .. 2G-1]   → prompt 1, …                              #
        # ------------------------------------------------------------------ #
        tiled_ids  = prompt_ids.repeat_interleave(G, dim=0)   # (B*G, P)
        tiled_mask = prompt_mask.repeat_interleave(G, dim=0)  # (B*G, P)

        self.model.eval()
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=tiled_ids,
                attention_mask=tiled_mask,
                max_new_tokens=cfg.max_gen_length,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,   # raw per-step logits → old log-probs, no extra pass
            )
        self.model.train()

        sequences = gen_out.sequences              # (B*G, P + gen_len)
        gen_ids   = sequences[:, P:]               # (B*G, gen_len)
        gen_mask  = (gen_ids != self.tokenizer.pad_token_id).float()  # (B*G, gen_len)
        gen_len   = gen_ids.shape[1]

        # Full-sequence attention mask for log-prob forward passes:
        # prompt portion uses tiled_mask (0=pad, 1=real, matching generation-time),
        # generated portion uses gen_mask (0=post-EOS pad, 1=real token).
        full_seq_mask = torch.cat([tiled_mask, gen_mask.long()], dim=1)  # (B*G, P + gen_len)

        # Old log-probs extracted from the generation-time logits — avoids a
        # separate forward pass.  Process one time-step at a time to avoid
        # materialising the full (B*G, gen_len, vocab) tensor (~8-17 GB at
        # B*G=32, gen_len=512, vocab=131k).
        if gen_out.logits and len(gen_out.logits) == gen_len:
            old_lps = torch.stack([
                F.log_softmax(logit_t, dim=-1)          # (B*G, vocab)
                 .gather(-1, gen_ids[:, t : t + 1])     # (B*G, 1)
                 .squeeze(-1)                            # (B*G,)
                for t, logit_t in enumerate(gen_out.logits)
            ], dim=1)                                   # (B*G, gen_len)
        else:
            # Edge-case fallback (e.g. older transformers version).
            with torch.no_grad():
                old_lps = self._compute_log_probs(
                    self.model,
                    sequences,
                    full_seq_mask,
                    gen_ids, P,
                )

        # ------------------------------------------------------------------ #
        # 3. Decode, expand targets / task-types, and score all B*G outputs. #
        # ------------------------------------------------------------------ #
        all_texts   = [
            self.tokenizer.decode(gen_ids[k], skip_special_tokens=True)
            for k in range(B * G)
        ]
        all_targets = [targets[i] for i in range(B) for _ in range(G)]
        all_types   = [cfg.task_type] * (B * G)

        if self.reward_fn is not None:
            results         = self.reward_fn.score_batch(all_texts, all_targets, all_types)
            all_rewards     = torch.tensor([r.reward          for r in results], dtype=torch.float32, device=self.device)
            process_rewards = torch.tensor([r.process_reward  for r in results], dtype=torch.float32, device=self.device)
            outcome_rewards = torch.tensor([r.outcome_reward  for r in results], dtype=torch.float32, device=self.device)
            total_segs      = sum(r.total_steps for r in results)
            sound_segs      = sum(r.sound_steps for r in results)
        else:
            all_rewards     = self._fallback_reward(all_texts, all_targets)
            process_rewards = torch.zeros_like(all_rewards)
            outcome_rewards = all_rewards.clone()
            total_segs      = 0
            sound_segs      = 0

        # ------------------------------------------------------------------ #
        # 4. Group-relative advantages (normalise within each prompt's group).#
        # ------------------------------------------------------------------ #
        advantages = torch.zeros(B * G, dtype=torch.float32, device=self.device)
        for i in range(B):
            sl      = slice(i * G, (i + 1) * G)
            group_r = all_rewards[sl]
            advantages[sl] = (group_r - group_r.mean()) / (group_r.std() + cfg.advantage_eps)

        # ------------------------------------------------------------------ #
        # 5. Single batched forward passes for new-policy and ref log-probs.  #
        # ------------------------------------------------------------------ #
        # New policy (adapter enabled, gradients on).
        new_lps = self._compute_log_probs(
            self.model,
            sequences,
            full_seq_mask,
            gen_ids, P,
        )  # (B*G, gen_len)

        # Reference policy (adapter disabled, no grad).
        ref_lps = self._compute_ref_log_probs(sequences, gen_ids, P, full_seq_mask)  # (B*G, gen_len)

        # ------------------------------------------------------------------ #
        # 6. GRPO loss (clipped surrogate + KL) averaged over real tokens.   #
        # ------------------------------------------------------------------ #
        ratio       = torch.exp(new_lps - old_lps.detach())
        adv_exp     = advantages.unsqueeze(-1).expand_as(ratio)
        surr1       = ratio * adv_exp
        surr2       = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_exp
        policy_loss = -torch.min(surr1, surr2)

        kl_per_tok  = new_lps - ref_lps.detach()
        kl_penalty  = cfg.kl_coeff * kl_per_tok

        per_tok_loss = (policy_loss + kl_penalty) * gen_mask
        n_toks       = gen_mask.sum().clamp(min=1)
        loss         = per_tok_loss.sum() / n_toks
        kl_mean      = (kl_per_tok * gen_mask).sum() / n_toks

        # ------------------------------------------------------------------ #
        # 7. Backward + gradient accumulation.                                #
        # ------------------------------------------------------------------ #
        (loss / cfg.gradient_accumulation_steps).backward()
        if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss":           loss.item(),
            "kl":             kl_mean.item(),
            "reward":         all_rewards.mean().item(),
            "adv":            advantages.std().item(),
            "batch_rewards":  all_rewards.tolist(),
            "process_reward": process_rewards.mean().item(),
            "outcome_reward": outcome_rewards.mean().item(),
            "total_segs":     float(total_segs),
            "sound_segs":     float(sound_segs),
        }

    def _compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the generated portion.

        Args:
            model: The language model (policy or reference).
            input_ids: (batch, seq_len) full sequences.
            attention_mask: (batch, seq_len) attention mask.
            labels: (batch, gen_len) generated token IDs we want log probs for.
            prompt_len: Length of the prompt prefix.

        Returns:
            log_probs: (batch, gen_len) per-token log probabilities.
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # logits shape: (batch, seq_len, vocab_size)
        logits = outputs.logits

        # We need log probs at positions [prompt_len-1 .. seq_len-2] because
        # the logit at position t predicts token at position t+1.
        gen_logits = logits[:, prompt_len - 1 : -1, :]  # (batch, gen_len, vocab)
        log_probs_all = F.log_softmax(gen_logits, dim=-1)  # (batch, gen_len, vocab)

        # Gather the log probs for the actual generated tokens.
        log_probs = log_probs_all.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, gen_len)

        return log_probs

    def _compute_ref_log_probs(
        self,
        sequences: torch.Tensor,
        gen_ids: torch.Tensor,
        prompt_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probs under the reference policy.

        We disable the LoRA adapter layers so that forward passes go through
        the frozen base model only, avoiding a full model copy.
        """
        self.model.eval()
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_lps = self._compute_log_probs(
                self.model,
                sequences,
                attention_mask,
                gen_ids,
                prompt_len,
            )
            self.model.enable_adapter_layers()
        self.model.train()
        return ref_lps


    def _fallback_reward(
        self, outputs: List[str], targets: List[str]
    ) -> torch.Tensor:
        """Simple fallback reward when SoundnessReward is unavailable.

        Uses token-level F1 overlap between the generated output and the
        target string, yielding a score in [0, 1].
        """
        scores: List[float] = []
        for out, tgt in zip(outputs, targets):
            out_tokens = set(out.lower().split())
            tgt_tokens = set(tgt.lower().split())
            if not tgt_tokens:
                scores.append(0.0)
                continue
            if not out_tokens:
                scores.append(0.0)
                continue
            precision = len(out_tokens & tgt_tokens) / len(out_tokens)
            recall = len(out_tokens & tgt_tokens) / len(tgt_tokens)
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                scores.append(f1)
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Qualitative monitoring
    # ------------------------------------------------------------------

    def _log_samples(self, step: int) -> None:
        """Generate and log sample completions for qualitative monitoring.

        Picks ``cfg.n_sample_prompts`` examples at random from the training
        data, generates with greedy decoding (deterministic), and logs the
        prompt tail + completion to the logger and to W&B as a Table.
        """
        cfg = self.config
        train_data = getattr(self, "_train_data", None)
        if not train_data:
            return

        n = min(cfg.n_sample_prompts, len(train_data))
        if NUMPY_AVAILABLE:
            rng = np.random.default_rng(seed=step)  # reproducible per step
            indices = rng.choice(len(train_data), n, replace=False).tolist()
        else:
            indices = list(range(n))
        samples = [train_data[i] for i in indices]

        self.model.eval()
        rows = []
        with torch.no_grad():
            for ex in samples:
                prompt = ex["prompt"]
                target = ex.get("target", "")
                enc = self.tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_prompt_length,
                ).to(self.device)
                gen_out = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=cfg.max_gen_length,
                    do_sample=False,  # greedy — deterministic snapshot of current policy
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                completion = self.tokenizer.decode(
                    gen_out[0, enc["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                # Show the tail of the prompt (most relevant context).
                prompt_tail = prompt[-300:] if len(prompt) > 300 else prompt
                rows.append((prompt_tail, completion, target))
                logger.info(
                    "SAMPLE step=%d\n  PROMPT (tail): %s\n  COMPLETION: %s\n  TARGET: %s",
                    step, prompt_tail, completion, target,
                )
        self.model.train()

        if WANDB_AVAILABLE and wandb.run is not None:
            tbl = wandb.Table(columns=["step", "prompt_tail", "completion", "target"])
            for prompt_tail, completion, target in rows:
                tbl.add_data(step, prompt_tail, completion, target)
            wandb.log({"grpo/samples": tbl}, step=step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the LoRA adapter, tokenizer, and training config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving checkpoint to %s", path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save training config as JSON for reproducibility.
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        with open(path / "grpo_config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Save optimizer state.
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            path / "training_state.pt",
        )
