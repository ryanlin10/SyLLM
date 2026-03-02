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
    seed: int = 42

    # Reward
    task_type: str = "free_form"   # for SoundnessReward.score


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

        total_prompts = len(train_data)
        self.global_step = 0
        running_stats: Dict[str, float] = {"loss": 0.0, "reward": 0.0, "kl": 0.0, "adv": 0.0}

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
                        "step=%d  epoch=%d  loss=%.4f  reward=%.4f  kl=%.4f  adv_std=%.4f",
                        self.global_step, epoch + 1, avg["loss"], avg["reward"], avg["kl"], avg["adv"],
                    )
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({f"grpo/{k}": v for k, v in avg.items()}, step=self.global_step)
                    running_stats = {k: 0.0 for k in running_stats}

                # Save checkpoint.
                if cfg.save_steps > 0 and self.global_step % cfg.save_steps == 0:
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

            mean_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
            logger.info("Epoch %d/%d complete  --  mean reward = %.4f", epoch + 1, cfg.num_epochs, mean_reward)

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

        logger.info("Loading base model: %s", hf_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
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

        self.reward_fn = SoundnessReward()

    def _setup_optimizer(self, num_train_examples: int) -> None:
        """Create the AdamW optimizer and a linear warmup scheduler."""
        cfg = self.config
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        total_steps = (
            (num_train_examples // cfg.batch_size) * cfg.num_epochs
        ) // cfg.gradient_accumulation_steps
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

        For each prompt we:
        1. Generate ``group_size`` responses.
        2. Score them with the reward function.
        3. Compute group-relative advantages.
        4. Compute the clipped surrogate + KL loss.
        5. Back-propagate (with gradient accumulation support).

        Returns:
            Dictionary of scalar metrics for this step.
        """
        cfg = self.config
        G = cfg.group_size
        B = len(prompts)

        # Tokenise prompts.
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        ).to(self.device)

        prompt_ids = prompt_encodings["input_ids"]         # (B, prompt_len)
        prompt_mask = prompt_encodings["attention_mask"]    # (B, prompt_len)
        prompt_lengths = prompt_mask.sum(dim=1).tolist()    # list of int

        # -- Generate G responses per prompt ----------------------------------
        all_sequences: List[torch.Tensor] = []        # each (G, seq_len)
        all_old_log_probs: List[torch.Tensor] = []    # each (G, gen_len)
        all_gen_masks: List[torch.Tensor] = []         # each (G, gen_len)

        self.model.eval()
        with torch.no_grad():
            for i in range(B):
                seqs, lps, gmask = self._generate_group(
                    prompt_ids[i : i + 1], prompt_mask[i : i + 1]
                )
                all_sequences.append(seqs)
                all_old_log_probs.append(lps)
                all_gen_masks.append(gmask)
        self.model.train()

        # -- Score responses ---------------------------------------------------
        all_rewards = self._decode_and_score(
            all_sequences, prompt_lengths, targets
        )  # (B * G,)
        batch_rewards_list = all_rewards.tolist()

        # -- Compute group-relative advantages --------------------------------
        advantages = torch.zeros_like(all_rewards)
        for i in range(B):
            group_r = all_rewards[i * G : (i + 1) * G]
            mean_r = group_r.mean()
            std_r = group_r.std() + cfg.advantage_eps
            advantages[i * G : (i + 1) * G] = (group_r - mean_r) / std_r

        # -- GRPO loss --------------------------------------------------------
        metrics = self._grpo_step(
            all_sequences, all_old_log_probs, all_gen_masks,
            prompt_lengths, advantages,
        )

        metrics["reward"] = all_rewards.mean().item()
        metrics["adv"] = advantages.std().item()
        metrics["batch_rewards"] = batch_rewards_list
        return metrics

    @torch.no_grad()
    def _generate_group(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate ``group_size`` responses for a single prompt.

        Args:
            prompt_ids: (1, prompt_len) token IDs.
            attention_mask: (1, prompt_len) attention mask.

        Returns:
            sequences: (G, seq_len) full sequences (prompt + generation).
            log_probs: (G, gen_len) per-token log probs under current policy.
            gen_mask:  (G, gen_len) mask indicating real generated tokens (not pad).
        """
        cfg = self.config
        G = cfg.group_size

        # Expand prompt for the whole group.
        expanded_ids = prompt_ids.expand(G, -1)        # (G, prompt_len)
        expanded_mask = attention_mask.expand(G, -1)    # (G, prompt_len)

        # Generate.
        outputs = self.model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=cfg.max_gen_length,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences  # (G, seq_len)
        prompt_len = prompt_ids.shape[1]
        gen_ids = sequences[:, prompt_len:]   # (G, gen_len)
        gen_len = gen_ids.shape[1]

        # Build generation mask (1 for real tokens, 0 for padding).
        gen_mask = (gen_ids != self.tokenizer.pad_token_id).float()

        # Compute per-token log probs by re-running the model on the full
        # sequences (more numerically stable than using output_scores from
        # generate which may go through different code paths).
        log_probs = self._compute_log_probs(
            self.model, sequences,
            torch.ones_like(sequences, dtype=torch.long),
            gen_ids,
            prompt_len,
        )  # (G, gen_len)

        return sequences, log_probs, gen_mask

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
                torch.ones_like(sequences, dtype=torch.long),
                gen_ids,
                prompt_len,
            )
            self.model.enable_adapter_layers()
        self.model.train()
        return ref_lps

    def _grpo_step(
        self,
        all_sequences: List[torch.Tensor],
        all_old_log_probs: List[torch.Tensor],
        all_gen_masks: List[torch.Tensor],
        prompt_lengths: List[int],
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute and apply one GRPO optimisation step.

        Args:
            all_sequences: Per-prompt list of (G, seq_len) tensors.
            all_old_log_probs: Per-prompt list of (G, gen_len) tensors.
            all_gen_masks: Per-prompt list of (G, gen_len) tensors.
            prompt_lengths: Prompt length for each prompt in the batch.
            advantages: (B * G,) group-relative advantage values.

        Returns:
            Dictionary with scalar metrics (loss, kl).
        """
        cfg = self.config
        G = cfg.group_size
        B = len(all_sequences)

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = 0.0
        total_tokens = 0

        for i in range(B):
            sequences = all_sequences[i]           # (G, seq_len)
            old_lps = all_old_log_probs[i]         # (G, gen_len)
            gen_mask = all_gen_masks[i]             # (G, gen_len)
            prompt_len = prompt_lengths[i]
            gen_ids = sequences[:, prompt_len:]     # (G, gen_len)
            group_adv = advantages[i * G : (i + 1) * G]  # (G,)

            # Current policy log probs.
            new_lps = self._compute_log_probs(
                self.model, sequences,
                torch.ones_like(sequences, dtype=torch.long),
                gen_ids, prompt_len,
            )  # (G, gen_len)

            # Reference log probs (adapter-disabled base model).
            ref_lps = self._compute_ref_log_probs(sequences, gen_ids, prompt_len)

            # Per-token importance sampling ratio.
            ratio = torch.exp(new_lps - old_lps.detach())  # (G, gen_len)

            # Expand advantages to per-token.
            adv_expanded = group_adv.unsqueeze(-1).expand_as(ratio)  # (G, gen_len)

            # Clipped surrogate objective (per token).
            surr1 = ratio * adv_expanded
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv_expanded
            # We take the min and negate because we want to *maximise* the objective.
            policy_loss = -torch.min(surr1, surr2)  # (G, gen_len)

            # KL divergence: KL(pi || pi_ref) = sum( pi * (log_pi - log_pi_ref) )
            # Approximated per token as: new_lp - ref_lp (since we condition on
            # the same sequence, this is the per-token KL contribution).
            kl_per_token = new_lps - ref_lps.detach()  # (G, gen_len)
            kl_penalty = cfg.kl_coeff * kl_per_token   # (G, gen_len)

            # Combine and mask.
            per_token_loss = (policy_loss + kl_penalty) * gen_mask  # (G, gen_len)
            n_tokens = gen_mask.sum()
            if n_tokens > 0:
                total_loss = total_loss + per_token_loss.sum() / n_tokens
            total_kl += (kl_per_token * gen_mask).sum().item() / max(n_tokens.item(), 1)
            total_tokens += n_tokens.item()

        # Average over the batch.
        mean_loss = total_loss / max(B, 1)

        # Backward and step (with gradient accumulation).
        scaled_loss = mean_loss / cfg.gradient_accumulation_steps
        scaled_loss.backward()

        if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss": mean_loss.item(),
            "kl": total_kl / max(B, 1),
        }

    # ------------------------------------------------------------------
    # Reward scoring
    # ------------------------------------------------------------------

    def _decode_and_score(
        self,
        all_sequences: List[torch.Tensor],
        prompt_lengths: List[int],
        targets: List[str],
    ) -> torch.Tensor:
        """Decode generated sequences and compute reward scores.

        Args:
            all_sequences: Per-prompt list of (G, seq_len) tensors.
            prompt_lengths: Prompt length for each prompt in the batch.
            targets: Ground-truth target strings (one per prompt).

        Returns:
            rewards: (B * G,) tensor of reward scores.
        """
        cfg = self.config
        G = cfg.group_size
        B = len(all_sequences)

        all_outputs: List[str] = []
        all_targets: List[str] = []
        all_task_types: List[str] = []

        for i in range(B):
            sequences = all_sequences[i]  # (G, seq_len)
            prompt_len = prompt_lengths[i]
            for g in range(G):
                gen_ids = sequences[g, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_outputs.append(text)
                all_targets.append(targets[i])
                all_task_types.append(cfg.task_type)

        # Score with SoundnessReward if available.
        if self.reward_fn is not None:
            results = self.reward_fn.score_batch(all_outputs, all_targets, all_task_types)
            rewards = torch.tensor(
                [r.reward for r in results], dtype=torch.float32, device=self.device
            )
        else:
            # Fallback: simple length-normalised overlap heuristic.
            rewards = self._fallback_reward(all_outputs, all_targets)

        return rewards

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
