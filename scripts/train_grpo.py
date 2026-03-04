#!/usr/bin/env python3
"""
Train with Group Relative Policy Optimization (GRPO).

GRPO generates multiple candidate responses per prompt, scores them with a
soundness-based reward function, and uses group-relative advantages to update
the policy without a learned value network.

Data format (JSONL, one object per line):
    {"prompt": "<PREMISE> fact1 </PREMISE> <PREMISE> fact2 </PREMISE>", "target": "expected conclusion"}

Examples:
    # Basic GRPO training from SFT checkpoint
    python scripts/train_grpo.py \\
        --train-path ./data/logic_train.jsonl \\
        --lora-adapter ./outputs/sft/final

    # Full configuration
    python scripts/train_grpo.py \\
        --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \\
        --train-path ./data/logic_train.jsonl \\
        --lora-adapter ./outputs/sft/final \\
        --group-size 8 \\
        --kl-coeff 0.04 \\
        --lr 5e-6 \\
        --epochs 5 \\
        --batch-size 4 \\
        --output-dir ./outputs/grpo \\
        --task-type proof \\
        --bf16

    # Lightweight run with 4-bit quantisation
    python scripts/train_grpo.py \\
        --train-path ./data/logic_train.jsonl \\
        --4bit --batch-size 1 --group-size 2
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on ``sys.path`` so that ``src.*`` imports resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.grpo import GRPOConfig, GRPOTrainer

logger = logging.getLogger("train_grpo")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load training data from a JSONL file.

    Each line must be a JSON object with at least ``"prompt"`` and ``"target"``
    keys.  Lines that cannot be parsed are skipped with a warning.
    """
    data: List[Dict[str, Any]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: %s", line_no, exc)
                skipped += 1
                continue

            if "prompt" not in obj or "target" not in obj:
                logger.warning(
                    "Skipping line %d: missing 'prompt' or 'target' key", line_no
                )
                skipped += 1
                continue

            data.append(obj)

    if skipped > 0:
        logger.warning("Skipped %d malformed lines from %s", skipped, path)

    return data


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GRPO training."""
    parser = argparse.ArgumentParser(
        description="Train with Group Relative Policy Optimization (GRPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Data format (JSONL, one object per line):\n"
            '  {"prompt": "<PREMISE> ... </PREMISE>", "target": "expected conclusion"}\n'
        ),
    )

    # -- Model ----------------------------------------------------------------
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        help="Model key from the registry or a HuggingFace model path (default: Mistral Small 24B Instruct)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Path to an existing SFT LoRA adapter to initialise from",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )

    # -- Data -----------------------------------------------------------------
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training data (JSONL with prompt/target pairs)",
    )

    # -- Output ---------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/grpo",
        help="Directory to save checkpoints and final model (default: ./outputs/grpo)",
    )

    # -- GRPO hyper-parameters ------------------------------------------------
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of sampled responses per prompt (default: 4)",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO-style clipping range (default: 0.2)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.05,
        help="KL penalty coefficient against the reference policy (default: 0.05)",
    )

    # -- Training parameters --------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of prompts per optimisation step (default: 2)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    # -- Generation -----------------------------------------------------------
    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Maximum prompt token length (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )

    # -- Reward / task --------------------------------------------------------
    parser.add_argument(
        "--task-type",
        type=str,
        default="free_form",
        choices=["free_form", "multiple_choice", "proof", "verdict"],
        help=(
            "Task type passed to the reward function (default: free_form). "
            "'verdict' matches FOLIO-style 'Verdict: True/False/Unknown' outputs."
        ),
    )
    parser.add_argument(
        "--verifier-weight",
        type=float,
        default=0.3,
        help=(
            "Weight for the verifier (process) reward component. "
            "Process reward = log(1 + sound_steps). (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.7,
        help=(
            "Weight for the outcome (correct answer) reward component (default: 0.7)"
        ),
    )
    parser.add_argument(
        "--correct-reward",
        type=float,
        default=1.0,
        help="Outcome signal for a correct answer (default: 1.0)",
    )
    parser.add_argument(
        "--wrong-reward",
        type=float,
        default=-1.0,
        help="Outcome signal for a wrong answer — negative applies a penalty (default: -1.0)",
    )

    # -- Speed / attention -------------------------------------------------------
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention 2 (falls back to SDPA). Flash Attention is on by default.",
    )
    parser.add_argument(
        "--skip-z3-verify",
        action="store_true",
        help=(
            "Skip per-step Z3 soundness verification and treat all proof steps as sound "
            "(optimistic process reward = 1.0). Removes synchronous Z3 blocking."
        ),
    )
    parser.add_argument(
        "--n-verify-workers",
        type=int,
        default=16,
        help=(
            "Number of worker threads for parallel Z3 segment verification. "
            "Z3's C extension releases the GIL, so threads give true parallelism. "
            "Set to 0 for sequential (legacy) verification. (default: 16)"
        ),
    )

    # -- Precision / memory ---------------------------------------------------
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 mixed precision (default)",
    )
    precision_group.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision (overrides FP16)",
    )
    parser.add_argument(
        "--4bit",
        dest="use_4bit",
        action="store_true",
        help="Load model in 4-bit quantisation (requires bitsandbytes)",
    )
    parser.add_argument(
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Load model in 8-bit quantisation (requires bitsandbytes)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    # -- Misc -----------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save a checkpoint every N steps (default: 200)",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=50,
        help=(
            "Generate and log qualitative sample completions every N steps "
            "(0 = disabled, default: 50)"
        ),
    )
    parser.add_argument(
        "--n-sample-prompts",
        type=int,
        default=2,
        help="Number of prompts to sample per qualitative logging event (default: 2)",
    )

    # -- W&B ------------------------------------------------------------------
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name. If omitted, W&B logging is disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated by wandb)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username). Defaults to your wandb default.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for GRPO training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # -- Validate inputs ------------------------------------------------------
    train_path = Path(args.train_path)
    if not train_path.exists():
        logger.error("Training file not found: %s", train_path)
        sys.exit(1)

    # -- Load data ------------------------------------------------------------
    logger.info("Loading training data from %s", train_path)
    train_data = load_jsonl(str(train_path))

    if not train_data:
        logger.error("No valid training examples found in %s", train_path)
        sys.exit(1)

    logger.info("Loaded %d training examples", len(train_data))

    # -- Build config ---------------------------------------------------------
    config = GRPOConfig(
        model_name=args.model,
        lora_adapter_path=args.lora_adapter,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        max_gen_length=args.max_gen_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        sample_steps=args.sample_steps,
        n_sample_prompts=args.n_sample_prompts,
        seed=args.seed,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        temperature=args.temperature,
        task_type=args.task_type,
        verifier_weight=args.verifier_weight,
        outcome_weight=args.outcome_weight,
        correct_reward=args.correct_reward,
        wrong_reward=args.wrong_reward,
        skip_z3_verify=args.skip_z3_verify,
        use_flash_attention=not args.no_flash_attn,
        n_verify_workers=args.n_verify_workers,
    )

    # -- Log configuration summary --------------------------------------------
    logger.info("=== GRPO Training Configuration ===")
    logger.info("  model:          %s", config.model_name)
    logger.info("  lora_adapter:   %s", config.lora_adapter_path or "(none -- fresh LoRA)")
    logger.info("  group_size:     %d", config.group_size)
    logger.info("  clip_epsilon:   %.3f", config.clip_epsilon)
    logger.info("  kl_coeff:       %.4f", config.kl_coeff)
    logger.info("  lr:             %.2e", config.learning_rate)
    logger.info("  epochs:         %d", config.num_epochs)
    logger.info("  batch_size:     %d", config.batch_size)
    logger.info("  grad_accum:     %d", config.gradient_accumulation_steps)
    logger.info("  temperature:    %.2f", config.temperature)
    logger.info("  task_type:      %s", config.task_type)
    logger.info("  verifier_weight: %.2f", config.verifier_weight)
    logger.info("  outcome_weight:  %.2f", config.outcome_weight)
    logger.info("  correct_reward:  %.2f", config.correct_reward)
    logger.info("  wrong_reward:    %.2f", config.wrong_reward)
    logger.info("  output_dir:     %s", config.output_dir)
    precision = "bf16" if config.bf16 else ("fp16" if config.fp16 else "fp32")
    if config.use_4bit:
        precision += "+4bit"
    elif config.use_8bit:
        precision += "+8bit"
    logger.info("  precision:      %s", precision)
    logger.info("  flash_attn:     %s", config.use_flash_attention)
    logger.info("  skip_z3:        %s", config.skip_z3_verify)
    logger.info("  n_verify_workers: %d", config.n_verify_workers)
    logger.info("===================================")

    # -- W&B init -------------------------------------------------------------
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={
                    "model": config.model_name,
                    "lora_rank": config.lora_rank,
                    "lora_alpha": config.lora_alpha,
                    "group_size": config.group_size,
                    "clip_epsilon": config.clip_epsilon,
                    "kl_coeff": config.kl_coeff,
                    "learning_rate": config.learning_rate,
                    "num_epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_gen_length": config.max_gen_length,
                    "max_prompt_length": config.max_prompt_length,
                    "temperature": config.temperature,
                    "task_type": config.task_type,
                    "verifier_weight": config.verifier_weight,
                    "outcome_weight": config.outcome_weight,
                    "correct_reward": config.correct_reward,
                    "wrong_reward": config.wrong_reward,
                    "skip_z3_verify": config.skip_z3_verify,
                    "use_flash_attention": config.use_flash_attention,
                    "seed": config.seed,
                },
            )
            logger.info("W&B run initialised: %s/%s", args.wandb_project, wandb.run.name)
        except ImportError:
            logger.warning("wandb not installed -- W&B logging disabled. Install with: pip install wandb")

    # -- Run training ---------------------------------------------------------
    trainer = GRPOTrainer(config)
    final_path = trainer.train(train_data)

    logger.info("Training complete. Final model: %s", final_path)

    # -- W&B finish -----------------------------------------------------------
    if args.wandb_project:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
