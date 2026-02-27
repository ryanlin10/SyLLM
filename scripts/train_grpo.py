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
        choices=["free_form", "multiple_choice", "proof"],
        help="Task type passed to the reward function (default: free_form)",
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
        seed=args.seed,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        temperature=args.temperature,
        task_type=args.task_type,
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
    logger.info("  output_dir:     %s", config.output_dir)
    precision = "bf16" if config.bf16 else ("fp16" if config.fp16 else "fp32")
    if config.use_4bit:
        precision += "+4bit"
    elif config.use_8bit:
        precision += "+8bit"
    logger.info("  precision:      %s", precision)
    logger.info("===================================")

    # -- Run training ---------------------------------------------------------
    trainer = GRPOTrainer(config)
    final_path = trainer.train(train_data)

    logger.info("Training complete. Final model: %s", final_path)


if __name__ == "__main__":
    main()
