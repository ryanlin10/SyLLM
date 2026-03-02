#!/usr/bin/env python3
"""Generate Stage 2 training data: semi-formal CoT on logic benchmarks.

Prompts an OpenAI model to solve FOLIO and LogiQA problems using
semi-formal curly-bracket notation interleaved with natural language
reasoning.  Only examples with correct final answers are kept.

Usage examples::

    # Generate from both benchmarks (default), no Z3 verification
    python scripts/generate_stage2_data.py --no-verify -o ./data/stage2_train.jsonl

    # FOLIO only, capped at 20 problems
    python scripts/generate_stage2_data.py \\
        --benchmarks folio \\
        --max-samples 20 \\
        --no-verify \\
        -o ./data/stage2_folio.jsonl

    # Full run with Z3 verification
    python scripts/generate_stage2_data.py --verify -o ./data/stage2_train.jsonl
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List

# Add project root to path so ``src`` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.stage2_generator import (
    Stage2Generator,
    Stage2Config,
    Stage2Example,
)


SUPPORTED_BENCHMARKS = ["folio", "logiqa"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 semi-formal CoT training data from logic benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported benchmarks: "
            + ", ".join(SUPPORTED_BENCHMARKS)
        ),
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="folio,logiqa",
        help=(
            "Comma-separated list of benchmarks to use "
            "(default: folio,logiqa)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of problems per benchmark (default: all).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        dest="verify",
        help="Enable Z3 verification of formal steps.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Disable Z3 verification (default).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Output JSONL file path. "
            "Auto-named as ./data/stage2_train.jsonl if omitted."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="OpenAI model to use (default: gpt-5.2-2025-12-11).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages.",
    )

    args = parser.parse_args()

    # -- Resolve benchmarks -----------------------------------------------
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in benchmarks:
        if b not in SUPPORTED_BENCHMARKS:
            print(
                f"Error: unsupported benchmark '{b}'. "
                f"Supported: {', '.join(SUPPORTED_BENCHMARKS)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- Resolve output path ----------------------------------------------
    output_path = args.output or "./data/stage2_train.jsonl"

    # -- Build config & generator -----------------------------------------
    config = Stage2Config(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=4096,
        verify_with_z3=args.verify,
    )

    try:
        generator = Stage2Generator(config)
    except (ImportError, ValueError) as exc:
        print(f"Error initializing generator: {exc}", file=sys.stderr)
        print(
            "\nEnsure the openai package is installed and "
            "OPENAI_API_KEY is set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Header -----------------------------------------------------------
    if not args.quiet:
        print("=" * 60)
        print("Stage 2: Semi-Formal CoT from Logic Benchmarks")
        print("=" * 60)
        print(f"Model           : {args.model}")
        print(f"Benchmarks      : {', '.join(benchmarks)}")
        print(f"Max samples     : {args.max_samples or 'all'}")
        print(f"Temperature     : {args.temperature}")
        print(f"Z3 verification : {'enabled' if args.verify else 'disabled'}")
        print(f"Output          : {output_path}")
        print()

    # -- Generation -------------------------------------------------------
    all_examples: List[Stage2Example] = generator.generate(
        benchmarks=benchmarks,
        max_samples=args.max_samples,
    )

    # -- Write JSONL -------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fout:
        for ex in all_examples:
            fout.write(json.dumps(ex.to_training_dict(), ensure_ascii=False) + "\n")

    # -- Summary ----------------------------------------------------------
    if not args.quiet:
        # Per-benchmark stats
        bench_stats: dict = {}
        for ex in all_examples:
            stats = bench_stats.setdefault(
                ex.benchmark, {"total": 0, "verified": 0}
            )
            stats["total"] += 1
            if ex.verification_status:
                stats["verified"] += 1

        total = len(all_examples)
        total_verified = sum(s["verified"] for s in bench_stats.values())

        print()
        print("=" * 60)
        print("Generation Complete")
        print("=" * 60)
        print(f"Total examples written : {total}")
        if args.verify:
            print(f"Total Z3-verified      : {total_verified}")
        print()
        print("By benchmark:")
        for bench, stats in bench_stats.items():
            line = f"  {bench}: {stats['total']} examples"
            if args.verify:
                line += f" ({stats['verified']} verified)"
            print(line)
        print(f"\nOutput: {output_path}")

        # Show a sample
        if all_examples:
            sample_ex = all_examples[0]
            print("\nSample output:")
            print(f"  Benchmark  : {sample_ex.benchmark}")
            print(f"  Item ID    : {sample_ex.item_id}")
            print(f"  Answer     : {sample_ex.predicted_answer}")
            print(f"  Reasoning  : {sample_ex.reasoning[:150]}...")


if __name__ == "__main__":
    main()
