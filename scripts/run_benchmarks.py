#!/usr/bin/env python3
"""
Comprehensive Benchmark Evaluation Pipeline.

Evaluates and compares base model vs LoRA-finetuned model on standard
and logic-specific benchmarks.

Usage:
    # Run all benchmarks
    python scripts/run_benchmarks.py

    # Run specific benchmarks
    python scripts/run_benchmarks.py --benchmarks mmlu arc_challenge logiqa

    # Quick test with limited samples
    python scripts/run_benchmarks.py --max-samples 50

    # Standard or logic only
    python scripts/run_benchmarks.py --standard-only
    python scripts/run_benchmarks.py --logic-only

    # List available benchmarks
    python scripts/run_benchmarks.py --list

    # Custom output path
    python scripts/run_benchmarks.py --output ./results/comparison.txt
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark_registry import (
    BENCHMARK_REGISTRY,
    BenchmarkCategory,
    list_benchmarks,
)

# All categories that --standard-only should include
_STANDARD_CATEGORIES = {BenchmarkCategory.STANDARD}
from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.evaluation.report_generator import ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation: base vs finetuned model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmarks.py --list
  python scripts/run_benchmarks.py --benchmarks logiqa folio --max-samples 50
  python scripts/run_benchmarks.py --standard-only
  python scripts/run_benchmarks.py --output ./results/my_comparison.txt
        """,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--model", "-m", type=str, help="Base model path (overrides config.yaml)"
    )
    model_group.add_argument(
        "--lora", "-l", type=str, help="LoRA adapter path (overrides config.yaml)"
    )
    model_group.add_argument(
        "--tensor-parallel", "-tp", type=int, default=1, help="Number of GPUs"
    )
    model_group.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9, help="GPU memory fraction"
    )
    model_group.add_argument(
        "--max-model-len", type=int, default=None, help="Max sequence length"
    )

    # Benchmark selection
    bench_group = parser.add_argument_group("Benchmark selection")
    bench_group.add_argument(
        "--benchmarks", "-b", nargs="+", help="Specific benchmarks to run"
    )
    bench_group.add_argument(
        "--list", action="store_true", help="List available benchmarks and exit"
    )
    bench_group.add_argument(
        "--standard-only", action="store_true", help="Run only standard benchmarks"
    )
    bench_group.add_argument(
        "--logic-only", action="store_true", help="Run only logic benchmarks"
    )
    bench_group.add_argument(
        "--code-only", action="store_true", help="Run only code benchmarks (HumanEval, MBPP)"
    )

    # Evaluation parameters
    eval_group = parser.add_argument_group("Evaluation parameters")
    eval_group.add_argument(
        "--max-samples", type=int, help="Max samples per benchmark (for quick testing)"
    )
    eval_group.add_argument(
        "--num-few-shot", type=int, help="Override default few-shot count"
    )
    eval_group.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy)"
    )
    eval_group.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate per item"
    )
    eval_group.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size"
    )

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output text file path (default: outputs/benchmark_comparison_<timestamp>.txt)",
    )
    out_group.add_argument(
        "--json-output", type=str, help="Also save JSON results to this path"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # List benchmarks
    if args.list:
        print("\nAvailable benchmarks:")
        print("-" * 72)
        # Group by category
        by_cat: dict = {}
        for key, cfg in BENCHMARK_REGISTRY.items():
            by_cat.setdefault(cfg.category.value, []).append((key, cfg))
        for cat_name, entries in by_cat.items():
            print(f"\n  [{cat_name.upper()}]")
            for key, cfg in entries:
                btype = cfg.benchmark_type.value
                token_info = (
                    f", max_tokens={cfg.max_tokens_override}"
                    if cfg.max_tokens_override
                    else ""
                )
                print(f"    {key:<18} {cfg.name} — {cfg.description}")
                print(
                    f"    {'':18} type={btype}, {cfg.num_few_shot}-shot"
                    f", baseline={cfg.random_baseline:.2f}{token_info}"
                )
        print()
        return

    # Load project config
    project_root = Path(__file__).parent.parent
    from src.utils.config_loader import load_config

    config = load_config(str(project_root / "config.yaml"))

    model_path = args.model or config["model"]["base_model"]
    lora_path = args.lora or config["model"].get("lora_adapter")
    if lora_path and not Path(lora_path).is_absolute():
        lora_path = str(project_root / lora_path)

    # Validate LoRA path
    if not lora_path:
        print("Error: No LoRA adapter path specified.")
        print("Set model.lora_adapter in config.yaml or use --lora <path>")
        sys.exit(1)
    if not Path(lora_path).exists():
        print(f"Error: LoRA adapter path does not exist: {lora_path}")
        print("Use --lora <path> to specify a valid adapter path")
        sys.exit(1)

    # Determine benchmarks to run
    if args.benchmarks:
        benchmarks = args.benchmarks
        # Validate
        for b in benchmarks:
            if b not in BENCHMARK_REGISTRY:
                available = ", ".join(BENCHMARK_REGISTRY.keys())
                print(f"Error: Unknown benchmark '{b}'. Available: {available}")
                sys.exit(1)
    elif args.standard_only:
        benchmarks = [
            k
            for k, v in BENCHMARK_REGISTRY.items()
            if v.category == BenchmarkCategory.STANDARD
        ]
    elif args.logic_only:
        benchmarks = [
            k
            for k, v in BENCHMARK_REGISTRY.items()
            if v.category == BenchmarkCategory.LOGIC
        ]
    elif args.code_only:
        benchmarks = [
            k
            for k, v in BENCHMARK_REGISTRY.items()
            if v.category == BenchmarkCategory.CODE
        ]
    else:
        benchmarks = list(BENCHMARK_REGISTRY.keys())

    print(f"\n{'=' * 60}")
    print("BENCHMARK EVALUATION PIPELINE")
    print(f"{'=' * 60}")
    print(f"Base model:    {model_path}")
    print(f"LoRA adapter:  {lora_path}")
    print(f"Benchmarks:    {', '.join(benchmarks)}")
    print(f"Max samples:   {args.max_samples or 'full'}")
    print(f"Temperature:   {args.temperature}")
    print(f"Batch size:    {args.batch_size}")
    print(f"{'=' * 60}")

    # Initialize predictor
    print("\nLoading model...")
    from src.inference.predictor import VLLMPredictor

    predictor = VLLMPredictor(
        model_path=model_path,
        lora_adapter_path=lora_path,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print("Model loaded.\n")

    # Run evaluation
    evaluator = BenchmarkEvaluator(
        predictor=predictor,
        benchmarks=benchmarks,
        max_samples=args.max_samples,
        num_few_shot=args.num_few_shot,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    comparisons = evaluator.evaluate_all()

    # Generate reports
    reporter = ReportGenerator(
        base_model_name=model_path,
        lora_adapter_path=lora_path or "none",
        comparisons=comparisons,
    )

    text_report = reporter.generate_text_report()

    # Determine output path
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"benchmark_comparison_{timestamp}.txt"

    reporter.save_text(str(output_path))
    print(f"\nText report saved to: {output_path}")

    # Save JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        reporter.save_json(str(json_path))
        print(f"JSON report saved to: {json_path}")

    # Print report to stdout
    print("\n" + text_report)


if __name__ == "__main__":
    main()
