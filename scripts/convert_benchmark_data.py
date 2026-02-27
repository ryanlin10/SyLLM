#!/usr/bin/env python3
"""Convert logic reasoning benchmarks to RL training format.

Converts FOLIO, ProofWriter, and ProntoQA datasets into the prompt/target
JSONL format used by GRPO training.

Output format (one JSON object per line):
    {"prompt": "<PREMISE> premise1 </PREMISE> <PREMISE> premise2 </PREMISE>", "target": "conclusion text"}

Usage:
    python scripts/convert_benchmark_data.py --benchmarks folio,proofwriter -o data/rl_train.jsonl
    python scripts/convert_benchmark_data.py --benchmarks prontoqa --split validation -o data/rl_val.jsonl
    python scripts/convert_benchmark_data.py --benchmarks folio --split train -o data/rl_train.jsonl --quiet
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path so we can import src modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_BENCHMARKS = ["folio", "proofwriter", "prontoqa"]


def _wrap_premise(text: str) -> str:
    """Wrap a single premise string in <PREMISE> tags."""
    return f"<PREMISE> {text.strip()} </PREMISE>"


def _split_on_periods(text: str) -> List[str]:
    """Split a text block into sentences on period boundaries.

    Handles trailing periods, multiple periods, and whitespace gracefully.
    Returns a list of non-empty stripped sentence strings.
    """
    parts = text.split(".")
    return [p.strip() for p in parts if p.strip()]


class BenchmarkConverter:
    """Converts logic reasoning benchmarks into prompt/target JSONL records."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # FOLIO
    # ------------------------------------------------------------------
    def convert_folio(self, split: str = "validation") -> List[Dict[str, str]]:
        """Convert the FOLIO dataset (yale-nlp/FOLIO).

        Each example contains:
          - premises: a string with premises (period-separated)
          - conclusion: a string
          - label: one of "True", "False", "Uncertain"

        The target includes the conclusion text and its truth label.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: Could not import `datasets` library. "
                  "Install with: pip install datasets")
            self.stats["folio"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        try:
            ds = load_dataset("yale-nlp/FOLIO", split=split)
        except Exception as e:
            print(f"Warning: Could not load FOLIO ({split}): {e}")
            self.stats["folio"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        examples: List[Dict[str, str]] = []
        skipped = 0

        for row in ds:
            premises_text = row.get("premises", "") or ""
            conclusion = (row.get("conclusion", "") or "").strip()
            label = (row.get("label", "") or "").strip()

            premise_parts = _split_on_periods(premises_text)

            if not premise_parts or not conclusion:
                skipped += 1
                continue

            prompt = " ".join(_wrap_premise(p) for p in premise_parts)
            target = f"{conclusion} [{label}]" if label else conclusion

            examples.append({"prompt": prompt, "target": target})

        self.stats["folio"] = {
            "total": len(ds),
            "converted": len(examples),
            "skipped": skipped,
        }
        return examples

    # ------------------------------------------------------------------
    # ProofWriter
    # ------------------------------------------------------------------
    def convert_proofwriter(self, split: str = "validation") -> List[Dict[str, str]]:
        """Convert the ProofWriter dataset (allenai/proofwriter-deduction-balanced).

        Each example contains:
          - question: a natural-language question about the theory
          - answer: "True" or "False"
          - meta: dict that may contain 'triples' and 'rules'

        The theory (premises) is reconstructed from the triples and rules
        stored in the meta field.  The target is the question with its
        answer label.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: Could not import `datasets` library. "
                  "Install with: pip install datasets")
            self.stats["proofwriter"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        try:
            ds = load_dataset(
                "allenai/proofwriter-deduction-balanced",
                "depth-5",
                split=split,
            )
        except Exception as e:
            print(f"Warning: Could not load ProofWriter ({split}): {e}")
            self.stats["proofwriter"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        examples: List[Dict[str, str]] = []
        skipped = 0

        for row in ds:
            question = (row.get("question", "") or "").strip()
            answer = (row.get("answer", "") or "").strip()

            # Build premise list from meta triples + rules
            meta = row.get("meta", {}) or {}
            premises: List[str] = []

            triples = meta.get("triples", {}) or {}
            for _key, triple_text in sorted(triples.items()):
                text = triple_text if isinstance(triple_text, str) else str(triple_text)
                text = text.strip().rstrip(".")
                if text:
                    premises.append(text)

            rules = meta.get("rules", {}) or {}
            for _key, rule_text in sorted(rules.items()):
                text = rule_text if isinstance(rule_text, str) else str(rule_text)
                text = text.strip().rstrip(".")
                if text:
                    premises.append(text)

            # Fallback: if meta had nothing useful, try the full theory field
            if not premises:
                theory_text = (row.get("theory", "") or "").strip()
                if theory_text:
                    premises = _split_on_periods(theory_text)

            if not premises or not question:
                skipped += 1
                continue

            prompt = " ".join(_wrap_premise(p) for p in premises)
            target = f"{question} [{answer}]" if answer else question

            examples.append({"prompt": prompt, "target": target})

        self.stats["proofwriter"] = {
            "total": len(ds),
            "converted": len(examples),
            "skipped": skipped,
        }
        return examples

    # ------------------------------------------------------------------
    # ProntoQA
    # ------------------------------------------------------------------
    def convert_prontoqa(self, split: str = "validation") -> List[Dict[str, str]]:
        """Convert the ProntoQA dataset.

        Each example contains:
          - context: a string with premises separated by periods
          - question: a string
          - answer: a string (e.g. "True" / "False")
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Warning: Could not import `datasets` library. "
                  "Install with: pip install datasets")
            self.stats["prontoqa"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        try:
            ds = load_dataset("ProntoQA", split=split)
        except Exception as e:
            print(f"Warning: Could not load ProntoQA ({split}): {e}")
            self.stats["prontoqa"] = {"total": 0, "converted": 0, "skipped": 0}
            return []

        examples: List[Dict[str, str]] = []
        skipped = 0

        for row in ds:
            context = (row.get("context", "") or "").strip()
            question = (row.get("question", "") or "").strip()
            answer = (row.get("answer", "") or "").strip()

            premise_parts = _split_on_periods(context)

            if not premise_parts or not question:
                skipped += 1
                continue

            prompt = " ".join(_wrap_premise(p) for p in premise_parts)
            target = f"{answer}: {question}" if answer else question

            examples.append({"prompt": prompt, "target": target})

        self.stats["prontoqa"] = {
            "total": len(ds),
            "converted": len(examples),
            "skipped": skipped,
        }
        return examples

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def convert(self, benchmark: str, split: str = "validation") -> List[Dict[str, str]]:
        """Dispatch to the appropriate converter by benchmark name."""
        converters = {
            "folio": self.convert_folio,
            "proofwriter": self.convert_proofwriter,
            "prontoqa": self.convert_prontoqa,
        }
        benchmark = benchmark.lower().strip()
        if benchmark not in converters:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Supported: {SUPPORTED_BENCHMARKS}"
            )
        return converters[benchmark](split)

    def print_stats(self) -> None:
        """Print a summary table of conversion statistics."""
        print("\n--- Conversion Summary ---")
        total_converted = 0
        for name, counts in self.stats.items():
            total = counts.get("total", 0)
            converted = counts.get("converted", 0)
            skipped = counts.get("skipped", 0)
            total_converted += converted
            print(f"  {name:15s}  total={total:5d}  converted={converted:5d}  skipped={skipped:5d}")
        print(f"  {'TOTAL':15s}  converted={total_converted:5d}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert logic reasoning benchmarks to RL training JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --benchmarks folio,proofwriter -o data/rl_train.jsonl\n"
            "  %(prog)s --benchmarks prontoqa --split validation -o data/rl_val.jsonl\n"
        ),
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help="Comma-separated list of benchmarks to convert. "
             f"Supported: {', '.join(SUPPORTED_BENCHMARKS)}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/rl_train.jsonl",
        help="Output JSONL file path (default: ./data/rl_train.jsonl)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary statistics output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse benchmark list
    benchmark_names = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    if not benchmark_names:
        print("Error: No benchmarks specified.", file=sys.stderr)
        sys.exit(1)

    for name in benchmark_names:
        if name not in SUPPORTED_BENCHMARKS:
            print(
                f"Error: Unknown benchmark '{name}'. "
                f"Supported: {SUPPORTED_BENCHMARKS}",
                file=sys.stderr,
            )
            sys.exit(1)

    converter = BenchmarkConverter()
    all_examples: List[Dict[str, str]] = []

    for name in benchmark_names:
        if not args.quiet:
            print(f"Converting {name} ({args.split})...")
        examples = converter.convert(name, split=args.split)
        all_examples.extend(examples)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if not args.quiet:
        converter.print_stats()
        print(f"Wrote {len(all_examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
