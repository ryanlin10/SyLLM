#!/usr/bin/env python3
"""Generate applied reasoning training data via OpenAI distillation.

Produces real-world, multi-step reasoning examples that use the project's
semi-formal curly-bracket notation.  Each example is optionally verified
with Z3 before being written to a JSONL training file.

Usage examples::

    # Generate across all domains (default 50 per domain, Z3-verified)
    python scripts/generate_applied_data.py -o ./data/applied_train.jsonl

    # Specific domains, more examples, no verification
    python scripts/generate_applied_data.py \\
        --domains "medical diagnosis,legal reasoning and contracts" \\
        --examples-per-domain 100 \\
        --no-verify \\
        -o ./data/applied_medical_legal.jsonl

    # Quieter output, custom model
    python scripts/generate_applied_data.py --quiet --model gpt-5.2-2025-12-11 -o out.jsonl
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List

# Add project root to path so ``src`` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.applied_chain_generator import (
    AppliedChainGenerator,
    AppliedGeneratorConfig,
    AppliedExample,
    APPLIED_DOMAINS,
)
from src.data.schema import Annotation


def _parse_domains(raw: str) -> List[str]:
    """Split a comma-separated domain string and strip whitespace."""
    return [d.strip() for d in raw.split(",") if d.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate applied reasoning training data via OpenAI distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available domains (pass a comma-separated subset via --domains):\n"
            + "\n".join(f"  - {d}" for d in APPLIED_DOMAINS)
        ),
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help=(
            "Comma-separated list of domains to generate for. "
            "Defaults to all APPLIED_DOMAINS."
        ),
    )
    parser.add_argument(
        "--examples-per-domain",
        type=int,
        default=50,
        help="Number of examples to generate per domain (default: 50).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        dest="verify",
        help="Enable Z3 verification of proof steps (default).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Disable Z3 verification.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/applied_train.jsonl",
        help="Output JSONL file path (default: ./data/applied_train.jsonl).",
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
        "--min-steps",
        type=int,
        default=2,
        help="Minimum proof-trace steps (default: 2).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum proof-trace steps (default: 5).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages.",
    )

    args = parser.parse_args()

    # -- Resolve domains --------------------------------------------------
    if args.domains:
        domains = _parse_domains(args.domains)
    else:
        domains = list(APPLIED_DOMAINS)

    # -- Build config & generator -----------------------------------------
    config = AppliedGeneratorConfig(
        api_key=args.api_key,
        model=args.model,
        max_tokens=4096,
        examples_per_domain=args.examples_per_domain,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        verify_with_z3=args.verify,
        temperature=0.7,
    )

    try:
        generator = AppliedChainGenerator(config)
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
        print("Applied Reasoning Data Generator")
        print("=" * 60)
        print(f"Model           : {args.model}")
        print(f"Domains         : {len(domains)}")
        print(f"Examples/domain : {args.examples_per_domain}")
        print(f"Proof steps     : {args.min_steps}-{args.max_steps}")
        print(f"Z3 verification : {'enabled' if args.verify else 'disabled'}")
        print(f"Output          : {args.output}")
        print()

    # -- Generation loop --------------------------------------------------
    all_annotations: List[Annotation] = []
    domain_stats: dict = {}

    for domain in domains:
        if not args.quiet:
            print(f"Generating for domain: {domain} ...")

        examples: List[AppliedExample] = generator.generate_examples(
            domain=domain,
            count=args.examples_per_domain,
        )

        verified_count = sum(1 for e in examples if e.verification_status)
        domain_stats[domain] = {
            "total": len(examples),
            "verified": verified_count,
        }

        for ex in examples:
            annotation = generator.render_to_annotation(ex)
            all_annotations.append(annotation)

        if not args.quiet:
            print(
                f"  -> {len(examples)} examples "
                f"({verified_count} verified)"
            )

    # -- Write JSONL output -----------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for ann in all_annotations:
            fout.write(ann.to_jsonl() + "\n")

    # -- Summary ----------------------------------------------------------
    if not args.quiet:
        total_examples = sum(s["total"] for s in domain_stats.values())
        total_verified = sum(s["verified"] for s in domain_stats.values())

        print()
        print("=" * 60)
        print("Generation Complete")
        print("=" * 60)
        print(f"Total examples written : {total_examples}")
        if args.verify:
            print(f"Total Z3-verified      : {total_verified}")
        print()
        print("By domain:")
        for domain, stats in domain_stats.items():
            line = f"  {domain}: {stats['total']} examples"
            if args.verify:
                line += f" ({stats['verified']} verified)"
            print(line)
        print(f"\nOutput: {args.output}")

        # Show a sample if available
        if all_annotations:
            sample = all_annotations[0]
            print("\nSample output:")
            print(f"  Premises   : {[p.text for p in sample.premises]}")
            print(f"  Content    : {sample.content[:120]}...")
            if sample.verifier_notes:
                notes = json.loads(sample.verifier_notes)
                print(f"  Domain     : {notes.get('domain', 'N/A')}")
                print(f"  Verified   : {notes.get('verification_status', 'N/A')}")


if __name__ == "__main__":
    main()
