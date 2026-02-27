#!/usr/bin/env python3
"""Generate multi-step proof chain training data.

Builds bottom-up natural-deduction proof chains, renders them to natural
language using the proposition pool, and writes Annotation-compatible JSONL
output.

Training stages
---------------
**Stage 0** (``--stage0``): Premises + final conclusion only — no intermediate
reasoning steps.  The ``content`` field contains each initial premise wrapped
in ``<PREMISE>`` tags followed by the final conclusion in a ``<CONCLUSION>``
tag.  Train with **conclusion-only loss** (the default in ``lora_finetune.py``)
so the model sees premises as context and learns to predict the final
conclusion directly.  This teaches complex multi-inference reasoning patterns
without requiring the model to reproduce intermediate derivations.

    Example content::

        <PREMISE> {if it rains, then the ground is wet} </PREMISE>
        <PREMISE> it rains </PREMISE>
        <PREMISE> {if the ground is wet, then the grass grows} </PREMISE>
        <CONCLUSION> the grass grows </CONCLUSION>

**Stage 1** (default): Full proof trace with per-step ``<PREMISE>``/
``<CONCLUSION>``/``<ASSUME>``/``<DISCHARGE>`` tag groups showing every
intermediate inference.  Train with **all-token loss** (``--train-on-all``)
so the model learns to write correct semi-formal bracket notation and
multi-step chain structure by reproducing the entire proof trace.

Workflow
--------
1. Generate stage 0 data and fine-tune with conclusion-only loss to teach
   the model *what* to conclude from a set of premises.
2. Generate stage 1 data and fine-tune with all-token loss to teach the
   model *how* to write structured proof traces.

Usage examples::

    # Stage 0: premises + conclusion (no intermediate steps)
    python scripts/generate_chain_data.py --stage0 -n 5000 -o ./data/chain_stage0.jsonl

    # Stage 1: full proof trace (default)
    python scripts/generate_chain_data.py -n 5000 -o ./data/chain_stage1.jsonl

    # Auto-named output (omit -o to get timestamped filenames)
    python scripts/generate_chain_data.py --use-fallback --stage0 -n 50
    # => ./data/chain_stage0_n50_len2-5_comp1_20260220_143052.jsonl

    python scripts/generate_chain_data.py --use-fallback -n 50 --seed 42
    python scripts/generate_chain_data.py --min-chain 3 --max-chain 6 --max-compression 2
"""

import sys
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path so ``src`` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.chain_generator import ChainGenerator, ChainGeneratorConfig
from src.data.atomic_proposition_generator import (
    AtomicPropositionGenerator,
    GeneratorConfig,
    create_fallback_pool,
)
from src.data.schema import Annotation, Premise


def build_pool(
    *,
    use_fallback: bool,
    api_key: Optional[str],
    model: str,
    quiet: bool,
    num_examples: int = 100,
):
    """Create a proposition pool (API or fallback).

    Pool sizes scale with *num_examples* so larger datasets get more
    diverse propositions, predicates, relations, and entity names.
    Skips the category API call since the chain generator does not use
    categories (they are only needed by the template-based inference
    generator).
    """
    if use_fallback:
        if not quiet:
            print("Using fallback pool (no API calls)...")
        return create_fallback_pool()

    # Scale pool sizes to the dataset size.  The proposition and predicate
    # prompts request (per_topic * 5 topics) items, so divide by 5.
    config = GeneratorConfig(
        api_key=api_key,
        model=model,
        propositions_per_topic=max(20, -(-num_examples // 5)),
        predicates_per_topic=max(15, -(-num_examples // 5)),
        relations_count=max(30, num_examples),
        entities_count=max(50, num_examples),
    )
    gen = AtomicPropositionGenerator(config)
    return gen.generate_pool(verbose=not quiet, skip_categories=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-step proof chain training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training stages:
  Stage 0 (--stage0)  Premises + final conclusion only.  Train with
                      conclusion-only loss (default in lora_finetune.py).
  Stage 1 (default)   Full proof trace with intermediate steps.  Train
                      with --train-on-all for loss on every token.

Examples:
  # Stage 0 — premises + final conclusion, no intermediate steps
  python scripts/generate_chain_data.py --use-fallback --stage0 -n 5000

  # Stage 1 — full proof trace (default)
  python scripts/generate_chain_data.py --use-fallback -n 5000

  # Longer chains with compression
  python scripts/generate_chain_data.py --min-chain 3 --max-chain 6 --max-compression 2 -n 200

  # Explicit output path (disables auto-naming)
  python scripts/generate_chain_data.py -n 500 -o ./data/chain_v2.jsonl --seed 123

  # Auto-named output (omit -o):
  #   ./data/chain_stage0_n500_len2-5_comp1_20260220_143052.jsonl
""",
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSONL file path.  If omitted, auto-generates a "
             "timestamped filename under ./data/ that includes the stage, "
             "example count, chain length range, and compression level.",
    )
    parser.add_argument(
        "--min-chain",
        type=int,
        default=2,
        help="Minimum proof chain length (default: 2)",
    )
    parser.add_argument(
        "--max-chain",
        type=int,
        default=5,
        help="Maximum proof chain length (default: 5)",
    )
    parser.add_argument(
        "--max-compression",
        type=int,
        default=1,
        help="Max steps to compress into one segment; 1 = no compression (default: 1)",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback proposition pool (no API calls)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="OpenAI model for pool generation (default: gpt-5.2-2025-12-11)",
    )
    parser.add_argument(
        "--stage0",
        action="store_true",
        help="Stage 0 mode: output only premises + final conclusion (no proof "
             "trace).  Train with --train-on-all for loss on all tokens.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Auto-generate output filename when the user doesn't supply -o.
    if args.output is None:
        stage_tag = "stage0" if args.stage0 else "stage1"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            f"./data/chain_{stage_tag}"
            f"_n{args.num_examples}"
            f"_len{args.min_chain}-{args.max_chain}"
            f"_comp{args.max_compression}"
            f"_{ts}.jsonl"
        )

    # Reproducibility.
    random.seed(args.seed)

    # Header.
    stage_label = "Stage 0 (premises + conclusion)" if args.stage0 else "Stage 1 (proof trace)"
    if not args.quiet:
        print("=" * 60)
        print("Proof Chain Training Data Generator")
        print("=" * 60)
        print(f"Mode:            {stage_label}")
        print(f"Seed:            {args.seed}")
        print(f"Examples:        {args.num_examples}")
        print(f"Chain length:    {args.min_chain} - {args.max_chain}")
        print(f"Max compression: {args.max_compression}")
        print(f"Pool:            {'fallback' if args.use_fallback else args.model}")
        print()

    # Build proposition pool.
    try:
        pool = build_pool(
            use_fallback=args.use_fallback,
            api_key=args.api_key,
            model=args.model,
            quiet=args.quiet,
            num_examples=args.num_examples,
        )
    except (ImportError, ValueError) as exc:
        if not args.use_fallback:
            print(f"Error creating pool: {exc}")
            print("Hint: use --use-fallback to skip API calls.")
            sys.exit(1)
        pool = create_fallback_pool()

    # Configure chain generator.
    chain_config = ChainGeneratorConfig(
        min_chain_length=args.min_chain,
        max_chain_length=args.max_chain,
        max_compression=args.max_compression,
        stage0=args.stage0,
        allow_not_intro=not args.stage0,
        allow_not_elim_final=not args.stage0,
    )
    generator = ChainGenerator(chain_config)

    # Generate examples.
    annotations: List[Annotation] = []
    failures = 0

    for i in range(args.num_examples):
        try:
            chain = generator.generate()
            rendered = generator.render(chain, pool)

            # Stage 0: use only the essential (backward-traced, deduplicated)
            # premises needed for the final conclusion.
            # Stage 1: use all undischarged premises (full proof context).
            if args.stage0:
                premise_objects = [
                    Premise(id=p["id"], text=p["text"])
                    for p in rendered["essential_premises"]
                ]
            else:
                premise_objects = [
                    Premise(id=p["id"], text=p["text"])
                    for p in rendered["premises"]
                ]

            # Stage 0: tagged premises + final conclusion — no intermediate
            # steps.  The model sees the premises as context and learns to
            # predict only the final conclusion.
            # Stage 1 (default): full proof trace with per-step
            # <PREMISE>/<CONCLUSION> groups including intermediate inferences.
            if args.stage0:
                parts = []
                for p in premise_objects:
                    parts.append(f"<PREMISE> {p.text} </PREMISE>")
                parts.append(
                    f"<CONCLUSION> {rendered['conclusion']} </CONCLUSION>"
                )
                content_field = "\n".join(parts)
            else:
                content_field = rendered["proof_trace"]

            annotation = Annotation(
                id=rendered["id"],
                premises=premise_objects,
                content=content_field,
                verifier_notes=rendered["verifier_notes"],
                annotator_id=rendered["annotator_id"],
                timestamp=rendered["timestamp"],
            )
            annotations.append(annotation)

            if not args.quiet and (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{args.num_examples} examples")

        except Exception as exc:
            failures += 1
            if not args.quiet and failures <= 5:
                print(f"  Warning: example {i + 1} failed: {exc}")

    # Shuffle.
    random.shuffle(annotations)

    # Write output.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for ann in annotations:
            fh.write(ann.to_jsonl() + "\n")

    if not args.quiet:
        print()
        print("=" * 60)
        print("Generation Complete")
        print("=" * 60)
        print(f"Total examples:  {len(annotations)}")
        print(f"Failures:        {failures}")
        print(f"Output:          {args.output}")

        # Detailed failure breakdown from generator.
        stats = generator.failure_stats
        print(f"\nGeneration statistics:")
        print(f"  Total attempts:               {stats['total_attempts']}")
        print(f"  Successes:                    {stats['successes']}")
        print(f"  Z3 backstop rejections:       {stats['z3_backstop_rejections']}")
        print(f"  Exceptions:                   {stats['exceptions']}")

        # Summary statistics.
        rule_counts: dict = {}
        order_counts = {"propositional": 0, "first_order": 0}
        chain_lengths: List[int] = []

        for ann in annotations:
            if ann.verifier_notes:
                notes = json.loads(ann.verifier_notes)
                for rule in notes.get("rules_used", []):
                    rule_counts[rule] = rule_counts.get(rule, 0) + 1
                order = notes.get("logic_order", "propositional")
                order_counts[order] = order_counts.get(order, 0) + 1
                chain_lengths.append(notes.get("chain_length", 0))

        print(f"\nBy logic order:")
        print(f"  Propositional:  {order_counts.get('propositional', 0)}")
        print(f"  First-order:    {order_counts.get('first_order', 0)}")

        if chain_lengths:
            avg_len = sum(chain_lengths) / len(chain_lengths)
            print(f"\nChain length:  avg={avg_len:.1f}  "
                  f"min={min(chain_lengths)}  max={max(chain_lengths)}")

        if rule_counts:
            print(f"\nRules used:")
            for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
                print(f"  {rule}: {count}")

        # Sample output.
        if annotations:
            sample = annotations[0]
            print(f"\nSample output:")
            print(f"  Premises:    {[p.text for p in sample.premises]}")
            print(f"  Content:     {sample.content}")


if __name__ == "__main__":
    main()
