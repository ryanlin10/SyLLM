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
import os
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path so ``src`` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.chain_generator import ChainGenerator, ChainGeneratorConfig
from src.data.atomic_proposition_generator import (
    AtomicPropositionGenerator,
    GeneratorConfig,
    PropositionPool,
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


def _divide_work(total: int, num_workers: int) -> List[int]:
    """Split *total* items into *num_workers* batch sizes."""
    base, remainder = divmod(total, num_workers)
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


def _generate_batch(
    chain_config_dict: Dict[str, Any],
    pool_dict: Dict[str, Any],
    batch_size: int,
    seed: int,
    stage0: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Worker function executed in a subprocess.

    Reconstructs the generator and pool from dicts (Z3 C bindings are
    per-process), generates *batch_size* examples, and returns serialized
    annotations plus failure stats.
    """
    # Ensure project root is importable inside the subprocess.
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.chain_generator import ChainGenerator, ChainGeneratorConfig
    from src.data.atomic_proposition_generator import PropositionPool
    from src.data.schema import Annotation, Premise

    random.seed(seed)

    config = ChainGeneratorConfig(**chain_config_dict)
    generator = ChainGenerator(config)
    pool = PropositionPool.from_dict(pool_dict)

    results: List[Dict[str, Any]] = []
    failures = 0

    for _ in range(batch_size):
        try:
            chain = generator.generate()
            rendered = generator.render(chain, pool)

            if stage0:
                premise_objects = [
                    {"id": p["id"], "text": p["text"]}
                    for p in rendered["essential_premises"]
                ]
            else:
                premise_objects = [
                    {"id": p["id"], "text": p["text"]}
                    for p in rendered["premises"]
                ]

            if stage0:
                parts = []
                for p in premise_objects:
                    parts.append(f"<PREMISE> {p['text']} </PREMISE>")
                parts.append(
                    f"<CONCLUSION> {rendered['conclusion']} </CONCLUSION>"
                )
                content_field = "\n".join(parts)
            else:
                content_field = rendered["proof_trace"]

            results.append({
                "id": rendered["id"],
                "premises": premise_objects,
                "content": content_field,
                "verifier_notes": rendered["verifier_notes"],
                "annotator_id": rendered["annotator_id"],
                "timestamp": rendered["timestamp"],
            })
        except Exception:
            failures += 1

    stats = generator.failure_stats
    stats["failures"] = failures
    return results, stats


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
        "--pool-cache",
        type=str,
        default=None,
        help="Path to cache the proposition pool as JSON.  If the file "
             "exists, the pool is loaded from it (skipping API calls); "
             "otherwise it is generated and saved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for chain generation.  "
             "0 = use all CPU cores, 1 = sequential (default: 1).",
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
        if args.pool_cache:
            print(f"Pool cache:      {args.pool_cache}")
        num_workers = args.workers if args.workers > 0 else os.cpu_count()
        print(f"Workers:         {num_workers}")
        print()

    # Build proposition pool (with optional caching).
    if args.pool_cache and Path(args.pool_cache).exists():
        if not args.quiet:
            print(f"Loading pool from cache: {args.pool_cache}")
        pool = PropositionPool.load(args.pool_cache)
    else:
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
        except Exception as exc:
            print(f"Error during pool generation: {type(exc).__name__}: {exc}")
            print("Hint: use --use-fallback to skip API calls.")
            sys.exit(1)

        # Save pool to cache if requested.
        if args.pool_cache:
            pool.save(args.pool_cache)
            if not args.quiet:
                print(f"Saved pool cache to: {args.pool_cache}")

    # Configure chain generator.
    chain_config = ChainGeneratorConfig(
        min_chain_length=args.min_chain,
        max_chain_length=args.max_chain,
        max_compression=args.max_compression,
        stage0=args.stage0,
        allow_not_intro=not args.stage0,
        allow_not_elim_final=not args.stage0,
    )

    # Determine effective worker count.
    num_workers = args.workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 1
    use_parallel = num_workers > 1

    # Serialize config and pool for cross-process transfer.
    from dataclasses import asdict
    chain_config_dict = asdict(chain_config)
    pool_dict = pool.to_dict()

    # Generate examples (parallel or sequential).
    annotations: List[Annotation] = []
    failures = 0
    aggregated_stats: Dict[str, int] = {
        "total_attempts": 0,
        "successes": 0,
        "z3_backstop_rejections": 0,
        "exceptions": 0,
    }

    if use_parallel:
        batch_sizes = _divide_work(args.num_examples, num_workers)
        if not args.quiet:
            print(f"Dispatching {args.num_examples} examples across "
                  f"{num_workers} workers (batches: {batch_sizes})...")

        try:
            completed = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _generate_batch,
                        chain_config_dict,
                        pool_dict,
                        bs,
                        args.seed + wi,
                        args.stage0,
                    ): wi
                    for wi, bs in enumerate(batch_sizes)
                    if bs > 0
                }
                for future in as_completed(futures):
                    wi = futures[future]
                    batch_results, batch_stats = future.result()
                    for r in batch_results:
                        ann = Annotation(
                            id=r["id"],
                            premises=[Premise(id=p["id"], text=p["text"])
                                      for p in r["premises"]],
                            content=r["content"],
                            verifier_notes=r["verifier_notes"],
                            annotator_id=r["annotator_id"],
                            timestamp=r["timestamp"],
                        )
                        annotations.append(ann)
                    failures += batch_stats.get("failures", 0)
                    for k in aggregated_stats:
                        aggregated_stats[k] += batch_stats.get(k, 0)
                    completed += batch_sizes[wi]
                    if not args.quiet:
                        print(f"  Worker {wi} done — "
                              f"{completed}/{args.num_examples} total")
        except Exception as exc:
            if not args.quiet:
                print(f"  Parallel generation failed ({exc}), "
                      f"falling back to sequential...")
            # Reset and fall through to sequential.
            annotations = []
            failures = 0
            aggregated_stats = {k: 0 for k in aggregated_stats}
            use_parallel = False

    if not use_parallel:
        # Sequential generation (single-process).
        generator = ChainGenerator(chain_config)
        random.seed(args.seed)

        for i in range(args.num_examples):
            try:
                chain = generator.generate()
                rendered = generator.render(chain, pool)

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

        aggregated_stats = generator.failure_stats

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

        # Detailed failure breakdown.
        print(f"\nGeneration statistics:")
        print(f"  Total attempts:               {aggregated_stats['total_attempts']}")
        print(f"  Successes:                    {aggregated_stats['successes']}")
        print(f"  Z3 backstop rejections:       {aggregated_stats['z3_backstop_rejections']}")
        print(f"  Exceptions:                   {aggregated_stats['exceptions']}")

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
