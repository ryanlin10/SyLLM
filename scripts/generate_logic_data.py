#!/usr/bin/env python3
"""
Logic Training Data Generation Pipeline

Generates synthetic logic reasoning examples using random syntax tree generation:
- Propositional logic (0th order): P ^ Q -> R
- First-order logic (1st order): forall x. Human(x) -> Mortal(x)

Uses nl_renderer.py for natural language rendering with:
- Curly brackets {} for disambiguation of compound formulas
- Explicit variable names in quantification (for all x, there exist y)
- Approved logical terms only

Atomic propositions are generated using OpenAI API (pooled approach).
Only 5 API calls are made to generate a pool, then thousands of examples
are created by sampling from this pool.

Usage:
    python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl
    python scripts/generate_logic_data.py --use-fallback -n 100  # No API calls
    python scripts/generate_logic_data.py --min-depth 2 --max-depth 4 -n 1000
    python scripts/generate_logic_data.py --inference-patterns modus_ponens hypothetical_syllogism -n 500
"""

import sys
import argparse
import json
import uuid
import random
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import Annotation, Premise
from src.data.atomic_proposition_generator import (
    AtomicPropositionGenerator, GeneratorConfig, PropositionPool,
    create_fallback_pool,
)
from src.data.syntax_tree import LogicOrder
from src.data.inference_generator import (
    InferenceGenerator, InferenceGeneratorConfig, InferencePattern,
    PROPOSITIONAL_PATTERNS, FOL_PATTERNS
)
from src.data.nl_renderer import InferenceRenderer


class LogicDataGenerator:
    """Generate logic training data using random syntax tree generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-2025-12-11",
        temperature: float = 0.9,
        use_fallback: bool = False,
        min_depth: int = 2,
        max_depth: int = 5,
        logic_order: str = "both",
        inference_patterns: Optional[List[str]] = None
    ):
        self.use_fallback = use_fallback
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pool: Optional[PropositionPool] = None
        self.prop_generator: Optional[AtomicPropositionGenerator] = None
        self.inference_generator: Optional[InferenceGenerator] = None
        self.renderer: Optional[InferenceRenderer] = None

        # Parse logic order
        self.logic_order_str = logic_order
        if logic_order == "first_order":
            self.logic_order = LogicOrder.FIRST_ORDER
        else:
            self.logic_order = LogicOrder.PROPOSITIONAL

        # Parse inference patterns
        self.patterns: Optional[List[InferencePattern]] = None
        if inference_patterns:
            self.patterns = []
            for p in inference_patterns:
                try:
                    self.patterns.append(InferencePattern(p))
                except ValueError:
                    print(f"Warning: Unknown inference pattern '{p}', skipping")
        elif logic_order == "first_order":
            self.patterns = FOL_PATTERNS
        elif logic_order == "propositional":
            self.patterns = PROPOSITIONAL_PATTERNS
        elif logic_order == "both":
            self.patterns = PROPOSITIONAL_PATTERNS + FOL_PATTERNS

        if not use_fallback:
            config = GeneratorConfig(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            self.prop_generator = AtomicPropositionGenerator(config)

    def initialize_pool(
        self,
        topics: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Initialize the proposition pool (makes API calls or uses fallback)."""
        if self.use_fallback:
            if verbose:
                print("Using fallback pool (no API calls)...")
            self.pool = create_fallback_pool()
        else:
            self.pool = self.prop_generator.generate_pool(topics=topics, verbose=verbose)

        # Initialize inference generator with logic order
        inference_config = InferenceGeneratorConfig(
            min_subformula_depth=self.min_depth,
            max_subformula_depth=self.max_depth,
            logic_order=self.logic_order,
            patterns=self.patterns
        )
        self.inference_generator = InferenceGenerator(inference_config)

        # Initialize renderer
        self.renderer = InferenceRenderer(self.pool)

    def generate_example(self) -> Optional[Annotation]:
        """Generate a single logic example using tree generation."""
        if self.pool is None:
            raise ValueError("Pool not initialized. Call initialize_pool() first.")

        # Generate inference
        inference = self.inference_generator.generate()

        # Render to natural language
        rendered = self.renderer.render_structured(
            inference.premises,
            inference.conclusion
        )

        # Create Premise objects
        premise_objects = [
            Premise(
                id=f"p{i+1}",
                text=premise_text
            )
            for i, premise_text in enumerate(rendered["premises"])
        ]

        return Annotation(
            id=str(uuid.uuid4()),
            premises=premise_objects,
            content=rendered["conclusion"],
            annotator_id=f"logic_generator_{inference.pattern.value}",
            verifier_notes=json.dumps({
                "pattern": inference.pattern.value,
                "formal_notation": inference.formal_notation,
                "full_formal": inference.to_formal(),
                "logic_order": inference.logic_order.value,
                "depth_range": [self.min_depth, self.max_depth]
            }),
            timestamp=datetime.now().isoformat()
        )

    def generate_dataset(
        self,
        num_examples: int = 1000,
        topics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> List[Annotation]:
        """
        Generate a dataset using random tree generation.

        Args:
            num_examples: Total number of examples to generate
            topics: List of topics to use for pool generation
            output_path: Path to save JSONL output
            verbose: Print progress messages

        Returns:
            List of Annotation objects
        """
        # Initialize pool first (this is where API calls happen)
        if self.pool is None:
            self.initialize_pool(topics=topics, verbose=verbose)

        annotations = []

        if verbose:
            print(f"\nGenerating {num_examples} examples...")
            print(f"Depth range: {self.min_depth} - {self.max_depth}")
            if self.patterns:
                print(f"Patterns: {[p.value for p in self.patterns]}")
            else:
                print("Patterns: all")
            print("(No additional API calls - sampling from pool)")

        for i in range(num_examples):
            try:
                annotation = self.generate_example()
                if annotation:
                    annotations.append(annotation)
                    if verbose and (i + 1) % 100 == 0:
                        print(f"  Generated {i + 1}/{num_examples} examples")
            except Exception as e:
                if verbose:
                    print(f"  Error generating example {i}: {e}")

        # Shuffle
        random.shuffle(annotations)

        # Save to JSONL
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(ann.to_jsonl() + "\n")
            if verbose:
                print(f"\nSaved {len(annotations)} examples to {output_path}")

        return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Generate logic training data using random syntax tree generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 examples (default, makes ~5 API calls for pool, then samples)
  python scripts/generate_logic_data.py -o ./data/logic_train.jsonl

  # Generate more examples
  python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl

  # Use fallback pool (no API calls at all)
  python scripts/generate_logic_data.py --use-fallback -o ./data/logic_fallback.jsonl

  # Control formula complexity
  python scripts/generate_logic_data.py --min-depth 2 --max-depth 4 -n 1000

  # Specific inference patterns only
  python scripts/generate_logic_data.py --inference-patterns modus_ponens hypothetical_syllogism -n 500

  # Only propositional logic
  python scripts/generate_logic_data.py --logic-order propositional -n 500

  # Only first-order logic
  python scripts/generate_logic_data.py --logic-order first_order -n 500

  # Custom API settings
  OPENAI_API_KEY=sk-... python scripts/generate_logic_data.py -n 1000 --model gpt-5.2-2025-12-11

Available inference patterns for --inference-patterns:
  Propositional: modus_ponens, modus_tollens, hypothetical_syllogism, disjunctive_syllogism,
                 conjunction_intro, conjunction_elim, disjunction_intro, double_negation_elim,
                 constructive_dilemma, biconditional_intro, biconditional_elim, absorption
  First-order:   universal_instantiation, universal_modus_ponens, existential_generalization,
                 universal_syllogism, universal_contraposition, existential_syllogism
"""
    )
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/logic_training_data.jsonl",
        help="Output JSONL file path (default: ./data/logic_training_data.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-2025-12-11",
        help="OpenAI model to use for pool generation (default: gpt-5.2-2025-12-11)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation temperature for pool creation (default: 0.9)"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to use for pool generation (default: all available topics)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Use fallback pool (no API calls, limited variety)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    parser.add_argument(
        "--logic-order",
        type=str,
        choices=["propositional", "first_order", "both"],
        default="both",
        help="Logic order to generate (default: both). Options: propositional (0th order), first_order (1st order), both"
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help="Minimum formula tree depth (default: 2)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum formula tree depth (default: 6)"
    )
    parser.add_argument(
        "--inference-patterns",
        nargs="+",
        default=None,
        help="Specific inference patterns to use (default: based on --logic-order)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create generator
    try:
        generator = LogicDataGenerator(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            use_fallback=args.use_fallback,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            logic_order=args.logic_order,
            inference_patterns=args.inference_patterns
        )
    except (ImportError, ValueError) as e:
        if args.use_fallback:
            generator = LogicDataGenerator(
                use_fallback=True,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                logic_order=args.logic_order,
                inference_patterns=args.inference_patterns
            )
        else:
            print(f"Error initializing generator: {e}")
            print("\nOptions:")
            print("  1. Install openai: pip install openai")
            print("  2. Set OPENAI_API_KEY environment variable or use --api-key")
            print("  3. Use --use-fallback to generate without API calls")
            sys.exit(1)

    # Print header
    if not args.quiet:
        print(f"{'='*60}")
        print("Logic Training Data Generator")
        print(f"{'='*60}")
        print(f"Logic order: {args.logic_order}")
        print(f"Seed: {args.seed}")
        print(f"Model: {args.model if not args.use_fallback else 'fallback (no API)'}")
        print(f"Examples to generate: {args.num_examples}")
        print(f"Depth range: {args.min_depth} - {args.max_depth}")
        if args.inference_patterns:
            print(f"Inference patterns: {args.inference_patterns}")
        if args.topics:
            print(f"Topics: {args.topics}")
        print()

    # Generate dataset
    annotations = generator.generate_dataset(
        num_examples=args.num_examples,
        topics=args.topics,
        output_path=args.output,
        verbose=not args.quiet
    )

    # Print summary
    if not args.quiet:
        print(f"\n{'='*60}")
        print("Generation Complete")
        print(f"{'='*60}")
        print(f"Total examples: {len(annotations)}")

        # Count by pattern and logic order
        pattern_counts = {}
        order_counts = {"propositional": 0, "first_order": 0}
        for a in annotations:
            if a.verifier_notes:
                notes = json.loads(a.verifier_notes)
                pattern = notes.get("pattern", "unknown")
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                logic_order = notes.get("logic_order", "propositional")
                order_counts[logic_order] = order_counts.get(logic_order, 0) + 1

        print(f"\nBy logic order:")
        print(f"  Propositional (0th order): {order_counts.get('propositional', 0)}")
        print(f"  First-order (1st order): {order_counts.get('first_order', 0)}")

        print(f"\nBy inference pattern:")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count}")
        print(f"\nOutput: {args.output}")

        # Show sample
        if annotations:
            print(f"\nSample output:")
            sample = annotations[0]
            print(f"  Premises: {[p.text for p in sample.premises]}")
            print(f"  Content: {sample.content}")
            if sample.verifier_notes:
                notes = json.loads(sample.verifier_notes)
                print(f"  Formal: {notes.get('full_formal', 'N/A')}")
                print(f"  Logic order: {notes.get('logic_order', 'N/A')}")


if __name__ == "__main__":
    main()
