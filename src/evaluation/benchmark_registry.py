"""Benchmark registry with metadata, prompt templates, and evaluation configs."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class BenchmarkTier(Enum):
    CRITICAL = "critical"  # 4 core benchmarks for quick evaluation
    EXTENDED = "extended"  # Broader coverage when compute allows
    SPECIALIZED = "specialized"  # Only when explicitly selected


class BenchmarkType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"
    ENTAILMENT = "entailment"
    CODE_GENERATION = "code_generation"


class BenchmarkCategory(Enum):
    STANDARD = "standard"
    LOGIC = "logic"
    CODE = "code"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    key: str
    hf_dataset: str
    hf_subset: Optional[str] = None
    split: str = "test"
    few_shot_split: str = "validation"
    category: BenchmarkCategory = BenchmarkCategory.STANDARD
    benchmark_type: BenchmarkType = BenchmarkType.MULTIPLE_CHOICE
    num_few_shot: int = 5
    num_choices: int = 4
    description: str = ""
    random_baseline: float = 0.25
    tier: BenchmarkTier = BenchmarkTier.EXTENDED
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    max_tokens_override: Optional[int] = None  # Per-benchmark token limit override


BENCHMARK_REGISTRY: Dict[str, BenchmarkConfig] = {
    # -------------------------------------------------------------------------
    # Standard benchmarks
    # -------------------------------------------------------------------------
    "mmlu": BenchmarkConfig(
        name="MMLU",
        key="mmlu",
        hf_dataset="cais/mmlu",
        hf_subset="all",
        split="test",
        few_shot_split="dev",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=4,
        description="Massive Multitask Language Understanding",
        random_baseline=0.25,
        tier=BenchmarkTier.EXTENDED,
    ),
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        key="arc_challenge",
        hf_dataset="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=25,
        num_choices=4,
        description="AI2 Reasoning Challenge (Challenge set)",
        random_baseline=0.25,
        tier=BenchmarkTier.CRITICAL,
    ),
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        key="hellaswag",
        hf_dataset="Rowan/hellaswag",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=10,
        num_choices=4,
        description="Grounded Commonsense Inference",
        random_baseline=0.25,
        tier=BenchmarkTier.EXTENDED,
    ),
    "winogrande": BenchmarkConfig(
        name="Winogrande",
        key="winogrande",
        hf_dataset="allenai/winogrande",
        hf_subset="winogrande_xl",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=2,
        description="Winograd Schema Challenge (large)",
        random_baseline=0.5,
        tier=BenchmarkTier.EXTENDED,
    ),
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        key="gsm8k",
        hf_dataset="openai/gsm8k",
        hf_subset="main",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.GENERATION,
        num_few_shot=8,
        num_choices=0,
        description="Grade School Math",
        random_baseline=0.0,
        tier=BenchmarkTier.EXTENDED,
    ),
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA",
        key="truthfulqa",
        hf_dataset="truthfulqa/truthful_qa",
        hf_subset="multiple_choice",
        split="validation",
        few_shot_split="validation",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=6,
        num_choices=4,
        description="TruthfulQA MC2",
        random_baseline=0.25,
        tier=BenchmarkTier.EXTENDED,
    ),
    # -------------------------------------------------------------------------
    # Additional standard benchmarks (generalist skills / catastrophic forgetting)
    # -------------------------------------------------------------------------
    "boolq": BenchmarkConfig(
        name="BoolQ",
        key="boolq",
        hf_dataset="google/boolq",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=2,
        description="Boolean Reading Comprehension (yes/no over passages)",
        random_baseline=0.5,
    ),
    "piqa": BenchmarkConfig(
        name="PIQA",
        key="piqa",
        hf_dataset="ybisk/piqa",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=2,
        description="Physical Intuition Question Answering",
        random_baseline=0.5,
    ),
    "arc_easy": BenchmarkConfig(
        name="ARC-Easy",
        key="arc_easy",
        hf_dataset="allenai/ai2_arc",
        hf_subset="ARC-Easy",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=25,
        num_choices=4,
        description="AI2 Reasoning Challenge (Easy set)",
        random_baseline=0.25,
    ),
    "triviaqa": BenchmarkConfig(
        name="TriviaQA",
        key="triviaqa",
        hf_dataset="trivia_qa",
        hf_subset="rc",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.GENERATION,
        num_few_shot=5,
        num_choices=0,
        description="Open-domain Factual QA (trivia knowledge)",
        random_baseline=0.0,
        metrics=["accuracy"],
    ),
    "math": BenchmarkConfig(
        name="MATH",
        key="math",
        hf_dataset="lighteval/MATH",
        hf_subset="all",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.GENERATION,
        num_few_shot=4,
        num_choices=0,
        description="Competition Mathematics (Algebra through Calculus)",
        random_baseline=0.0,
        max_tokens_override=512,
    ),
    # -------------------------------------------------------------------------
    # Code benchmarks
    # -------------------------------------------------------------------------
    "humaneval": BenchmarkConfig(
        name="HumanEval",
        key="humaneval",
        hf_dataset="openai/openai_humaneval",
        split="test",
        few_shot_split="test",
        category=BenchmarkCategory.CODE,
        benchmark_type=BenchmarkType.CODE_GENERATION,
        num_few_shot=0,
        num_choices=0,
        description="Python function synthesis with unit test evaluation (pass@1)",
        random_baseline=0.0,
        metrics=["pass@1"],
        max_tokens_override=512,
    ),
    "mbpp": BenchmarkConfig(
        name="MBPP",
        key="mbpp",
        hf_dataset="google-research-datasets/mbpp",
        hf_subset="sanitized",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.CODE,
        benchmark_type=BenchmarkType.CODE_GENERATION,
        num_few_shot=3,
        num_choices=0,
        description="Mostly Basic Python Programming with assertion-based evaluation",
        random_baseline=0.0,
        metrics=["pass@1"],
        max_tokens_override=512,
    ),
    # -------------------------------------------------------------------------
    # Logic benchmarks
    # -------------------------------------------------------------------------
    "logiqa": BenchmarkConfig(
        name="LogiQA",
        key="logiqa",
        hf_dataset="datatune/LogiQA2.0",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.LOGIC,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=4,
        description="Logic Question Answering",
        random_baseline=0.25,
    ),
    "folio": BenchmarkConfig(
        name="FOLIO",
        key="folio",
        hf_dataset="tasksource/folio",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.LOGIC,
        benchmark_type=BenchmarkType.ENTAILMENT,
        num_few_shot=5,
        num_choices=3,
        description="First-Order Logic Reasoning",
        random_baseline=1.0 / 3.0,
    ),
}


def get_benchmark_config(name: str) -> BenchmarkConfig:
    """Get benchmark configuration by key."""
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARK_REGISTRY[name]


def list_benchmarks() -> Dict[str, str]:
    """List all available benchmarks with descriptions."""
    return {k: f"{v.name} - {v.description}" for k, v in BENCHMARK_REGISTRY.items()}
