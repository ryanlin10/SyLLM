"""Core benchmark evaluation engine."""

import os
import sys
import time
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .benchmark_registry import (
    BenchmarkConfig,
    BenchmarkType,
    BenchmarkCategory,
    get_benchmark_config,
    BENCHMARK_REGISTRY,
)
from .benchmark_loaders import get_loader, BenchmarkItem
from .answer_parser import (
    parse_multiple_choice_answer,
    parse_gsm8k_answer,
    parse_true_false_unknown,
    parse_code_answer,
    parse_math_answer,
    normalize_math_answer,
    parse_triviaqa_answer,
    normalize_answer,
)
from .statistics import bootstrap_confidence_interval, compute_normalized_score


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class SingleResult:
    """Result for a single benchmark item."""

    item_id: str
    correct: bool
    predicted_answer: int
    correct_answer: int
    predicted_text: str
    correct_text: str
    raw_output: str
    parse_confidence: float
    latency_seconds: float
    subject: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated results for one benchmark, one model configuration."""

    benchmark_key: str
    benchmark_name: str
    model_label: str
    accuracy: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    normalized_score: float
    num_total: int
    num_correct: int
    num_parse_failures: int
    per_subject_accuracy: Dict[str, float] = field(default_factory=dict)
    per_subject_count: Dict[str, int] = field(default_factory=dict)
    avg_latency_seconds: float = 0.0
    total_time_seconds: float = 0.0
    throughput_items_per_sec: float = 0.0
    timestamp: str = ""
    individual_results: List[SingleResult] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of base vs finetuned for one benchmark."""

    benchmark_key: str
    benchmark_name: str
    category: str
    base_result: BenchmarkResult
    finetuned_result: BenchmarkResult
    accuracy_delta: float
    is_improvement: bool


# =============================================================================
# Prompt formatting
# =============================================================================

MC_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer multiple choice questions by selecting "
    "the correct option. Respond with just the letter of your answer."
)

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve problems step by step and provide "
    "your final answer after ####."
)

ENTAILMENT_SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Determine whether a conclusion "
    "follows from the given premises. Answer with True, False, or Unknown."
)

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step. "
    "Put your final answer inside \\boxed{}."
)

TRIVIAQA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer factual questions concisely. "
    "Respond with just the answer, no explanation."
)

CODE_GENERATION_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function based on "
    "the description and any provided examples. Output only valid Python code "
    "with no extra explanation."
)


def _format_mc_example(item: BenchmarkItem, include_answer: bool = False) -> str:
    """Format a single multiple-choice example."""
    lines = []
    if item.context:
        lines.append(f"Context: {item.context}\n")
    lines.append(f"Question: {item.question}")
    for i, choice in enumerate(item.choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    if include_answer:
        answer_letter = chr(ord("A") + item.correct_answer)
        lines.append(f"Answer: {answer_letter}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _format_gsm8k_example(item: BenchmarkItem, include_answer: bool = False) -> str:
    """Format a GSM8K example."""
    lines = [f"Question: {item.question}"]
    if include_answer:
        solution = item.metadata.get("full_solution", item.correct_answer_text)
        lines.append(f"Solution: {solution}")
    else:
        lines.append("Solution:")
    return "\n".join(lines)


def _format_entailment_example(
    item: BenchmarkItem, include_answer: bool = False
) -> str:
    """Format an entailment example (FOLIO-style)."""
    lines = []
    if item.context:
        lines.append(f"Premises: {item.context}\n")
    lines.append(item.question)
    if include_answer:
        lines.append(f"Answer: {item.correct_answer_text}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _format_math_example(item: BenchmarkItem, include_answer: bool = False) -> str:
    """Format a MATH benchmark problem."""
    lines = [f"Problem: {item.question}"]
    if include_answer:
        solution = item.metadata.get("full_solution", "")
        if solution:
            lines.append(f"Solution: {solution}")
        else:
            lines.append(f"Answer: \\boxed{{{item.correct_answer_text}}}")
    else:
        lines.append("Solution:")
    return "\n".join(lines)


def _format_triviaqa_example(
    item: BenchmarkItem, include_answer: bool = False
) -> str:
    """Format a TriviaQA question."""
    lines = [f"Question: {item.question}"]
    if include_answer:
        lines.append(f"Answer: {item.correct_answer_text}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _format_code_example(item: BenchmarkItem, include_answer: bool = False) -> str:
    """Format a code generation example (HumanEval / MBPP).

    For HumanEval the prompt already contains the function signature + docstring.
    For MBPP the prompt is a task description; few-shot answers show the solution.
    """
    if include_answer:
        code = item.correct_answer_text or item.metadata.get("code", "")
        return f"{item.question}\n{code}"
    return item.question


def format_prompt(
    item: BenchmarkItem,
    few_shot: List[BenchmarkItem],
    config: BenchmarkConfig,
) -> str:
    """Format a full prompt with few-shot examples."""
    parts = []

    if config.benchmark_type == BenchmarkType.MULTIPLE_CHOICE:
        formatter = _format_mc_example
    elif config.benchmark_type == BenchmarkType.GENERATION:
        if config.key == "math":
            formatter = _format_math_example
        elif config.key == "triviaqa":
            formatter = _format_triviaqa_example
        else:
            formatter = _format_gsm8k_example  # GSM8K default
    elif config.benchmark_type == BenchmarkType.ENTAILMENT:
        formatter = _format_entailment_example
    elif config.benchmark_type == BenchmarkType.CODE_GENERATION:
        formatter = _format_code_example
    else:
        formatter = _format_mc_example

    # Few-shot demonstrations
    for fs in few_shot:
        parts.append(formatter(fs, include_answer=True))

    # Target question
    parts.append(formatter(item, include_answer=False))

    return "\n\n".join(parts)


def get_system_prompt(config: BenchmarkConfig) -> str:
    """Get the system prompt for a benchmark type."""
    if config.benchmark_type == BenchmarkType.MULTIPLE_CHOICE:
        return MC_SYSTEM_PROMPT
    elif config.benchmark_type == BenchmarkType.GENERATION:
        if config.key == "math":
            return MATH_SYSTEM_PROMPT
        elif config.key == "triviaqa":
            return TRIVIAQA_SYSTEM_PROMPT
        return GENERATION_SYSTEM_PROMPT
    elif config.benchmark_type == BenchmarkType.ENTAILMENT:
        return ENTAILMENT_SYSTEM_PROMPT
    elif config.benchmark_type == BenchmarkType.CODE_GENERATION:
        return CODE_GENERATION_SYSTEM_PROMPT
    return MC_SYSTEM_PROMPT


# =============================================================================
# Code execution helpers
# =============================================================================


def _run_in_subprocess(code: str, timeout: int = 10) -> bool:
    """Write code to a temp file and execute it; return True if exit code is 0."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _execute_humaneval(
    generated: str, test: str, entry_point: str, timeout: int = 10
) -> bool:
    """Evaluate a HumanEval completion via subprocess execution.

    Combines the generated code with the provided test harness and calls
    check(<entry_point>).  Returns True if all assertions pass.
    """
    code = parse_code_answer(generated, entry_point)
    # HumanEval test functions are named check(candidate)
    full_program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    return _run_in_subprocess(full_program, timeout=timeout)


def _execute_mbpp(
    generated: str, test_list: List[str], test_imports: List[str], timeout: int = 10
) -> bool:
    """Evaluate an MBPP completion via subprocess execution.

    Runs the generated code followed by the assert-based test list.
    """
    code = parse_code_answer(generated)
    imports = "\n".join(test_imports)
    assertions = "\n".join(test_list)
    full_program = f"{imports}\n{code}\n{assertions}\n"
    return _run_in_subprocess(full_program, timeout=timeout)


# =============================================================================
# Core evaluator
# =============================================================================


class BenchmarkEvaluator:
    """Orchestrates benchmark evaluation for base and finetuned models.

    Uses VLLMPredictor with use_lora toggle to switch between base and
    finetuned inference on a single vLLM engine.
    """

    def __init__(
        self,
        predictor,
        benchmarks: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        num_few_shot: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        batch_size: int = 32,
    ):
        """
        Args:
            predictor: VLLMPredictor instance (loaded with enable_lora=True).
            benchmarks: List of benchmark keys to run (default: all).
            max_samples: Max evaluation items per benchmark (None=full).
            num_few_shot: Override default few-shot count per benchmark.
            temperature: Sampling temperature (0.0=greedy for benchmarks).
            max_tokens: Maximum tokens to generate per item.
            batch_size: Batch size for inference.
        """
        self.predictor = predictor
        self.benchmarks = benchmarks or list(BENCHMARK_REGISTRY.keys())
        self.max_samples = max_samples
        self.num_few_shot = num_few_shot
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def evaluate_single_benchmark(
        self,
        benchmark_key: str,
        use_lora: bool,
        model_label: str,
    ) -> BenchmarkResult:
        """Run evaluation on a single benchmark with a specific model config."""
        config = get_benchmark_config(benchmark_key)
        loader = get_loader(benchmark_key)

        # Load data
        items = loader.load(config, max_samples=self.max_samples)
        if not items:
            print(f"  Warning: No items loaded for {benchmark_key}")
            return self._empty_result(benchmark_key, config, model_label)

        # Load few-shot examples
        n_shot = (
            self.num_few_shot if self.num_few_shot is not None else config.num_few_shot
        )
        few_shot = loader.get_few_shot_examples(config, n=n_shot)

        # Format prompts
        prompts = [format_prompt(item, few_shot, config) for item in items]
        system_prompt = get_system_prompt(config)

        # Use per-benchmark token override if set, else fall back to evaluator default
        effective_max_tokens = config.max_tokens_override or self.max_tokens

        # Run inference in batches
        results: List[SingleResult] = []
        total_start = time.time()

        for batch_start in range(0, len(prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_items = items[batch_start:batch_end]

            batch_start_time = time.time()
            outputs = self.predictor.generate_batch(
                messages=batch_prompts,
                system_prompt=system_prompt,
                max_tokens=effective_max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                use_lora=use_lora,
            )
            batch_latency = time.time() - batch_start_time
            per_item_latency = batch_latency / len(batch_prompts)

            for item, output in zip(batch_items, outputs):
                predicted, confidence, correct = self._evaluate_item(
                    output, item, config
                )

                predicted_text = self._predicted_text(predicted, item, config)

                results.append(
                    SingleResult(
                        item_id=item.id,
                        correct=correct,
                        predicted_answer=predicted,
                        correct_answer=item.correct_answer,
                        predicted_text=predicted_text,
                        correct_text=item.correct_answer_text,
                        raw_output=output,
                        parse_confidence=confidence,
                        latency_seconds=per_item_latency,
                        subject=item.subject,
                    )
                )

            print(
                f"    Batch {batch_start // self.batch_size + 1}/"
                f"{(len(prompts) + self.batch_size - 1) // self.batch_size} "
                f"complete"
            )

        total_time = time.time() - total_start
        return self._aggregate_results(
            benchmark_key, config, model_label, results, total_time
        )

    def evaluate_all(self) -> List[ComparisonResult]:
        """Run all benchmarks for both base and finetuned, return comparisons."""
        comparisons = []

        for benchmark_key in self.benchmarks:
            config = get_benchmark_config(benchmark_key)
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {config.name} ({benchmark_key})")
            print(f"{'=' * 60}")

            # Finetuned (with LoRA)
            print(f"  Running finetuned model...")
            finetuned_result = self.evaluate_single_benchmark(
                benchmark_key, use_lora=True, model_label="finetuned"
            )
            print(
                f"  Finetuned accuracy: {finetuned_result.accuracy:.2%} "
                f"({finetuned_result.num_correct}/{finetuned_result.num_total})"
            )

            # Base (without LoRA)
            print(f"  Running base model...")
            base_result = self.evaluate_single_benchmark(
                benchmark_key, use_lora=False, model_label="base"
            )
            print(
                f"  Base accuracy: {base_result.accuracy:.2%} "
                f"({base_result.num_correct}/{base_result.num_total})"
            )

            delta = finetuned_result.accuracy - base_result.accuracy
            comparisons.append(
                ComparisonResult(
                    benchmark_key=benchmark_key,
                    benchmark_name=config.name,
                    category=config.category.value,
                    base_result=base_result,
                    finetuned_result=finetuned_result,
                    accuracy_delta=delta,
                    is_improvement=(delta > 0),
                )
            )

        return comparisons

    def _evaluate_item(
        self, output: str, item: BenchmarkItem, config: BenchmarkConfig
    ):
        """Return (predicted_answer, confidence, correct) for one benchmark item."""
        btype = config.benchmark_type

        # ----- Multiple Choice -----
        if btype == BenchmarkType.MULTIPLE_CHOICE:
            predicted, confidence = parse_multiple_choice_answer(
                output, config.num_choices
            )
            correct = predicted == item.correct_answer
            return predicted, confidence, correct

        # ----- Entailment (True/False/Unknown) -----
        if btype == BenchmarkType.ENTAILMENT:
            predicted = parse_true_false_unknown(output)
            correct = predicted == item.correct_answer
            return predicted, 1.0, correct

        # ----- Generation -----
        if btype == BenchmarkType.GENERATION:
            return self._evaluate_generation(output, item, config)

        # ----- Code Generation -----
        if btype == BenchmarkType.CODE_GENERATION:
            return self._evaluate_code(output, item, config)

        return 0, 0.0, False

    def _evaluate_generation(
        self, output: str, item: BenchmarkItem, config: BenchmarkConfig
    ):
        """Dispatch generation correctness by benchmark key."""
        key = config.key

        # GSM8K — numeric answer extraction
        if key == "gsm8k":
            pred_num = parse_gsm8k_answer(output)
            try:
                expected_num = float(item.correct_answer_text)
                correct = pred_num is not None and abs(pred_num - expected_num) < 1e-6
            except (ValueError, TypeError):
                correct = False
            confidence = 1.0 if pred_num is not None else 0.0
            return -1, confidence, correct

        # MATH — LaTeX boxed answer extraction + normalized comparison
        if key == "math":
            pred_ans = normalize_math_answer(parse_math_answer(output))
            gold_ans = normalize_math_answer(item.correct_answer_text)
            correct = bool(pred_ans) and pred_ans == gold_ans
            # Try numeric fallback (e.g., "6" vs "6.0")
            if not correct and pred_ans and gold_ans:
                try:
                    correct = abs(float(pred_ans) - float(gold_ans)) < 1e-6
                except (ValueError, TypeError):
                    pass
            confidence = 1.0 if pred_ans else 0.0
            return -1, confidence, correct

        # TriviaQA — alias-based string matching
        if key == "triviaqa":
            pred_text = parse_triviaqa_answer(output)
            aliases = [
                normalize_answer(a)
                for a in item.metadata.get("aliases", [item.correct_answer_text])
            ]
            correct = pred_text in aliases
            confidence = 1.0 if pred_text else 0.0
            return -1, confidence, correct

        # Unknown generation benchmark — fall back to no-op
        return -1, 0.0, False

    def _evaluate_code(
        self, output: str, item: BenchmarkItem, config: BenchmarkConfig
    ):
        """Execute generated code against test suite; return pass/fail."""
        key = config.key

        if key == "humaneval":
            test = item.metadata.get("test", "")
            entry_point = item.metadata.get("entry_point", "")
            correct = _execute_humaneval(output, test, entry_point)
        elif key == "mbpp":
            test_list = item.metadata.get("test_list", [])
            test_imports = item.metadata.get("test_imports", [])
            correct = _execute_mbpp(output, test_list, test_imports)
        else:
            correct = False

        confidence = 1.0  # Execution verdict is definitive
        return -1, confidence, correct

    @staticmethod
    def _predicted_text(predicted: int, item: BenchmarkItem, config: BenchmarkConfig) -> str:
        """Convert predicted index to a display string."""
        if config.benchmark_type == BenchmarkType.MULTIPLE_CHOICE:
            return chr(ord("A") + predicted) if predicted >= 0 else "?"
        if config.benchmark_type in (BenchmarkType.GENERATION, BenchmarkType.CODE_GENERATION):
            return str(predicted)  # raw index; raw_output is the actual prediction
        # Entailment
        if predicted < len(item.choices):
            return item.choices[predicted]
        return str(predicted)

    def _aggregate_results(
        self,
        benchmark_key: str,
        config: BenchmarkConfig,
        model_label: str,
        results: List[SingleResult],
        total_time: float,
    ) -> BenchmarkResult:
        """Compute aggregate metrics from individual results."""
        if not results:
            return self._empty_result(benchmark_key, config, model_label)

        correctness = [r.correct for r in results]
        accuracy = sum(correctness) / len(correctness)
        ci_lower, ci_upper = bootstrap_confidence_interval(correctness)
        normalized = compute_normalized_score(accuracy, config.random_baseline)

        # Per-subject breakdown
        subject_correct = defaultdict(int)
        subject_total = defaultdict(int)
        for r in results:
            subj = r.subject or "overall"
            subject_total[subj] += 1
            if r.correct:
                subject_correct[subj] += 1

        per_subject_accuracy = {
            s: subject_correct[s] / subject_total[s] for s in subject_total
        }

        return BenchmarkResult(
            benchmark_key=benchmark_key,
            benchmark_name=config.name,
            model_label=model_label,
            accuracy=accuracy,
            accuracy_ci_lower=ci_lower,
            accuracy_ci_upper=ci_upper,
            normalized_score=normalized,
            num_total=len(results),
            num_correct=sum(correctness),
            num_parse_failures=sum(1 for r in results if r.parse_confidence < 0.5),
            per_subject_accuracy=per_subject_accuracy,
            per_subject_count=dict(subject_total),
            avg_latency_seconds=sum(r.latency_seconds for r in results) / len(results),
            total_time_seconds=total_time,
            throughput_items_per_sec=len(results) / total_time if total_time > 0 else 0,
            timestamp=datetime.now().isoformat(),
            individual_results=results,
        )

    def _empty_result(
        self, benchmark_key: str, config: BenchmarkConfig, model_label: str
    ) -> BenchmarkResult:
        """Return an empty result when no items were loaded."""
        return BenchmarkResult(
            benchmark_key=benchmark_key,
            benchmark_name=config.name,
            model_label=model_label,
            accuracy=0.0,
            accuracy_ci_lower=0.0,
            accuracy_ci_upper=0.0,
            normalized_score=0.0,
            num_total=0,
            num_correct=0,
            num_parse_failures=0,
            timestamp=datetime.now().isoformat(),
        )
