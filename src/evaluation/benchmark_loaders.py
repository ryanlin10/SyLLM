"""Loaders that normalize HuggingFace datasets into a uniform benchmark format."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import re

from datasets import load_dataset

from .benchmark_registry import BenchmarkConfig


@dataclass
class BenchmarkItem:
    """Normalized benchmark item across all benchmarks."""

    id: str
    question: str
    choices: List[str]
    correct_answer: int  # Index for MC, class index for entailment
    correct_answer_text: str
    context: Optional[str] = None
    subject: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base loader
# =============================================================================


class BaseBenchmarkLoader:
    """Base class for benchmark data loaders."""

    def load(
        self, config: BenchmarkConfig, max_samples: Optional[int] = None
    ) -> List[BenchmarkItem]:
        raise NotImplementedError

    def get_few_shot_examples(
        self, config: BenchmarkConfig, n: int = 5
    ) -> List[BenchmarkItem]:
        raise NotImplementedError

    def _truncate(
        self, items: List[BenchmarkItem], max_samples: Optional[int]
    ) -> List[BenchmarkItem]:
        if max_samples and len(items) > max_samples:
            return items[:max_samples]
        return items


# =============================================================================
# Standard benchmark loaders
# =============================================================================


class MMLULoader(BaseBenchmarkLoader):
    """Loader for MMLU (Massive Multitask Language Understanding)."""

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            for idx, row in enumerate(dataset):
                choices = list(row["choices"])
                answer_idx = int(row["answer"])
                items.append(
                    BenchmarkItem(
                        id=f"mmlu_{idx}",
                        question=row["question"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx]
                        if answer_idx < len(choices)
                        else "",
                        subject=row.get("subject", "unknown"),
                    )
                )
        except Exception as e:
            print(f"Error loading MMLU: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=5):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.few_shot_split
            )
            for idx, row in enumerate(dataset):
                if idx >= n:
                    break
                choices = list(row["choices"])
                answer_idx = int(row["answer"])
                items.append(
                    BenchmarkItem(
                        id=f"mmlu_fs_{idx}",
                        question=row["question"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx]
                        if answer_idx < len(choices)
                        else "",
                        subject=row.get("subject", "unknown"),
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load MMLU few-shot examples: {e}")
        return items


class ARCChallengeLoader(BaseBenchmarkLoader):
    """Loader for ARC-Challenge."""

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            for idx, row in enumerate(dataset):
                choices = list(row["choices"]["text"])
                labels = list(row["choices"]["label"])
                answer_key = row["answerKey"]
                # Map answer key (A/B/C/D or 1/2/3/4) to index
                if answer_key in labels:
                    answer_idx = labels.index(answer_key)
                else:
                    answer_idx = 0
                items.append(
                    BenchmarkItem(
                        id=row.get("id", f"arc_{idx}"),
                        question=row["question"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx]
                        if answer_idx < len(choices)
                        else "",
                        subject="science",
                    )
                )
        except Exception as e:
            print(f"Error loading ARC-Challenge: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=25):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.few_shot_split
            )
            for idx, row in enumerate(dataset):
                if idx >= n:
                    break
                choices = list(row["choices"]["text"])
                labels = list(row["choices"]["label"])
                answer_key = row["answerKey"]
                answer_idx = labels.index(answer_key) if answer_key in labels else 0
                items.append(
                    BenchmarkItem(
                        id=f"arc_fs_{idx}",
                        question=row["question"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx]
                        if answer_idx < len(choices)
                        else "",
                        subject="science",
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load ARC few-shot examples: {e}")
        return items


class HellaSwagLoader(BaseBenchmarkLoader):
    """Loader for HellaSwag."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove HellaSwag-specific artifacts from text."""
        text = re.sub(r"\[.*?\]", "", text)
        text = text.strip()
        return text

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=config.split)
            for idx, row in enumerate(dataset):
                ctx = self._clean_text(row.get("ctx", ""))
                endings = [self._clean_text(e) for e in row["endings"]]
                label = int(row["label"])
                items.append(
                    BenchmarkItem(
                        id=f"hellaswag_{idx}",
                        question=f"{ctx}",
                        choices=endings,
                        correct_answer=label,
                        correct_answer_text=endings[label]
                        if label < len(endings)
                        else "",
                        subject=row.get("activity_label", "unknown"),
                    )
                )
        except Exception as e:
            print(f"Error loading HellaSwag: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=10):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=config.few_shot_split)
            for idx, row in enumerate(dataset):
                if idx >= n:
                    break
                ctx = self._clean_text(row.get("ctx", ""))
                endings = [self._clean_text(e) for e in row["endings"]]
                label = int(row["label"])
                items.append(
                    BenchmarkItem(
                        id=f"hellaswag_fs_{idx}",
                        question=f"{ctx}",
                        choices=endings,
                        correct_answer=label,
                        correct_answer_text=endings[label]
                        if label < len(endings)
                        else "",
                        subject=row.get("activity_label", "unknown"),
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load HellaSwag few-shot examples: {e}")
        return items


class WinograndeLoader(BaseBenchmarkLoader):
    """Loader for Winogrande."""

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            for idx, row in enumerate(dataset):
                sentence = row["sentence"]
                option1 = row["option1"]
                option2 = row["option2"]
                # Answer is "1" or "2" (string), convert to 0-indexed
                answer_str = row.get("answer", "1")
                answer_idx = int(answer_str) - 1 if answer_str in ("1", "2") else 0
                choices = [option1, option2]
                items.append(
                    BenchmarkItem(
                        id=f"winogrande_{idx}",
                        question=sentence,
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx],
                    )
                )
        except Exception as e:
            print(f"Error loading Winogrande: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=5):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.few_shot_split
            )
            for idx, row in enumerate(dataset):
                if idx >= n:
                    break
                choices = [row["option1"], row["option2"]]
                answer_str = row.get("answer", "1")
                answer_idx = int(answer_str) - 1 if answer_str in ("1", "2") else 0
                items.append(
                    BenchmarkItem(
                        id=f"winogrande_fs_{idx}",
                        question=row["sentence"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx],
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load Winogrande few-shot examples: {e}")
        return items


class GSM8KLoader(BaseBenchmarkLoader):
    """Loader for GSM8K (Grade School Math)."""

    @staticmethod
    def _extract_answer(answer_text: str) -> str:
        """Extract the numerical answer from GSM8K answer field."""
        match = re.search(r"####\s*([\-]?\d[\d,]*\.?\d*)", answer_text)
        if match:
            return match.group(1).replace(",", "")
        return answer_text.strip()

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            for idx, row in enumerate(dataset):
                answer_text = self._extract_answer(row["answer"])
                items.append(
                    BenchmarkItem(
                        id=f"gsm8k_{idx}",
                        question=row["question"],
                        choices=[],
                        correct_answer=-1,
                        correct_answer_text=answer_text,
                        metadata={"full_solution": row["answer"]},
                    )
                )
        except Exception as e:
            print(f"Error loading GSM8K: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=8):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.few_shot_split
            )
            for idx, row in enumerate(dataset):
                if idx >= n:
                    break
                items.append(
                    BenchmarkItem(
                        id=f"gsm8k_fs_{idx}",
                        question=row["question"],
                        choices=[],
                        correct_answer=-1,
                        correct_answer_text=self._extract_answer(row["answer"]),
                        metadata={"full_solution": row["answer"]},
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load GSM8K few-shot examples: {e}")
        return items


class TruthfulQALoader(BaseBenchmarkLoader):
    """Loader for TruthfulQA (MC2 format)."""

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            for idx, row in enumerate(dataset):
                mc2 = row.get("mc2_targets", {})
                choices = list(mc2.get("choices", []))
                labels = list(mc2.get("labels", []))

                if not choices:
                    continue

                # Find first correct answer for MC evaluation
                correct_idx = 0
                for i, label in enumerate(labels):
                    if label == 1:
                        correct_idx = i
                        break

                items.append(
                    BenchmarkItem(
                        id=f"truthfulqa_{idx}",
                        question=row["question"],
                        choices=choices[:4],  # Limit to 4 choices for consistency
                        correct_answer=min(correct_idx, 3),
                        correct_answer_text=choices[correct_idx]
                        if correct_idx < len(choices)
                        else "",
                        metadata={
                            "all_labels": labels,
                            "correct_indices": [
                                i for i, l in enumerate(labels) if l == 1
                            ],
                        },
                    )
                )
        except Exception as e:
            print(f"Error loading TruthfulQA: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=6):
        """TruthfulQA uses same split; take from end to avoid overlap."""
        items = []
        try:
            dataset = load_dataset(
                config.hf_dataset, config.hf_subset, split=config.split
            )
            # Take from end of dataset to reduce overlap with eval items
            start = max(0, len(dataset) - n)
            for idx in range(start, len(dataset)):
                row = dataset[idx]
                mc2 = row.get("mc2_targets", {})
                choices = list(mc2.get("choices", []))
                labels = list(mc2.get("labels", []))
                if not choices:
                    continue
                correct_idx = 0
                for i, l in enumerate(labels):
                    if l == 1:
                        correct_idx = i
                        break
                items.append(
                    BenchmarkItem(
                        id=f"truthfulqa_fs_{idx}",
                        question=row["question"],
                        choices=choices[:4],
                        correct_answer=min(correct_idx, 3),
                        correct_answer_text=choices[correct_idx]
                        if correct_idx < len(choices)
                        else "",
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load TruthfulQA few-shot examples: {e}")
        return items


# =============================================================================
# Additional standard benchmark loaders
# =============================================================================


class BoolQLoader(BaseBenchmarkLoader):
    """Loader for BoolQ (Boolean Question Answering over passages)."""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=split)
            for idx, row in enumerate(dataset):
                answer = row["answer"]  # bool
                correct_idx = 0 if answer else 1  # 0=Yes, 1=No
                items.append(
                    BenchmarkItem(
                        id=f"boolq_{idx}",
                        question=row["question"],
                        choices=["Yes", "No"],
                        correct_answer=correct_idx,
                        correct_answer_text="Yes" if answer else "No",
                        context=row.get("passage", ""),
                        subject="reading_comprehension",
                    )
                )
        except Exception as e:
            print(f"Error loading BoolQ: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=5):
        return self._load_split(config, config.few_shot_split, max_samples=n)


class PIQALoader(BaseBenchmarkLoader):
    """Loader for PIQA (Physical Intuition QA)."""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=split)
            for idx, row in enumerate(dataset):
                sol1, sol2 = row["sol1"], row["sol2"]
                label = int(row["label"])  # 0 or 1
                items.append(
                    BenchmarkItem(
                        id=f"piqa_{idx}",
                        question=row["goal"],
                        choices=[sol1, sol2],
                        correct_answer=label,
                        correct_answer_text=[sol1, sol2][label],
                        subject="physical_intuition",
                    )
                )
        except Exception as e:
            print(f"Error loading PIQA: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=5):
        return self._load_split(config, config.few_shot_split, max_samples=n)


class ARCEasyLoader(BaseBenchmarkLoader):
    """Loader for ARC-Easy (same format as ARC-Challenge, different subset)."""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, config.hf_subset, split=split)
            for idx, row in enumerate(dataset):
                choices = list(row["choices"]["text"])
                labels = list(row["choices"]["label"])
                answer_key = row["answerKey"]
                answer_idx = labels.index(answer_key) if answer_key in labels else 0
                items.append(
                    BenchmarkItem(
                        id=row.get("id", f"arc_easy_{idx}"),
                        question=row["question"],
                        choices=choices,
                        correct_answer=answer_idx,
                        correct_answer_text=choices[answer_idx]
                        if answer_idx < len(choices)
                        else "",
                        subject="science",
                    )
                )
        except Exception as e:
            print(f"Error loading ARC-Easy: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=25):
        return self._load_split(config, config.few_shot_split, max_samples=n)


class TriviaQALoader(BaseBenchmarkLoader):
    """Loader for TriviaQA (open-domain factual QA, generation format)."""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, config.hf_subset, split=split)
            for idx, row in enumerate(dataset):
                answer = row["answer"]
                canonical = answer.get("value", "")
                # Prefer normalized_aliases for robust matching; fall back to aliases
                aliases = list(
                    answer.get("normalized_aliases")
                    or answer.get("aliases")
                    or []
                )
                if canonical and canonical.lower() not in [a.lower() for a in aliases]:
                    aliases = [canonical] + aliases
                items.append(
                    BenchmarkItem(
                        id=row.get("question_id", f"triviaqa_{idx}"),
                        question=row["question"],
                        choices=[],
                        correct_answer=-1,
                        correct_answer_text=canonical,
                        subject="factual_knowledge",
                        metadata={"aliases": aliases},
                    )
                )
        except Exception as e:
            print(f"Error loading TriviaQA: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=5):
        return self._load_split(config, config.few_shot_split, max_samples=n)


class MATHLoader(BaseBenchmarkLoader):
    """Loader for the MATH benchmark (competition-level math problems)."""

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Extract content from the innermost \\boxed{...} accounting for nesting."""
        match = re.search(r"\\boxed\{", text)
        if not match:
            return ""
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
        return ""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            if config.hf_subset:
                dataset = load_dataset(config.hf_dataset, config.hf_subset, split=split)
            else:
                dataset = load_dataset(config.hf_dataset, split=split)
            for idx, row in enumerate(dataset):
                problem = row.get("problem", row.get("question", ""))
                solution = row.get("solution", "")
                # Use explicit answer field when available; otherwise extract from solution
                answer = row.get("answer") or self._extract_boxed(solution)
                subject = row.get("type", row.get("subject", "mathematics"))
                level = row.get("level", "")
                items.append(
                    BenchmarkItem(
                        id=f"math_{idx}",
                        question=problem,
                        choices=[],
                        correct_answer=-1,
                        correct_answer_text=answer,
                        subject=subject,
                        metadata={"full_solution": solution, "level": level},
                    )
                )
        except Exception as e:
            print(f"Error loading MATH: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=4):
        return self._load_split(config, config.few_shot_split, max_samples=n)


# =============================================================================
# Code benchmark loaders
# =============================================================================


class HumanEvalLoader(BaseBenchmarkLoader):
    """Loader for HumanEval (Python function synthesis, pass@1 via execution)."""

    def load(self, config, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=config.split)
            for idx, row in enumerate(dataset):
                items.append(
                    BenchmarkItem(
                        id=row.get("task_id", f"humaneval_{idx}"),
                        question=row["prompt"],
                        choices=[],
                        correct_answer=-1,
                        correct_answer_text="",
                        subject="coding",
                        metadata={
                            "test": row["test"],
                            "entry_point": row["entry_point"],
                            "canonical_solution": row["canonical_solution"],
                        },
                    )
                )
        except Exception as e:
            print(f"Error loading HumanEval: {e}")
        return self._truncate(items, max_samples)

    def get_few_shot_examples(self, config, n=0):
        return []  # HumanEval is evaluated 0-shot


class MBPPLoader(BaseBenchmarkLoader):
    """Loader for MBPP (Mostly Basic Python Programming)."""

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, config.hf_subset, split=split)
            for idx, row in enumerate(dataset):
                code = row.get("code", "")
                items.append(
                    BenchmarkItem(
                        id=f"mbpp_{row.get('task_id', idx)}",
                        question=row["text"],
                        choices=[],
                        correct_answer=-1,
                        # Store canonical code as correct_answer_text for few-shot display
                        correct_answer_text=code,
                        subject="coding",
                        metadata={
                            "test_list": list(row.get("test_list", [])),
                            "test_imports": list(row.get("test_imports", [])),
                            "code": code,
                        },
                    )
                )
        except Exception as e:
            print(f"Error loading MBPP: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=3):
        return self._load_split(config, config.few_shot_split, max_samples=n)


# =============================================================================
# Logic benchmark loaders
# =============================================================================


class LogiQALoader(BaseBenchmarkLoader):
    """Loader for LogiQA 2.0 benchmark.

    Uses datatune/LogiQA2.0 (parquet, no deprecated scripts).
    Each row has a JSON-encoded 'text' field with: id, answer, text,
    question, options, type.
    """

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            dataset = load_dataset(config.hf_dataset, split=split)

            for idx, row in enumerate(dataset):
                # Parse JSON from text column
                raw = row.get("text", "")
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    data = raw

                context = data.get("text", "")
                question = data.get("question", "")
                options = data.get("options", [])
                label = data.get("answer", 0)

                if isinstance(label, str):
                    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    label = label_map.get(label.upper(), 0)

                # Extract category from type dict
                type_info = data.get("type", {})
                if isinstance(type_info, dict):
                    category = next(iter(type_info.keys()), "general")
                else:
                    category = str(type_info) if type_info else "general"

                items.append(
                    BenchmarkItem(
                        id=f"logiqa_{data.get('id', idx)}",
                        question=question,
                        choices=list(options),
                        correct_answer=int(label),
                        correct_answer_text=options[int(label)]
                        if int(label) < len(options)
                        else "",
                        context=context,
                        subject=category,
                    )
                )
        except Exception as e:
            print(f"Error loading LogiQA: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=5):
        return self._load_split(config, config.few_shot_split, max_samples=n)


class FOLIOLoader(BaseBenchmarkLoader):
    """Loader for FOLIO (First-Order Logic Reasoning).

    Uses tasksource/folio (no gated access required).
    Fields: premises, conclusion, label (True/False/Uncertain).
    """

    LABEL_MAP = {"True": 0, "False": 1, "Unknown": 2, "Uncertain": 2}
    OPTIONS = ["True", "False", "Unknown"]

    def _load_split(self, config, split, max_samples=None):
        items = []
        try:
            actual_split = split if split != "test" else "validation"
            dataset = load_dataset(config.hf_dataset, split=actual_split)

            for idx, row in enumerate(dataset):
                premises = row.get("premises", "")
                conclusion = row.get("conclusion", "")
                label = row.get("label", "Unknown")
                correct = self.LABEL_MAP.get(label, 2)

                question = (
                    f"Based on the premises, is the following conclusion "
                    f"true, false, or unknown?\n\nConclusion: {conclusion}"
                )

                items.append(
                    BenchmarkItem(
                        id=f"folio_{idx}",
                        question=question,
                        choices=self.OPTIONS[:],
                        correct_answer=correct,
                        correct_answer_text=self.OPTIONS[correct],
                        context=premises,
                        subject="first_order_logic",
                    )
                )
        except Exception as e:
            print(f"Error loading FOLIO: {e}")
        return self._truncate(items, max_samples)

    def load(self, config, max_samples=None):
        return self._load_split(config, config.split, max_samples)

    def get_few_shot_examples(self, config, n=5):
        return self._load_split(config, config.few_shot_split, max_samples=n)


# =============================================================================
# Loader registry
# =============================================================================

LOADER_MAP = {
    # Standard
    "mmlu": MMLULoader(),
    "arc_challenge": ARCChallengeLoader(),
    "hellaswag": HellaSwagLoader(),
    "winogrande": WinograndeLoader(),
    "gsm8k": GSM8KLoader(),
    "truthfulqa": TruthfulQALoader(),
    # Additional standard (generalist / catastrophic-forgetting probes)
    "boolq": BoolQLoader(),
    "piqa": PIQALoader(),
    "arc_easy": ARCEasyLoader(),
    "triviaqa": TriviaQALoader(),
    "math": MATHLoader(),
    # Code
    "humaneval": HumanEvalLoader(),
    "mbpp": MBPPLoader(),
    # Logic
    "logiqa": LogiQALoader(),
    "folio": FOLIOLoader(),
}


def get_loader(benchmark_key: str) -> BaseBenchmarkLoader:
    """Get the loader for a given benchmark key."""
    if benchmark_key not in LOADER_MAP:
        available = ", ".join(LOADER_MAP.keys())
        raise ValueError(f"No loader for benchmark: {benchmark_key}. Available: {available}")
    return LOADER_MAP[benchmark_key]
