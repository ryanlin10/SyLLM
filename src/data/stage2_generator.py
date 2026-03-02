"""Generate Stage 2 training data: semi-formal CoT on logic benchmarks.

Prompts an OpenAI model to solve FOLIO and LogiQA problems using
semi-formal curly-bracket notation interleaved with natural language
reasoning.  Only examples where the LLM's final answer matches the
ground truth are kept.  Formal ``<PREMISE>/<CONCLUSION>`` steps can
optionally be verified with Z3.

The output format is compatible with ``lora_finetune.py`` and with
``SoundnessReward._extract_segments()`` used during GRPO training.

Usage (programmatic)::

    from src.data.stage2_generator import Stage2Generator, Stage2Config

    config = Stage2Config(verify_with_z3=False)
    gen = Stage2Generator(config)
    results = gen.generate(benchmarks=["folio"], max_samples=10)

See ``scripts/generate_stage2_data.py`` for the CLI entry point.
"""

import json
import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None

from .schema import Annotation, Premise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Stage2Example:
    """A single Stage 2 example: benchmark problem solved with semi-formal CoT."""

    benchmark: str
    item_id: str
    question: str               # original benchmark question text
    problem_text: str           # formatted problem shown to the LLM
    reasoning: str              # full CoT (NL + formal tags)
    predicted_answer: str       # LLM's final answer
    correct_answer: str         # ground truth
    is_correct: bool
    verification_status: bool   # whether formal steps passed Z3

    def to_training_dict(self) -> Dict:
        """Convert to flat dict for JSONL output.

        The output omits ``premises`` (always empty for Stage 2) and
        surfaces ``question``, ``answer``, and ``item_id`` as top-level
        columns.
        """
        return {
            "id": str(uuid.uuid4()),
            "item_id": self.item_id,
            "question": self.question,
            "answer": self.correct_answer,
            "content": self.reasoning,
            "benchmark": self.benchmark,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "verification_status": self.verification_status,
            "annotator_id": f"stage2_generator_{self.benchmark}",
            "timestamp": datetime.now().isoformat(),
        }


@dataclass
class Stage2Config:
    """Configuration for the Stage 2 generator."""

    api_key: Optional[str] = None
    model: str = "gpt-5.2-2025-12-11"
    temperature: float = 0.7
    max_tokens: int = 4096
    verify_with_z3: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


# ---------------------------------------------------------------------------
# Few-shot examples (hardcoded, real-world scenarios)
# ---------------------------------------------------------------------------

# Three examples covering True / False / Unknown answers and demonstrating
# all major connectives: forall, if-then, and, not, there-exist.
_FEW_SHOT_EXAMPLES = """\

--- Example 1 ---
Problem:
Premises:
All patients with a fever and a cough are referred for a flu test.
Anyone referred for a flu test receives their result the same day.
Emma is a patient with a fever and a cough.

Based on the premises, is the following conclusion true, false, or unknown?

Conclusion: Emma receives her result the same day.

Reasoning:
Emma is a patient with a fever and a cough. The first premise states that \
all such patients are referred for a flu test.

<PREMISE> {for all x, {if {x is a patient with a fever and x has a cough}, \
then x is referred for a flu test}} </PREMISE>
<PREMISE> {Emma is a patient with a fever and Emma has a cough} </PREMISE>
<CONCLUSION> Emma is referred for a flu test </CONCLUSION>

The second premise states that anyone referred for a flu test receives \
their result the same day.

<PREMISE> {for all x, {if x is referred for a flu test, then x receives \
their result the same day}} </PREMISE>
<PREMISE> Emma is referred for a flu test </PREMISE>
<CONCLUSION> Emma receives her result the same day </CONCLUSION>

The conclusion follows directly from the premises.

Answer: True

--- Example 2 ---
Problem:
Premises:
All employees who work remotely live outside the city.
No one who lives outside the city commutes by subway.
David is an employee who works remotely.

Based on the premises, is the following conclusion true, false, or unknown?

Conclusion: David commutes by subway.

Reasoning:
David is an employee who works remotely. By the first premise, all remote \
employees live outside the city.

<PREMISE> {for all x, {if x works remotely, then x lives outside the \
city}} </PREMISE>
<PREMISE> David works remotely </PREMISE>
<CONCLUSION> David lives outside the city </CONCLUSION>

The second premise tells us that no one who lives outside the city commutes \
by subway.

<PREMISE> {for all x, {if x lives outside the city, then {it is not the \
case that x commutes by subway}}} </PREMISE>
<PREMISE> David lives outside the city </PREMISE>
<CONCLUSION> {it is not the case that David commutes by subway} </CONCLUSION>

The premises entail that David does not commute by subway, which directly \
contradicts the conclusion.

Answer: False

--- Example 3 ---
Problem:
Premises:
All students in the honors program take advanced mathematics.
Some students who take advanced mathematics also tutor younger students.
Rachel is a student in the honors program.

Based on the premises, is the following conclusion true, false, or unknown?

Conclusion: Rachel tutors younger students.

Reasoning:
Rachel is in the honors program. By the first premise, all honors students \
take advanced mathematics.

<PREMISE> {for all x, {if x is in the honors program, then x takes \
advanced mathematics}} </PREMISE>
<PREMISE> Rachel is in the honors program </PREMISE>
<CONCLUSION> Rachel takes advanced mathematics </CONCLUSION>

The second premise tells us {there exist x such that {x takes advanced \
mathematics and x tutors younger students}}. This means at least one \
student who takes advanced mathematics also tutors, but we cannot determine \
whether Rachel specifically is among them. The conclusion cannot be derived \
from the premises, nor can it be ruled out.

Answer: Unknown"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Stage2Generator:
    """Generate Stage 2 training data from logic benchmarks.

    Loads problems via the existing benchmark loaders, prompts an OpenAI
    model to solve each problem using semi-formal CoT, filters for
    correctness, and optionally Z3-verifies the formal steps.
    """

    # FOLIO entailment labels
    _FOLIO_LABELS = ["True", "False", "Unknown"]

    def __init__(self, config: Stage2Config = None):
        self.config = config or Stage2Config()

        # ----- OpenAI client -----
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if openai is None:
            raise ImportError("Install openai: pip install openai")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Set via config or environment variable."
            )
        self.client = openai.OpenAI(api_key=api_key)

        # ----- Optional Z3 verifier -----
        self.verifier = None
        if self.config.verify_with_z3:
            try:
                from ..verification.verifier import VerifierPipeline
                self.verifier = VerifierPipeline()
            except Exception as exc:
                logger.warning(
                    "Could not initialize VerifierPipeline (%s). "
                    "Continuing without Z3 verification.", exc
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        benchmarks: List[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[Stage2Example]:
        """Generate Stage 2 examples across the specified benchmarks.

        Parameters
        ----------
        benchmarks:
            Benchmark keys to use (default: ``["folio", "logiqa"]``).
        max_samples:
            Maximum number of problems to process per benchmark.
            ``None`` means use all available problems.

        Returns
        -------
        List[Stage2Example]
            Successfully generated (correct, optionally verified) examples.
        """
        if benchmarks is None:
            benchmarks = ["folio", "logiqa"]

        items_by_benchmark = self._load_benchmark_problems(
            benchmarks, max_samples
        )

        examples: List[Stage2Example] = []
        for bench_key, items in items_by_benchmark.items():
            for item in items:
                try:
                    example = self._process_item(bench_key, item)
                    if example is not None:
                        examples.append(example)
                except Exception as exc:
                    logger.warning(
                        "Error processing %s item %s: %s",
                        bench_key, item.id, exc,
                    )

        return examples

    def render_to_annotation(self, example: Stage2Example) -> Annotation:
        """Convert a :class:`Stage2Example` to :class:`Annotation` format.

        Stage 2 annotations have **empty premises** — the full problem
        goes in the user message (via ``verifier_notes``), and the
        assistant message is the CoT reasoning in ``content``.
        """
        verifier_notes = json.dumps({
            "benchmark": example.benchmark,
            "item_id": example.item_id,
            "problem_text": example.problem_text,
            "predicted_answer": example.predicted_answer,
            "correct_answer": example.correct_answer,
            "is_correct": example.is_correct,
            "verification_status": example.verification_status,
            "stage": 2,
        }, ensure_ascii=False)

        return Annotation(
            id=str(uuid.uuid4()),
            premises=[],
            content=example.reasoning,
            verifier_notes=verifier_notes,
            annotator_id=f"stage2_generator_{example.benchmark}",
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Benchmark loading
    # ------------------------------------------------------------------

    def _load_benchmark_problems(
        self,
        benchmarks: List[str],
        max_samples: Optional[int],
    ) -> Dict[str, list]:
        """Load problems from the specified benchmarks.

        Uses the existing benchmark loaders and registry.
        """
        from ..evaluation.benchmark_loaders import get_loader
        from ..evaluation.benchmark_registry import get_benchmark_config

        items_by_benchmark: Dict[str, list] = {}
        for bench_key in benchmarks:
            config = get_benchmark_config(bench_key)
            loader = get_loader(bench_key)
            items = loader.load(config, max_samples=max_samples)
            items_by_benchmark[bench_key] = items
            logger.info(
                "Loaded %d items from %s", len(items), bench_key
            )

        return items_by_benchmark

    # ------------------------------------------------------------------
    # Problem formatting
    # ------------------------------------------------------------------

    def _format_problem(self, bench_key: str, item) -> str:
        """Convert a BenchmarkItem into a problem string for the LLM.

        Dispatches by benchmark type: entailment (FOLIO) vs
        multiple-choice (LogiQA).
        """
        if bench_key == "folio":
            return self._format_folio_problem(item)
        else:
            return self._format_mc_problem(item)

    @staticmethod
    def _format_folio_problem(item) -> str:
        """Format a FOLIO entailment problem."""
        lines = []
        if item.context:
            lines.append("Premises:")
            lines.append(item.context)
            lines.append("")
        lines.append(item.question)
        return "\n".join(lines)

    @staticmethod
    def _format_mc_problem(item) -> str:
        """Format a multiple-choice problem (LogiQA, etc.)."""
        lines = []
        if item.context:
            lines.append("Passage:")
            lines.append(item.context)
            lines.append("")
        lines.append(f"Question: {item.question}")
        lines.append("")
        labels = "ABCDEFGH"
        for i, choice in enumerate(item.choices):
            label = labels[i] if i < len(labels) else str(i)
            lines.append(f"{label}. {choice}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_generation_prompt(self, bench_key: str, problem_text: str) -> str:
        """Build the system + user prompt for the OpenAI API."""
        if bench_key == "folio":
            answer_instruction = (
                'Your final answer must be exactly one of: "True", "False", or "Unknown".'
            )
        else:
            answer_instruction = (
                'Your final answer must be exactly one letter: "A", "B", "C", or "D".'
            )

        prompt = f"""\
You are solving a logic reasoning problem. Show your work using \
semi-formal notation, then give your final answer.

=== Notation rules ===
- {{if A, then B}} for "if A then B" (implication)
- {{A and B}} for "A and B" (conjunction)
- {{A or B}} for "A or B" (disjunction)
- {{it is not the case that A}} for "not A" (negation)
- {{A if and only if B}} for "A iff B" (biconditional)
- {{for all x, P(x)}} for universal quantification
- {{there exist x such that P(x)}} for existential quantification
- Plain text for atomic facts (e.g., "the patient has a fever")

Curly brackets disambiguate nested compound formulas. \
Atomic propositions do NOT get brackets.

=== Proof format ===
Interleave natural language reasoning with formal inference steps. \
Mark each inference step with <PREMISE> and <CONCLUSION> tags:

<PREMISE> first premise </PREMISE>
<PREMISE> {{if first premise, then second conclusion}} </PREMISE>
<CONCLUSION> second conclusion </CONCLUSION>

You may write multiple inference steps. Each step should have at \
least one <PREMISE> and exactly one <CONCLUSION>.

=== Output format ===
Return a JSON object with exactly two fields:
- "reasoning": your full chain-of-thought with interleaved formal steps
- "answer": your final answer

{answer_instruction}

=== Examples ===
The following examples demonstrate how to solve problems using the \
semi-formal notation described above.
{_FEW_SHOT_EXAMPLES}

=== Problem ===
{problem_text}
"""
        return prompt

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process_item(self, bench_key: str, item) -> Optional[Stage2Example]:
        """Process a single benchmark item: prompt → parse → filter → verify."""
        problem_text = self._format_problem(bench_key, item)
        prompt = self._build_generation_prompt(bench_key, problem_text)

        response_text = self._call_api(prompt)
        parsed = self._parse_response(response_text)
        if parsed is None:
            logger.debug("Failed to parse response for %s", item.id)
            return None

        reasoning, predicted_answer = parsed

        # Check correctness
        correct_answer = self._get_correct_answer(bench_key, item)
        is_correct = self._check_correctness(
            bench_key, predicted_answer, correct_answer, item
        )
        if not is_correct:
            logger.debug(
                "Incorrect answer for %s: predicted=%s, correct=%s",
                item.id, predicted_answer, correct_answer,
            )
            return None

        # Optional Z3 verification
        verification_status = False
        if self.config.verify_with_z3 and self.verifier is not None:
            verification_status = self._verify_formal_steps(reasoning)
            if not verification_status:
                logger.debug(
                    "Formal steps failed Z3 verification for %s", item.id
                )
                # Still keep the example — Z3 failure is not a hard filter
                # (the answer is correct, formal notation is present)

        return Stage2Example(
            benchmark=bench_key,
            item_id=item.id,
            question=problem_text,
            problem_text=problem_text,
            reasoning=reasoning,
            predicted_answer=predicted_answer,
            correct_answer=correct_answer,
            is_correct=True,
            verification_status=verification_status,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response_text: str
    ) -> Optional[Tuple[str, str]]:
        """Extract reasoning and answer from the model's JSON response.

        Returns ``(reasoning, answer)`` or ``None`` if parsing fails.
        """
        text = response_text.strip()

        # Strip optional markdown code fences
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("JSON decode error: %.200s...", text)
            return None

        reasoning = data.get("reasoning")
        answer = data.get("answer")
        if not reasoning or not answer:
            logger.debug("Missing reasoning or answer fields")
            return None

        return str(reasoning), str(answer).strip()

    # ------------------------------------------------------------------
    # Correctness checking
    # ------------------------------------------------------------------

    @staticmethod
    def _get_correct_answer(bench_key: str, item) -> str:
        """Get the ground truth answer string for a benchmark item."""
        if bench_key == "folio":
            return item.correct_answer_text  # "True", "False", or "Unknown"
        else:
            # Multiple-choice: map index to letter
            labels = "ABCDEFGH"
            idx = item.correct_answer
            if 0 <= idx < len(labels):
                return labels[idx]
            return item.correct_answer_text

    @staticmethod
    def _check_correctness(
        bench_key: str,
        predicted: str,
        correct: str,
        item,
    ) -> bool:
        """Check if the predicted answer matches the ground truth."""
        pred = predicted.strip().rstrip(".").strip()
        gold = correct.strip()

        if bench_key == "folio":
            # Normalize entailment labels
            pred_norm = pred.lower()
            gold_norm = gold.lower()
            # Handle "uncertain" as synonym for "unknown"
            if pred_norm == "uncertain":
                pred_norm = "unknown"
            if gold_norm == "uncertain":
                gold_norm = "unknown"
            return pred_norm == gold_norm
        else:
            # Multiple-choice: compare letters
            pred_letter = pred.upper().lstrip("(").rstrip(")")
            # Extract just the letter if the model wrote e.g. "A) ..."
            if pred_letter and pred_letter[0].isalpha():
                pred_letter = pred_letter[0]
            return pred_letter == gold.upper()

    # ------------------------------------------------------------------
    # Z3 verification of formal steps
    # ------------------------------------------------------------------

    def _verify_formal_steps(self, reasoning: str) -> bool:
        """Z3-verify each formal inference step in the reasoning.

        Reuses the extraction pattern from
        :meth:`AppliedChainGenerator._extract_proof_steps`.
        """
        if self.verifier is None:
            return False

        steps = self._extract_proof_steps(reasoning)
        if not steps:
            # No formal steps found — not necessarily an error
            return False

        for step_premises, step_conclusion in steps:
            try:
                result = self.verifier.verify_inference(
                    step_premises, step_conclusion,
                )
                if not result:
                    return False
            except Exception as exc:
                logger.debug(
                    "Verification raised for step (%s -> %s): %s",
                    step_premises, step_conclusion, exc,
                )
                return False

        return True

    @staticmethod
    def _extract_proof_steps(
        text: str,
    ) -> List[Tuple[List[str], str]]:
        """Extract inference steps from text with PREMISE/CONCLUSION tags.

        Each step is ``(premises, conclusion)`` where *premises* is a
        list of strings and *conclusion* is a single string.
        """
        steps: List[Tuple[List[str], str]] = []
        current_premises: List[str] = []

        tag_pattern = re.compile(
            r"<(PREMISE|CONCLUSION)>\s*(.*?)\s*</\1>",
            re.DOTALL,
        )

        for match in tag_pattern.finditer(text):
            tag = match.group(1)
            content = match.group(2).strip()
            if tag == "PREMISE":
                current_premises.append(content)
            elif tag == "CONCLUSION":
                if current_premises:
                    steps.append((list(current_premises), content))
                    current_premises = []

        return steps

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Call the OpenAI API with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "API call failed (attempt %d/%d): %s",
                    attempt + 1, self.config.retry_attempts, exc,
                )
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)

        raise last_exc  # type: ignore[misc]
