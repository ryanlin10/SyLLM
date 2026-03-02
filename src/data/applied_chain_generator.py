"""Generate applied reasoning examples using OpenAI API distillation.

Produces real-world multi-step reasoning problems expressed in the project's
semi-formal curly-bracket notation.  Each example includes a domain-specific
problem statement, formalized premises, a tagged proof trace, and an optional
Z3 verification pass.

The generated data is intended for fine-tuning reasoning models on applied
(as opposed to purely abstract) logical inference tasks.

Usage (programmatic)::

    from src.data.applied_chain_generator import (
        AppliedChainGenerator, AppliedGeneratorConfig,
    )

    config = AppliedGeneratorConfig(verify_with_z3=False)
    gen = AppliedChainGenerator(config)
    examples = gen.generate_examples("medical diagnosis", count=10)

See ``scripts/generate_applied_data.py`` for the CLI entry point.
"""

import json
import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None

from .schema import Annotation, Premise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain catalogue
# ---------------------------------------------------------------------------

APPLIED_DOMAINS: List[str] = [
    "economic policy and markets",
    "ethical dilemmas and philosophy",
    "scientific hypothesis testing",
    "legal reasoning and contracts",
    "medical diagnosis",
    "engineering trade-offs",
    "strategic decision-making",
    "causal reasoning about events",
    "tool selection and usage",
    "resource allocation",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AppliedExample:
    """A single applied reasoning example with formal proof trace."""

    domain: str
    problem: str                  # real-world problem statement
    premises: List[str]           # formalized premises in bracket notation
    proof_trace: str              # tagged proof trace in bracket notation
    final_answer: str             # answer to the problem
    target_conclusion: str        # formal conclusion in bracket notation
    verification_status: bool     # whether Z3 verified the proof trace


@dataclass
class AppliedGeneratorConfig:
    """Configuration for the applied chain generator."""

    api_key: Optional[str] = None
    model: str = "gpt-5.2-2025-12-11"
    max_tokens: int = 4096
    examples_per_domain: int = 50
    min_steps: int = 2
    max_steps: int = 5
    verify_with_z3: bool = True
    retry_attempts: int = 3
    temperature: float = 0.7
    retry_delay: float = 1.0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AppliedChainGenerator:
    """Generate applied reasoning examples via OpenAI distillation.

    Each example is produced by prompting an OpenAI model to create a
    realistic reasoning problem in a given domain, formalize the premises
    using curly-bracket notation, and derive a conclusion through an
    explicit proof trace.  Steps can optionally be verified with Z3.
    """

    def __init__(self, config: AppliedGeneratorConfig = None):
        self.config = config or AppliedGeneratorConfig()

        # ----- OpenAI client -----
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if openai is None:
            raise ImportError(
                "Install openai: pip install openai"
            )
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

    def generate_examples(
        self, domain: str, count: int
    ) -> List[AppliedExample]:
        """Generate applied reasoning examples for a domain.

        Steps for each example:
        1. Prompt the model to create a real-world problem in *domain*.
        2. The model formalizes premises in bracket notation.
        3. The model produces a step-by-step proof trace with
           ``<PREMISE>``/``<CONCLUSION>`` tags.
        4. Optionally Z3-verify each inference step.
        5. Keep only verified examples (or all if ``verify_with_z3`` is
           ``False``).

        Parameters
        ----------
        domain:
            One of the entries in :data:`APPLIED_DOMAINS` (or any free-text
            domain description).
        count:
            Number of examples to attempt to generate.  The returned list
            may be smaller if some examples fail parsing or verification.

        Returns
        -------
        List[AppliedExample]
            Successfully generated (and, if configured, verified) examples.
        """
        prompt = self._build_generation_prompt(domain)
        examples: List[AppliedExample] = []

        for i in range(count):
            try:
                response_text = self._call_api(prompt)
                example = self._parse_response(response_text, domain)
                if example is None:
                    logger.debug(
                        "Failed to parse response for domain=%s (attempt %d)",
                        domain, i + 1,
                    )
                    continue

                # Verification
                if self.config.verify_with_z3 and self.verifier is not None:
                    verified = self._verify_example(example)
                    example.verification_status = verified
                    if not verified:
                        logger.debug(
                            "Example failed Z3 verification (domain=%s, "
                            "attempt %d)", domain, i + 1,
                        )
                        continue
                else:
                    # No verification requested or verifier unavailable
                    example.verification_status = False

                examples.append(example)

            except Exception as exc:
                logger.warning(
                    "Error generating example (domain=%s, attempt %d): %s",
                    domain, i + 1, exc,
                )

        return examples

    def render_to_annotation(self, example: AppliedExample) -> Annotation:
        """Convert an :class:`AppliedExample` to :class:`Annotation` format.

        The mapping is:

        * ``premises`` -- the formalized premises as :class:`Premise` objects.
        * ``content`` -- the proof trace (this becomes the assistant
          response during fine-tuning).
        * ``verifier_notes`` -- serialized JSON containing the domain,
          verification status, final answer, and target conclusion.

        Parameters
        ----------
        example:
            The applied example to convert.

        Returns
        -------
        Annotation
        """
        premise_objects = [
            Premise(id=f"p{i + 1}", text=text)
            for i, text in enumerate(example.premises)
        ]

        verifier_notes = json.dumps({
            "domain": example.domain,
            "problem": example.problem,
            "verification_status": example.verification_status,
            "final_answer": example.final_answer,
            "target_conclusion": example.target_conclusion,
        }, ensure_ascii=False)

        return Annotation(
            id=str(uuid.uuid4()),
            premises=premise_objects,
            content=example.proof_trace,
            verifier_notes=verifier_notes,
            annotator_id=f"applied_chain_generator_{example.domain}",
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_generation_prompt(self, domain: str) -> str:
        """Build the user prompt for the OpenAI API.

        The prompt instructs the model to:
        1. Create a realistic problem in the given domain.
        2. Identify 2-5 premises expressed using curly-bracket notation.
        3. Produce a multi-step proof trace using ``<PREMISE>`` and
           ``<CONCLUSION>`` tags.
        4. State the final answer and formal conclusion.
        5. Return a single JSON object.
        """
        min_s = self.config.min_steps
        max_s = self.config.max_steps

        prompt = f"""\
You are generating training data for a reasoning model that uses \
semi-formal curly-bracket notation.

=== Notation rules ===
- {{if A, then B}} for "if A then B" (implication)
- {{A and B}} for "A and B" (conjunction)
- {{A or B}} for "A or B" (disjunction)
- {{it is not the case that A}} for "not A" (negation)
- {{A if and only if B}} for "A iff B" (biconditional)
- {{for all x, P(x)}} for universal quantification
- {{there exist x such that P(x)}} for existential quantification
- Plain text for atomic facts (e.g., "the patient has a fever")

Curly brackets are used to disambiguate nested compound formulas.  \
Atomic propositions do NOT get brackets.

=== Proof trace format ===
Each inference step lists the premises used and the derived conclusion \
using XML-like tags.  A multi-step proof chains these together:

<PREMISE> the patient has a fever </PREMISE>
<PREMISE> {{if the patient has a fever, then an infection is likely}} </PREMISE>
<CONCLUSION> an infection is likely </CONCLUSION>
<PREMISE> an infection is likely </PREMISE>
<PREMISE> {{if an infection is likely, then antibiotics should be considered}} </PREMISE>
<CONCLUSION> antibiotics should be considered </CONCLUSION>

=== Task ===
Generate a realistic reasoning problem in the domain of: **{domain}**

Requirements:
1. Write a clear, specific problem statement grounded in {domain}.
2. Identify 2-5 premises.  Express each premise using the curly-bracket \
notation where appropriate (compound formulas MUST use brackets; plain \
atomic facts do not).
3. Produce a {min_s}-{max_s} step proof trace.  Each step must be a \
valid logical inference (modus ponens, conjunction elimination, \
disjunctive syllogism, etc.).  Use the <PREMISE> and <CONCLUSION> tags \
exactly as shown above.
4. State a clear final answer to the problem.
5. State the formal target conclusion in bracket notation.

Return ONLY a single JSON object (no markdown fences, no commentary) \
with exactly these fields:

{{
    "problem": "A clear description of the real-world reasoning problem.",
    "premises": [
        "first premise in bracket notation",
        "second premise in bracket notation"
    ],
    "proof_trace": "<PREMISE> ... </PREMISE>\\n<PREMISE> ... </PREMISE>\\n<CONCLUSION> ... </CONCLUSION>\\n...",
    "final_answer": "The answer to the problem in plain language.",
    "target_conclusion": "the formal conclusion in bracket notation"
}}
"""
        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response_text: str, domain: str
    ) -> Optional[AppliedExample]:
        """Parse the model's JSON response into an :class:`AppliedExample`.

        Handles responses that may be wrapped in markdown code fences.
        Returns ``None`` if parsing fails or required fields are missing.
        """
        text = response_text.strip()

        # Strip optional markdown code fences
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1)
        else:
            # Try to extract a top-level JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("JSON decode error for response: %.200s...", text)
            return None

        # Validate required keys
        required = ("problem", "premises", "proof_trace",
                     "final_answer", "target_conclusion")
        if not all(k in data for k in required):
            missing = [k for k in required if k not in data]
            logger.debug("Missing required fields: %s", missing)
            return None

        premises = data["premises"]
        if not isinstance(premises, list) or len(premises) < 1:
            logger.debug("Premises field is not a non-empty list.")
            return None

        return AppliedExample(
            domain=domain,
            problem=str(data["problem"]),
            premises=[str(p) for p in premises],
            proof_trace=str(data["proof_trace"]),
            final_answer=str(data["final_answer"]),
            target_conclusion=str(data["target_conclusion"]),
            verification_status=False,
        )

    # ------------------------------------------------------------------
    # Z3 verification
    # ------------------------------------------------------------------

    def _verify_example(self, example: AppliedExample) -> bool:
        """Z3-verify each step in the proof trace.

        Parses the proof trace into individual inference steps by
        extracting sequences of ``<PREMISE>...</PREMISE>`` blocks
        followed by a ``<CONCLUSION>...</CONCLUSION>`` block.  For each
        step, calls :meth:`VerifierPipeline.verify_inference`.

        Returns ``True`` only if **all** steps verify successfully.
        """
        if self.verifier is None:
            return False

        steps = self._extract_proof_steps(example.proof_trace)
        if not steps:
            logger.debug("No proof steps extracted from trace.")
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
        proof_trace: str,
    ) -> List[Tuple[List[str], str]]:
        """Extract individual inference steps from a tagged proof trace.

        Each step is a tuple ``(premises, conclusion)`` where *premises*
        is a list of strings and *conclusion* is a single string.

        The trace format is a sequence of ``<PREMISE>`` blocks followed
        by a ``<CONCLUSION>`` block, possibly repeated.
        """
        steps: List[Tuple[List[str], str]] = []
        current_premises: List[str] = []

        # Match all PREMISE and CONCLUSION tags
        tag_pattern = re.compile(
            r"<(PREMISE|CONCLUSION)>\s*(.*?)\s*</\1>",
            re.DOTALL,
        )

        for match in tag_pattern.finditer(proof_trace):
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
        """Call the OpenAI API with retry logic.

        Retries up to ``config.retry_attempts`` times with a delay of
        ``config.retry_delay`` seconds between attempts.

        Returns
        -------
        str
            The text content of the model's response.

        Raises
        ------
        Exception
            Re-raises the last exception after all retries are exhausted.
        """
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
