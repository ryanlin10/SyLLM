"""Process + outcome reward scoring for RL training (Phase 3).

Implements a reward function that combines:
- **Process reward**: fraction of proof steps that are Z3-verified sound.
- **Outcome reward**: 1.0 if the model reaches the correct answer, else 0.0.

The combined reward is ``process_reward * outcome_reward``, which means:
- Trivial valid steps with wrong answer -> reward = 0 (blocked).
- Invalid steps -> process < 1 -> reward reduced even if answer is correct.
- All steps sound AND correct answer -> full reward.

Proof traces use XML-style tags::

    <PREMISE> fact1 </PREMISE>
    <PREMISE> fact2 </PREMISE>
    <CONCLUSION> derived </CONCLUSION>

Each ``<CONCLUSION>`` terminates a segment. All ``<PREMISE>`` tags between
the previous conclusion (or start of text) and the current conclusion form
the premises of that inference step.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single inference step extracted from model output."""

    premises: List[str]
    conclusion: str


@dataclass
class RewardResult:
    """Result of reward scoring."""

    reward: float  # final combined reward
    process_reward: float  # fraction of sound steps
    outcome_reward: float  # 1.0 if goal matched, else 0.0
    total_steps: int
    sound_steps: int
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compiled patterns (module-level for reuse)
# ---------------------------------------------------------------------------

_PREMISE_RE = re.compile(
    r"<PREMISE>\s*(.*?)\s*</PREMISE>", re.DOTALL | re.IGNORECASE
)
_CONCLUSION_RE = re.compile(
    r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", re.DOTALL | re.IGNORECASE
)
# Pattern to find premise/conclusion tags in order for segment extraction
_TAG_RE = re.compile(
    r"<(PREMISE|CONCLUSION)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE
)
# Multiple-choice answer extraction: "Answer: A", "(B)", "the answer is C", etc.
_MC_ANSWER_RE = re.compile(
    r"(?:"
    r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Da-d])\)?"
    r"|\(([A-Da-d])\)"
    r"|^([A-Da-d])(?:\.|$)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Main reward class
# ---------------------------------------------------------------------------

class SoundnessReward:
    """Process + outcome reward for RL training.

    Extracts ``<PREMISE>``/``<CONCLUSION>`` segments from model output,
    Z3-verifies each step, and combines with outcome match to produce
    a single scalar reward.

    Parameters
    ----------
    verifier_config : optional
        A ``VerifierConfig`` instance.  If ``None``, the default config
        is used.  If Z3 is not installed, the verifier degrades to an
        optimistic fallback (all steps treated as sound).
    """

    def __init__(self, verifier_config=None):
        try:
            from ..verification.verifier import VerifierPipeline, VerifierConfig

            cfg = verifier_config if verifier_config is not None else VerifierConfig()
            self.verifier = VerifierPipeline(cfg)
            self._z3_available = True
        except (ImportError, Exception):
            self.verifier = None
            self._z3_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        output: str,
        target: str,
        task_type: str = "free_form",
    ) -> RewardResult:
        """Score a model output against a target answer.

        Parameters
        ----------
        output:
            Model's generated text, potentially containing a proof trace
            with ``<PREMISE>`` / ``<CONCLUSION>`` tags.
        target:
            Expected answer or conclusion string.
        task_type:
            One of ``"free_form"``, ``"multiple_choice"``, or ``"proof"``.

        Returns
        -------
        RewardResult
            Combined reward and diagnostics.
        """
        # 1. Extract proof segments from output
        segments = self._extract_segments(output)

        # 2. Compute process reward (Z3 verification of each step)
        if not segments:
            process_reward = 0.0
            sound_steps = 0
            total_steps = 0
        else:
            total_steps = len(segments)
            sound_steps = sum(1 for s in segments if self._verify_segment(s))
            process_reward = sound_steps / total_steps

        # 3. Compute outcome reward (does the output reach the right answer?)
        outcome_reward = (
            1.0 if self._check_goal_match(output, target, task_type) else 0.0
        )

        # 4. Combine: reward = process * outcome
        reward = process_reward * outcome_reward

        return RewardResult(
            reward=reward,
            process_reward=process_reward,
            outcome_reward=outcome_reward,
            total_steps=total_steps,
            sound_steps=sound_steps,
            details={
                "task_type": task_type,
                "z3_available": self._z3_available,
                "segments_found": len(segments),
            },
        )

    def score_batch(
        self,
        outputs: List[str],
        targets: List[str],
        task_types: Optional[List[str]] = None,
    ) -> List[RewardResult]:
        """Score a batch of model outputs.

        Parameters
        ----------
        outputs:
            List of generated texts.
        targets:
            Corresponding expected answers.
        task_types:
            Per-example task types.  Defaults to ``"free_form"`` for all.

        Returns
        -------
        list of RewardResult
        """
        if task_types is None:
            task_types = ["free_form"] * len(outputs)
        return [
            self.score(o, t, tt)
            for o, t, tt in zip(outputs, targets, task_types)
        ]

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _extract_segments(self, text: str) -> List[Segment]:
        """Extract ``<PREMISE>``/``<CONCLUSION>`` segments from *text*.

        Each segment consists of the ``<PREMISE>`` tags that appear before
        a ``<CONCLUSION>`` tag (back to the previous conclusion or start
        of text).  This preserves the natural reading order of a proof
        trace where each conclusion depends on the premises directly
        preceding it.

        Returns
        -------
        list of Segment
            One segment per ``<CONCLUSION>`` tag found.  Segments with no
            premises are still included (they will fail verification).
        """
        segments: List[Segment] = []
        pending_premises: List[str] = []

        for match in _TAG_RE.finditer(text):
            tag_name = match.group(1).upper()
            content = match.group(2).strip()

            if tag_name == "PREMISE":
                pending_premises.append(content)
            elif tag_name == "CONCLUSION":
                segments.append(
                    Segment(
                        premises=list(pending_premises),
                        conclusion=content,
                    )
                )
                # Previous conclusions become available as implicit premises
                # for subsequent steps, but we start fresh for explicit tags.
                pending_premises = []

        return segments

    # ------------------------------------------------------------------
    # Z3 verification of a single step
    # ------------------------------------------------------------------

    def _verify_segment(self, segment: Segment) -> bool:
        """Z3-verify that *segment.conclusion* follows from *segment.premises*.

        If the verifier (Z3) is not available, returns ``True`` as an
        optimistic fallback so that training can proceed without Z3
        installed (process reward will always be 1.0 in that case, and
        the outcome reward alone gates the final score).

        Parameters
        ----------
        segment:
            The inference step to verify.

        Returns
        -------
        bool
        """
        if self.verifier is None:
            return True  # optimistic fallback

        if not segment.premises or not segment.conclusion:
            return False

        try:
            return self.verifier.verify_inference(
                segment.premises, segment.conclusion
            )
        except Exception:
            # Verification errors (parse failures, Z3 timeouts, etc.)
            # are treated as unsound rather than crashing training.
            return False

    # ------------------------------------------------------------------
    # Goal / outcome matching
    # ------------------------------------------------------------------

    def _check_goal_match(
        self, output: str, target: str, task_type: str
    ) -> bool:
        """Check whether *output* reaches the expected *target* answer.

        Dispatch by *task_type*:

        ``"multiple_choice"``
            Extract an answer letter (A/B/C/D) from *output* and compare
            case-insensitively to *target*.

        ``"proof"``
            Extract the last ``<CONCLUSION>`` from *output* and check
            whether it matches *target* (normalized string comparison,
            with Z3 equivalence as fallback).

        ``"free_form"`` (default)
            Normalized substring match, with Z3 equivalence as fallback.

        Parameters
        ----------
        output:
            Full model-generated text.
        target:
            Expected answer string.
        task_type:
            Determines matching strategy.

        Returns
        -------
        bool
        """
        if task_type == "multiple_choice":
            return self._match_multiple_choice(output, target)
        elif task_type == "proof":
            return self._match_proof(output, target)
        else:
            return self._match_free_form(output, target)

    # -- multiple choice ------------------------------------------------

    def _match_multiple_choice(self, output: str, target: str) -> bool:
        """Extract answer letter from *output* and compare to *target*.

        Looks for patterns such as ``Answer: A``, ``(B)``, ``the answer
        is C``, or a bare letter at the start of a line.
        """
        target_letter = self._normalize_mc(target)
        if not target_letter:
            return False

        match = _MC_ANSWER_RE.search(output)
        if match:
            # The pattern has three groups; exactly one will be non-None
            extracted = next(
                (g.upper() for g in match.groups() if g is not None), None
            )
            return extracted == target_letter

        # Fallback: check if the target letter appears as a standalone
        # token (surrounded by whitespace/punctuation) in the output.
        pattern = rf"(?<![A-Za-z]){re.escape(target_letter)}(?![A-Za-z])"
        return bool(re.search(pattern, output, re.IGNORECASE))

    @staticmethod
    def _normalize_mc(text: str) -> Optional[str]:
        """Normalize a multiple-choice target to a single uppercase letter."""
        text = text.strip().strip("()").strip().upper()
        if len(text) == 1 and text in "ABCD":
            return text
        return None

    # -- proof ----------------------------------------------------------

    def _match_proof(self, output: str, target: str) -> bool:
        """Check if the final ``<CONCLUSION>`` matches *target*."""
        conclusions = _CONCLUSION_RE.findall(output)
        if not conclusions:
            return False

        last_conclusion = conclusions[-1].strip()
        if self._normalized_eq(last_conclusion, target):
            return True

        # Z3 equivalence fallback
        return self._z3_equivalent(last_conclusion, target)

    # -- free form ------------------------------------------------------

    def _match_free_form(self, output: str, target: str) -> bool:
        """Normalized substring match with Z3 equivalence fallback."""
        norm_output = self._normalize_text(output)
        norm_target = self._normalize_text(target)

        if not norm_target:
            return False

        # Direct substring containment
        if norm_target in norm_output:
            return True

        # Z3 equivalence fallback: parse both as formulas and check
        # mutual entailment.
        return self._z3_equivalent(output, target)

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip whitespace, collapse multiple spaces."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalized_eq(a: str, b: str) -> bool:
        """Case-insensitive, whitespace-normalized equality."""
        return (
            re.sub(r"\s+", " ", a.strip().lower())
            == re.sub(r"\s+", " ", b.strip().lower())
        )

    def _z3_equivalent(self, text_a: str, text_b: str) -> bool:
        """Check if *text_a* and *text_b* are logically equivalent via Z3.

        Constructs two entailment checks (a |= b and b |= a).  Returns
        ``True`` only if both hold.  Returns ``False`` on any error or
        if Z3 is unavailable.
        """
        if self.verifier is None:
            return False

        try:
            forward = self.verifier.verify_inference([text_a], text_b)
            backward = self.verifier.verify_inference([text_b], text_a)
            return forward and backward
        except Exception:
            return False
