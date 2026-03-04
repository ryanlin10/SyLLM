"""Process + outcome reward scoring for RL training (Phase 3).

Implements a reward function that combines:
- **Process reward** (verifier reward): ``log(1 + sound_steps)`` — logarithmically
  increasing with the number of Z3-verified sound inference steps, giving
  diminishing returns and an incentive to produce more reasoning steps without
  an unbounded arms race.
- **Outcome reward**: ``correct_reward`` (+1.0) if correct, ``wrong_reward`` (-1.0)
  if a wrong verdict is given, or 0.0 if no verdict is detected at all (no penalty
  for not attempting).

The combined reward uses an **outcome-gated** process bonus::

    if correct:    reward = outcome_weight * correct_reward
                          + verifier_weight * log(1 + sound_steps)
    if wrong:      reward = outcome_weight * wrong_reward      # full penalty, no offset
    if no verdict: reward = verifier_weight * log(1 + sound_steps)  # encourage reasoning

The key property: the process reward **cannot offset a wrong-answer penalty**.
A model that reasons extensively but reaches the wrong conclusion still receives
the full ``outcome_weight * wrong_reward`` penalty.  However, the process reward
is included for the no-verdict case so that responses that produce reasoning steps
receive a small positive gradient even when the model has not yet learned the
output format — preventing the KL penalty from dominating and erasing reasoning.

The incentive ordering is:

    correct + thorough reasoning  >  correct + little reasoning
    >  no verdict + thorough reasoning  >  no verdict + no reasoning
    >  wrong (regardless of reasoning)

Examples (verifier_weight=0.3, outcome_weight=0.7, correct_reward=1.0,
wrong_reward=-1.0, natural log):

- 10 steps, correct:     0.7 * 1.0 + 0.3 * ln(11) ≈ +1.42
-  0 steps, correct:     0.7 * 1.0 + 0.3 * ln(1)   = +0.70
- 10 steps, no verdict:  0.3 * ln(11)              ≈ +0.72
-  0 steps, no verdict:  0.3 * ln(1)                = +0.00
- wrong (any steps):     0.7 * -1.0                 = -0.70
- max correct/wrong gap at 10 steps: 1.42 − (−0.70)  = 2.12

Proof traces use XML-style tags::

    <PREMISE> fact1 </PREMISE>
    <PREMISE> fact2 </PREMISE>
    <CONCLUSION> derived </CONCLUSION>

Each ``<CONCLUSION>`` terminates a segment. All ``<PREMISE>`` tags between
the previous conclusion (or start of text) and the current conclusion form
the premises of that inference step.
"""

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as _mp
    _FUTURES_AVAILABLE = True
except ImportError:
    _FUTURES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level for ProcessPoolExecutor
# pickling with "spawn" start method).
#
# WHY ProcessPoolExecutor?
# Z3's Python bindings use a module-global Context (z3.main_ctx()) by default.
# All Solver / BoolRef objects created without an explicit ctx share this
# global context.  Calling solver.check() from multiple threads simultaneously
# corrupts Z3's internal AST, causing C++ assertion violations.
# ProcessPoolExecutor with "spawn" gives each worker its own Python process
# (and thus its own Z3 global context), eliminating the race completely.
# The IPC overhead for string serialisation is negligible vs Z3 solve time.
# ---------------------------------------------------------------------------

# Per-worker process state (populated by _init_verify_worker).
_WORKER_PIPELINE: Any = None


def _init_verify_worker(timeout_ms: int) -> None:
    """Run once in each worker process to initialise the Z3 verifier."""
    global _WORKER_PIPELINE
    # Add project root to sys.path so relative imports work in spawned process.
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        from src.verification.verifier import VerifierPipeline, VerifierConfig  # noqa: PLC0415
        _WORKER_PIPELINE = VerifierPipeline(VerifierConfig(timeout_ms=timeout_ms))
    except Exception:
        _WORKER_PIPELINE = None  # fall back to unsound for this worker


def _verify_segment_worker(premises: List[str], conclusion: str) -> bool:
    """Verify one inference step inside a worker process."""
    if _WORKER_PIPELINE is None or not premises or not conclusion:
        return False
    try:
        return _WORKER_PIPELINE.verify_inference(premises, conclusion)
    except Exception:
        return False


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
# Pattern to find premise/conclusion/assume/discharge tags in order for segment extraction
_TAG_RE = re.compile(
    r"<(PREMISE|CONCLUSION|ASSUME|DISCHARGE)>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)
# FOLIO-style verdict line: "Verdict: True/False/Unknown" or "Answer: True/False/Unknown"
_VERDICT_RE = re.compile(
    r"(?:Verdict|Answer)\s*:\s*(True|False|Unknown)\b",
    re.IGNORECASE,
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
    """Combined verifier + outcome reward for RL training.

    Extracts ``<PREMISE>``/``<CONCLUSION>`` segments from model output,
    Z3-verifies each step (process/verifier reward), checks whether the
    output reaches the correct answer (outcome reward), and combines them
    as a weighted sum::

        reward = verifier_weight * process_reward + outcome_weight * outcome_reward

    Parameters
    ----------
    verifier_config : optional
        A ``VerifierConfig`` instance.  If ``None``, the default config
        is used.  If Z3 is not installed, the verifier degrades to an
        optimistic fallback (all steps treated as sound).
    verifier_weight : float
        Weight for the process/verifier reward component (default: 0.3).
    outcome_weight : float
        Weight for the outcome (correct answer) reward component (default: 0.7).
    """

    def __init__(
        self,
        verifier_config=None,
        verifier_weight: float = 0.3,
        outcome_weight: float = 0.7,
        correct_reward: float = 1.0,
        wrong_reward: float = -1.0,
        skip_verify: bool = False,
        n_verify_workers: int = 0,
    ):
        self.verifier_weight = verifier_weight
        self.outcome_weight = outcome_weight
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.skip_verify = skip_verify
        try:
            from ..verification.verifier import VerifierPipeline, VerifierConfig

            cfg = verifier_config if verifier_config is not None else VerifierConfig()
            self.verifier = VerifierPipeline(cfg)
            self._z3_available = True
        except (ImportError, Exception):
            self.verifier = None
            self._z3_available = False

        # Parallel Z3 verification config.
        # Workers are spawned per scoring call (context manager) rather than
        # kept alive persistently.  Persistent idle workers get killed between
        # scoring calls (~70 s gap during generate+backward) — by the OOM
        # killer or Z3 C-level issues — causing BrokenProcessPool every step.
        # Per-call pools exist only for the 1-2 s of actual Z3 work and are
        # cleaned up immediately; no idle worker survives to be killed.
        # Startup overhead is skipped entirely when no tagged segments exist.
        self._pool_timeout_ms: int = 5000
        self._n_verify_workers: int = n_verify_workers  # max workers per call
        self._use_parallel: bool = (
            n_verify_workers > 0
            and not skip_verify
            and self._z3_available
            and _FUTURES_AVAILABLE
        )

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
            One of ``"free_form"``, ``"multiple_choice"``, ``"proof"``, or
            ``"verdict"`` (FOLIO-style "Verdict: True/False/Unknown" output).

        Returns
        -------
        RewardResult
            Combined reward and diagnostics.
        """
        # 1. Extract proof segments from output
        segments = self._extract_segments(output)

        # 2. Compute process reward: log(1 + sound_steps).
        #    Logarithmic scaling gives diminishing returns on additional steps
        #    and avoids a degenerate incentive to maximise sequence length.
        #    When skip_verify=True, all steps are treated as sound (optimistic).
        if not segments:
            process_reward = 0.0
            sound_steps = 0
            total_steps = 0
        elif self.skip_verify:
            total_steps = len(segments)
            sound_steps = total_steps  # optimistic: all steps assumed sound
            process_reward = math.log(1 + sound_steps)
        else:
            total_steps = len(segments)
            sound_steps = sum(1 for s in segments if self._verify_segment(s))
            process_reward = math.log(1 + sound_steps)

        # 3. Three-way outcome check.
        #    Returns True (correct), False (wrong verdict given), or None (no verdict).
        goal_result = self._check_goal_match(output, target, task_type)

        # 4. Outcome-gated reward combination.
        #    Process bonus only applies to correct answers so it cannot offset
        #    the wrong-answer penalty (which is always the full outcome_weight * wrong_reward).
        if goal_result is True:
            outcome_reward = self.correct_reward
            reward = (self.outcome_weight * self.correct_reward
                      + self.verifier_weight * process_reward)
        elif goal_result is False:
            outcome_reward = self.wrong_reward
            reward = self.outcome_weight * self.wrong_reward  # no process offset
        else:  # None — no verdict detected; encourage reasoning without penalising
            outcome_reward = 0.0
            reward = self.verifier_weight * process_reward

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
                "verifier_weight": self.verifier_weight,
                "outcome_weight": self.outcome_weight,
                "correct_reward": self.correct_reward,
                "wrong_reward": self.wrong_reward,
            },
        )

    def score_batch(
        self,
        outputs: List[str],
        targets: List[str],
        task_types: Optional[List[str]] = None,
    ) -> List[RewardResult]:
        """Score a batch of model outputs.

        When a thread pool is available (``n_verify_workers > 0``), all Z3
        segment verifications across the entire batch are submitted in parallel
        before any results are collected, giving near-linear speedup up to the
        number of workers.

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
        if self._use_parallel:
            return self._score_batch_parallel(outputs, targets, task_types)
        return [
            self.score(o, t, tt)
            for o, t, tt in zip(outputs, targets, task_types)
        ]

    def _score_batch_parallel(
        self,
        outputs: List[str],
        targets: List[str],
        task_types: List[str],
    ) -> List[RewardResult]:
        """Score a batch with all Z3 verifications running in parallel.

        A fresh ``ProcessPoolExecutor`` is created as a context manager for
        each call and torn down immediately after.  This avoids the persistent
        idle-worker problem: workers that sit idle for the ~70 s between
        scoring calls (during generate + backward) get killed by the OS or
        Z3 C-level issues, causing ``BrokenProcessPool`` on every step.
        Per-call pools exist only for the ~1-2 s of actual Z3 work.

        If no tagged segments exist in the batch, pool creation is skipped
        entirely (zero overhead for responses without proof structure).
        """
        import logging
        _logger = logging.getLogger(__name__)

        # 1. Extract all segments upfront (fast, no Z3).
        all_segs: List[List[Segment]] = [self._extract_segments(o) for o in outputs]

        # 2. Flatten to (output_index, segment) task list.
        tasks: List[Tuple[int, Segment]] = [
            (i, seg) for i, segs in enumerate(all_segs) for seg in segs
        ]

        sound_steps = [0] * len(outputs)

        if tasks:
            # Cap workers to actual task count — no point spawning 16 workers for 3 tasks.
            n_workers = min(self._n_verify_workers, len(tasks))
            try:
                import os
                # Limit OpenBLAS/OMP/MKL threads in spawned workers to 1.
                # Spawned workers re-import the main script, pulling in scipy/sklearn/
                # transformers, which cause OpenBLAS to try to create 64 threads and
                # hit RLIMIT_NPROC.  These env vars are inherited by spawn children
                # and are read by OpenBLAS before it initialises its thread pool.
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                spawn_ctx = _mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=spawn_ctx,
                    initializer=_init_verify_worker,
                    initargs=(self._pool_timeout_ms,),
                ) as pool:
                    futures_to_idx: Dict[Any, int] = {
                        pool.submit(_verify_segment_worker, seg.premises, seg.conclusion): i
                        for i, seg in tasks
                    }
                    for fut in as_completed(futures_to_idx):
                        idx = futures_to_idx[fut]
                        try:
                            if fut.result():
                                sound_steps[idx] += 1
                        except Exception:
                            pass  # verification error → unsound
            except Exception as exc:
                _logger.warning(
                    "Parallel Z3 pool failed (%s); segments treated as unsound.", exc
                )

        # 4. Build RewardResult for each output using outcome-gated formula.
        results: List[RewardResult] = []
        for i, (output, target, task_type) in enumerate(zip(outputs, targets, task_types)):
            segs = all_segs[i]
            total_steps = len(segs)
            s_steps = sound_steps[i]
            process_reward = math.log(1 + s_steps)

            goal_result = self._check_goal_match(output, target, task_type)

            if goal_result is True:
                outcome_reward = self.correct_reward
                reward = (self.outcome_weight * self.correct_reward
                          + self.verifier_weight * process_reward)
            elif goal_result is False:
                outcome_reward = self.wrong_reward
                reward = self.outcome_weight * self.wrong_reward
            else:  # None — no verdict
                outcome_reward = 0.0
                reward = self.verifier_weight * process_reward

            results.append(RewardResult(
                reward=reward,
                process_reward=process_reward,
                outcome_reward=outcome_reward,
                total_steps=total_steps,
                sound_steps=s_steps,
                details={
                    "task_type": task_type,
                    "z3_available": self._z3_available,
                    "segments_found": len(segs),
                    "verifier_weight": self.verifier_weight,
                    "outcome_weight": self.outcome_weight,
                    "correct_reward": self.correct_reward,
                    "wrong_reward": self.wrong_reward,
                },
            ))

        return results

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _extract_segments(self, text: str) -> List[Segment]:
        """Extract ``<PREMISE>``/``<CONCLUSION>`` segments from *text*.

        Each segment consists of the ``<PREMISE>`` tags that appear before
        a ``<CONCLUSION>`` tag (back to the previous conclusion or start
        of text).  ``<ASSUME>`` tags push an assumption into scope for all
        subsequent segments until the matching ``<DISCHARGE>`` removes it.
        This ensures that sub-steps inside assumption blocks are verified
        with the assumption available as a premise.

        Returns
        -------
        list of Segment
            One segment per ``<CONCLUSION>`` tag found.  Segments with no
            premises (after injecting active assumptions) are still included
            (they will fail verification unless the assumption itself
            trivially entails the conclusion).
        """
        segments: List[Segment] = []
        pending_premises: List[str] = []
        # Assumptions currently in scope (pushed by <ASSUME>, popped by <DISCHARGE>)
        active_assumptions: List[str] = []

        for match in _TAG_RE.finditer(text):
            tag_name = match.group(1).upper()
            content = match.group(2).strip()

            if tag_name == "ASSUME":
                active_assumptions.append(content)
            elif tag_name == "PREMISE":
                pending_premises.append(content)
            elif tag_name == "CONCLUSION":
                # Inject currently-active assumptions as extra premises so that
                # every step inside an assumption block is verifiable with the
                # assumption in scope.  This makes the assumption-introduction
                # step itself check ``A |= A`` (trivially valid) and ensures
                # inner steps have the assumed formula available to Z3.
                all_premises = active_assumptions + pending_premises
                segments.append(
                    Segment(
                        premises=all_premises,
                        conclusion=content,
                    )
                )
                pending_premises = []
            elif tag_name == "DISCHARGE":
                # Remove the discharged assumption from scope (search from the
                # end so nested assumptions with the same text are handled LIFO).
                for i in range(len(active_assumptions) - 1, -1, -1):
                    if active_assumptions[i] == content:
                        active_assumptions.pop(i)
                        break

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
    ) -> Optional[bool]:
        """Check whether *output* reaches the expected *target* answer.

        Returns ``True`` (correct), ``False`` (wrong answer given), or
        ``None`` (no answer detected — model did not attempt).

        Dispatch by *task_type*:

        ``"verdict"``
            Finds the last ``Verdict: X`` line and compares to *target*.
            Returns ``None`` if no verdict line is present (no penalty).

        ``"multiple_choice"``
            Extracts an answer letter (A/B/C/D); ``None`` if none found.

        ``"proof"``
            Checks the last ``<CONCLUSION>`` tag; ``False`` if no conclusion.

        ``"free_form"`` (default)
            Normalized substring match — always returns True or False.
        """
        if task_type == "multiple_choice":
            return self._match_multiple_choice(output, target)
        elif task_type == "proof":
            return self._match_proof(output, target)
        elif task_type == "verdict":
            return self._match_verdict(output, target)
        else:
            return self._match_free_form(output, target)

    # -- multiple choice ------------------------------------------------

    def _match_multiple_choice(self, output: str, target: str) -> Optional[bool]:
        """Extract answer letter from *output* and compare to *target*.

        Returns ``None`` if no answer letter can be extracted (no penalty).
        """
        target_letter = self._normalize_mc(target)
        if not target_letter:
            return False

        match = _MC_ANSWER_RE.search(output)
        if match:
            extracted = next(
                (g.upper() for g in match.groups() if g is not None), None
            )
            if extracted is not None:
                return extracted == target_letter

        return None  # no answer detected

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

    # -- verdict (FOLIO) ------------------------------------------------

    # Compiled patterns for verdict tag and last-line fallback
    _VERDICT_TAG_RE = re.compile(
        r"<VERDICT>\s*(.*?)\s*</VERDICT>", re.DOTALL | re.IGNORECASE
    )
    _LABEL_RE = re.compile(r"\b(True|False|Unknown)\b", re.IGNORECASE)

    def _match_verdict(self, output: str, target: str) -> Optional[bool]:
        """Match a FOLIO-style verdict from model output.

        Tries three patterns in order, returning on the first match:

        1. Explicit ``Verdict: X`` or ``Answer: X`` line (``_VERDICT_RE``).
        2. Last ``<VERDICT>`` tag — extracts True/False/Unknown from its content.
        3. Last line of output — scans for a bare True/False/Unknown label.

        Returns:
            True  — a label was found and it matches *target*.
            False — a label was found but it does not match *target*.
            None  — no recognisable label detected; no penalty applied.
        """
        target_lower = target.strip().lower()

        # 1. Explicit "Verdict: X" or "Answer: X"
        matches = _VERDICT_RE.findall(output)
        if matches:
            return matches[-1].strip().lower() == target_lower

        # 2. Last <VERDICT>...</VERDICT> tag — look for a label inside
        tag_matches = self._VERDICT_TAG_RE.findall(output)
        if tag_matches:
            tag_content = tag_matches[-1].strip()
            label_match = self._LABEL_RE.search(tag_content)
            if label_match:
                return label_match.group(1).lower() == target_lower
            # Tag found but no label inside — treat as wrong verdict
            return False

        # 3. Last non-empty line ending with a bare label word
        for line in reversed(output.splitlines()):
            line = line.strip()
            if not line:
                continue
            label_match = self._LABEL_RE.search(line)
            if label_match:
                # Only accept if the label is at the very end or the line IS the label
                if line.lower().endswith(label_match.group(1).lower()):
                    return label_match.group(1).lower() == target_lower
            break  # only check the last non-empty line

        return None  # no verdict detected

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
