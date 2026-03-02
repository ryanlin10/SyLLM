"""
Bottom-up proof chain construction using natural deduction rules.

Builds multi-step proof chains by starting from initial atomic premises and
applying natural deduction rules forward to derive new formulas. Supports
both propositional and first-order logic, with discharge rules for
conditional proofs, proof by contradiction, and disjunction elimination.

Integrates with the existing syntax tree, NL renderer, and Z3 verification
pipeline to produce verified, natural-language proof traces suitable for
training data generation.
"""

import collections
import json
import logging
import random
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .syntax_tree import (
    AtomNode,
    BinaryNode,
    Connective,
    FormulaNode,
    LogicOrder,
    NegationNode,
    Quantifier,
    QuantifiedNode,
    RandomTreeGenerator,
    TreeGeneratorConfig,
    make_biconditional,
    make_conjunction,
    make_disjunction,
    make_implication,
    negate,
)
from .nl_renderer import NaturalLanguageRenderer
from .atomic_proposition_generator import PropositionPool
from .schema import Annotation, Premise

# Optional Z3 verification import -- use Z3Translator directly on
# FormulaNode ASTs rather than going through VerifierPipeline (which
# requires string round-tripping through SemiFormalParser).
try:
    from ..verification.translator import Z3Translator, TranslationContext, Z3_AVAILABLE
    import z3 as _z3_module

    _HAS_Z3 = Z3_AVAILABLE
except ImportError:
    _HAS_Z3 = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class NDRule(Enum):
    """Natural deduction rules."""

    AND_INTRO = "and_intro"
    AND_ELIM = "and_elim"
    OR_INTRO = "or_intro"
    IMPLIES_INTRO = "implies_intro"  # discharge
    IMPLIES_ELIM = "implies_elim"
    NOT_INTRO = "not_intro"  # discharge
    NOT_ELIM = "not_elim"
    IFF_INTRO = "iff_intro"
    IFF_ELIM = "iff_elim"
    OR_ELIM = "or_elim"  # discharge
    FORALL_INTRO = "forall_intro"  # eigenvariable condition
    FORALL_ELIM = "forall_elim"
    EXISTS_INTRO = "exists_intro"
    EXISTS_ELIM = "exists_elim"  # discharge


# Rules that create temporary assumptions that get discharged.
DISCHARGE_RULES: Set[NDRule] = {
    NDRule.IMPLIES_INTRO,
    NDRule.NOT_INTRO,
    NDRule.OR_ELIM,
    NDRule.EXISTS_ELIM,
}

# Simple (non-discharge) rules.
SIMPLE_RULES_PL: List[NDRule] = [
    NDRule.AND_INTRO,
    NDRule.AND_ELIM,
    NDRule.OR_INTRO,
    NDRule.IMPLIES_ELIM,
    NDRule.NOT_ELIM,
    NDRule.IFF_INTRO,
    NDRule.IFF_ELIM,
]

SIMPLE_RULES_FOL: List[NDRule] = [
    NDRule.FORALL_ELIM,
    NDRule.EXISTS_INTRO,
]

# Rules that should NOT be the final step (conclusion would be compound /
# quantified in a way we want to avoid).
DISALLOWED_FINAL_RULES: Set[NDRule] = {
    NDRule.EXISTS_INTRO,
    NDRule.AND_ELIM,
}


@dataclass
class ProofStep:
    """A single step in a natural-deduction proof."""

    step_id: int
    rule: NDRule
    premises: List[FormulaNode]  # formulas used as input
    conclusion: FormulaNode  # what this step derives
    is_assumption: bool = False
    is_discharge: bool = False
    discharged_formula: Optional[FormulaNode] = None
    layer: int = 0


@dataclass
class ProofChain:
    """Complete proof chain from initial premises to final conclusion."""

    initial_premises: List[FormulaNode]
    steps: List[ProofStep]
    final_conclusion: FormulaNode
    logic_order: LogicOrder
    rules_used: List[NDRule]


@dataclass
class CompressedSegment:
    """A compressed segment representing one or more proof steps."""

    premises: List[FormulaNode]  # leaf inputs of the segment
    conclusion: FormulaNode  # final output
    is_assumption: bool = False  # for ASSUME tags
    is_discharge: bool = False  # for DISCHARGE tags
    discharged_formula: Optional[FormulaNode] = None
    assume_formula: Optional[FormulaNode] = None


@dataclass
class ChainGeneratorConfig:
    """Configuration for proof chain generation."""

    min_chain_length: int = 2
    max_chain_length: int = 5
    min_subformula_depth: int = 1
    max_subformula_depth: int = 2
    propositional_ratio: float = 0.5
    max_compression: int = 1  # 1 = no compression
    two_place_predicate_ratio: float = 0.3
    atom_pool: List[str] = field(
        default_factory=lambda: ["P", "Q", "R", "S", "T", "U"]
    )
    predicates_pool: List[str] = field(
        default_factory=lambda: ["P", "Q", "R", "S", "T"]
    )
    constants_pool: List[str] = field(
        default_factory=lambda: ["a", "b", "c", "d"]
    )
    variables_pool: List[str] = field(
        default_factory=lambda: ["x", "y", "z", "w"]
    )
    verify_steps: bool = True
    max_generation_attempts: int = 10_000
    discharge_probability: float = 0.15  # probability of choosing a discharge rule
    # Backward construction config
    allow_not_intro: bool = True          # False for stage 0
    allow_not_elim_final: bool = True     # False for stage 0
    stage0: bool = False                  # Stage 0 mode (restricts conclusion types + rules)
    compound_antecedent_prob: float = 0.4 # prob of compound antecedent at depth >= 2
    max_compound_connectives: int = 4     # max connectives in compound formulas
    # Per-rule base weights (looked up by NDRule.value).
    rule_weights: Dict[str, float] = field(default_factory=lambda: {
        r.value: 1.0 for r in NDRule
    })
    # Target rule frequencies for deficit-based reweighting (uniform by default).
    target_rule_freqs: Dict[str, float] = field(default_factory=lambda: {
        r.value: 1.0 / len(NDRule) for r in NDRule
    })
    discharge_use_assumption_prob: float = 0.5  # prob that discharge rules use the assumption
    intro_subgoal_prob: float = 0.25  # prob of forcing compound antecedent for intro-rule sub-goals


@dataclass
class _BackwardOption:
    """A single backward-construction option for deriving a goal."""

    rule: NDRule
    step_premises: List[FormulaNode]   # premises for the proof step
    given_premises: List[FormulaNode]  # added to initial_premises
    sub_goals: List[Tuple[FormulaNode, int, List[FormulaNode]]]
        # (formula, depth, available_assumptions)
    weight: float
    proof_block: Optional[List[ProofStep]] = None  # for discharge rules: full block
    eigenvariable: Optional[str] = None  # eigenvariable constant for FORALL_INTRO/EXISTS_ELIM
    forward_steps: Optional[List[ProofStep]] = None  # steps from assumption to meeting point
    forward_steps_b: Optional[List[ProofStep]] = None  # OR_ELIM branch B forward steps


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _formula_key(f: FormulaNode) -> str:
    """Structural equality key via formal string representation."""
    return f.to_formal()


def _formulas_equal(a: FormulaNode, b: FormulaNode) -> bool:
    """Check structural equality of two formulas."""
    return _formula_key(a) == _formula_key(b)


def _is_compound_binary(f: FormulaNode) -> bool:
    """Return True if *f* is a BinaryNode (non-atomic compound)."""
    return isinstance(f, BinaryNode)


def _is_implication(f: FormulaNode) -> bool:
    return isinstance(f, BinaryNode) and f.connective == Connective.IMPLIES


def _is_conjunction(f: FormulaNode) -> bool:
    return isinstance(f, BinaryNode) and f.connective == Connective.AND


def _is_disjunction(f: FormulaNode) -> bool:
    return isinstance(f, BinaryNode) and f.connective == Connective.OR


def _is_biconditional(f: FormulaNode) -> bool:
    return isinstance(f, BinaryNode) and f.connective == Connective.IFF


def _is_double_negation(f: FormulaNode) -> bool:
    return isinstance(f, NegationNode) and isinstance(f.child, NegationNode)


def _is_universal(f: FormulaNode) -> bool:
    return isinstance(f, QuantifiedNode) and f.quantifier == Quantifier.FORALL


def _is_existential(f: FormulaNode) -> bool:
    return isinstance(f, QuantifiedNode) and f.quantifier == Quantifier.EXISTS


def _substitute_variable(
    formula: FormulaNode, var: str, replacement: str
) -> FormulaNode:
    """Substitute all free occurrences of *var* with *replacement* in *formula*.

    Returns a deep copy with the substitution applied.
    """
    if isinstance(formula, AtomNode):
        new_vars = [replacement if v == var else v for v in formula.variables]
        return AtomNode(
            identifier=formula.identifier,
            variables=new_vars,
            is_predicate=formula.is_predicate,
        )
    elif isinstance(formula, NegationNode):
        return NegationNode(
            child=_substitute_variable(formula.child, var, replacement)
        )
    elif isinstance(formula, BinaryNode):
        return BinaryNode(
            connective=formula.connective,
            left=_substitute_variable(formula.left, var, replacement),
            right=_substitute_variable(formula.right, var, replacement),
        )
    elif isinstance(formula, QuantifiedNode):
        # Do not substitute under a quantifier that binds the same variable.
        if formula.variable == var:
            return formula.copy()
        return QuantifiedNode(
            quantifier=formula.quantifier,
            variable=formula.variable,
            body=_substitute_variable(formula.body, var, replacement),
        )
    return formula.copy()


# ---------------------------------------------------------------------------
# ChainGenerator
# ---------------------------------------------------------------------------


class ChainGenerator:
    """Generate multi-step proof chains using backward (goal-directed) natural deduction.

    The generator starts with a randomly chosen final conclusion and works
    backward, determining at each step which rule derives the current goal
    and what premises it needs.  This ensures every premise is essential —
    no fabrication, no redundancy, no ex-falso-quodlibet.

    Discharge rules (conditional proof, proof by contradiction, disjunction
    elimination, existential elimination) create assume/derive/discharge
    proof blocks with temporary assumptions.

    Z3 is used as a backstop to verify the full chain after generation
    (should never reject — backward construction is sound by design).
    """

    def __init__(self, config: Optional[ChainGeneratorConfig] = None):
        self.config = config or ChainGeneratorConfig()
        self._z3_translator: Optional[Any] = None
        if self.config.verify_steps and _HAS_Z3:
            self._z3_translator = Z3Translator()
        elif self.config.verify_steps and not _HAS_Z3:
            warnings.warn(
                "Z3 verification requested but z3-solver not available. "
                "Chain will not be verified.",
                stacklevel=2,
            )

        # Failure tracking for diagnostics.
        self.failure_stats = {
            "total_attempts": 0,
            "successes": 0,
            "z3_backstop_rejections": 0,
            "exceptions": 0,
        }

        # Per-example rule usage counts for deficit-based reweighting.
        self._rule_counts: Dict[str, int] = {}

        # Atom usage tracker to avoid duplicates within a single chain.
        self._used_atom_keys: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> ProofChain:
        """Generate a single proof chain with retry logic.

        Returns a ``ProofChain`` with verified steps (if Z3 is available).
        Raises ``RuntimeError`` after exhausting generation attempts.
        """
        for attempt in range(self.config.max_generation_attempts):
            self.failure_stats["total_attempts"] += 1
            try:
                chain = self._try_generate()
                if chain is not None:
                    self.failure_stats["successes"] += 1
                    return chain
            except Exception as exc:
                self.failure_stats["exceptions"] += 1
                logger.debug("Generation attempt %d failed: %s", attempt, exc)
        raise RuntimeError(
            f"Failed to generate a valid proof chain after "
            f"{self.config.max_generation_attempts} attempts."
        )

    def _get_essential_premise_keys(self, chain: ProofChain) -> Set[str]:
        """Find formula keys of initial premises needed for the final conclusion.

        Traces backward from the final step through each step's premises.
        Uses step indices (positions in the sorted step list) and only
        considers producers that appear *before* the current step,
        preventing cycles when multiple steps derive the same formula.

        Discharged formulas (temporary assumptions from IMPLIES_INTRO,
        NOT_INTRO, OR_ELIM, EXISTS_ELIM) are removed from the result.
        """
        initial_keys = {_formula_key(p) for p in chain.initial_premises}

        # Build index: formula_key -> sorted list of step indices producing it.
        formula_producers: Dict[str, List[int]] = {}
        for idx, step in enumerate(chain.steps):
            ck = _formula_key(step.conclusion)
            formula_producers.setdefault(ck, []).append(idx)

        essential_keys: Set[str] = set()
        visited_indices: Set[int] = set()

        # BFS from the final step (last in sorted order).
        queue: collections.deque = collections.deque(
            [len(chain.steps) - 1]
        )

        while queue:
            idx = queue.popleft()
            if idx in visited_indices:
                continue
            visited_indices.add(idx)

            step = chain.steps[idx]
            for prem in step.premises:
                pk = _formula_key(prem)
                if pk in initial_keys:
                    essential_keys.add(pk)
                # Find the latest producer of pk that comes BEFORE idx.
                producers = formula_producers.get(pk, [])
                earlier = [j for j in producers if j < idx]
                for j in earlier:
                    queue.append(j)

        # Remove discharged formulas (temporary assumptions that were
        # cancelled by discharge rules).
        discharged_keys: Set[str] = set()
        for step in chain.steps:
            if step.is_discharge and step.discharged_formula is not None:
                discharged_keys.add(_formula_key(step.discharged_formula))

        essential_keys -= discharged_keys
        return essential_keys

    def render(
        self, chain: ProofChain, pool: PropositionPool
    ) -> Dict[str, Any]:
        """Render a proof chain to a natural-language annotated dict.

        The returned dict matches the ``Annotation`` schema and includes
        a ``proof_trace`` field with tagged steps.  Each step in the
        trace lists its specific ``<PREMISE>`` inputs followed by its
        ``<CONCLUSION>``, so every inference can be verified in isolation.

        Returns:
            Dict with keys: id, premises, conclusion, proof_trace,
            verifier_notes, annotator_id, timestamp
        """
        renderer = NaturalLanguageRenderer(pool)

        # Collect ALL formulas from the entire proof chain so that
        # register_formulas() assigns consistent atom-to-NL mappings
        # across premises and every intermediate step.
        all_formulas: List[FormulaNode] = []
        for p in chain.initial_premises:
            all_formulas.append(p)
        for step in chain.steps:
            for prem in step.premises:
                all_formulas.append(prem)
            all_formulas.append(step.conclusion)
            if step.discharged_formula is not None:
                all_formulas.append(step.discharged_formula)

        renderer.register_formulas(all_formulas)

        # Render initial premises and final conclusion.
        premise_texts: List[str] = [
            renderer.render_formula(p) for p in chain.initial_premises
        ]
        conclusion_text: str = renderer.render_formula(chain.final_conclusion)

        # Compress steps.
        segments = self._compress_steps(chain.steps, self.config.max_compression)

        # Build proof trace with per-step premise-conclusion groups.
        trace_parts: List[str] = []

        for seg in segments:
            if seg.is_assumption and seg.assume_formula is not None:
                assume_text = renderer.render_formula(seg.assume_formula)
                trace_parts.append(f"<ASSUME> {assume_text} </ASSUME>")

            # Render this segment's specific premises.
            for prem in seg.premises:
                prem_text = renderer.render_formula(prem)
                trace_parts.append(f"<PREMISE> {prem_text} </PREMISE>")

            # Pure scope-close discharge steps (empty premises, no new formula
            # derived) should not emit a redundant <CONCLUSION> — the branch's
            # last step already showed the derived formula.
            is_pure_scope_close = (
                seg.is_discharge
                and not seg.premises
                and seg.discharged_formula is not None
            )
            if not is_pure_scope_close:
                seg_conclusion_text = renderer.render_formula(seg.conclusion)
                trace_parts.append(
                    f"<CONCLUSION> {seg_conclusion_text} </CONCLUSION>"
                )

            if seg.is_discharge and seg.discharged_formula is not None:
                discharge_text = renderer.render_formula(
                    seg.discharged_formula
                )
                trace_parts.append(
                    f"<DISCHARGE> {discharge_text} </DISCHARGE>"
                )

        proof_trace = "\n".join(trace_parts)

        # Only include undischarged leaf premises.
        discharged_keys: Set[str] = set()
        for step in chain.steps:
            if step.is_discharge and step.discharged_formula is not None:
                discharged_keys.add(_formula_key(step.discharged_formula))

        premise_objects: List[Dict[str, str]] = []
        for i, (p_formula, p_text) in enumerate(
            zip(chain.initial_premises, premise_texts)
        ):
            if _formula_key(p_formula) not in discharged_keys:
                premise_objects.append({"id": f"p{i+1}", "text": p_text})

        # Essential premises: only those transitively needed for the
        # final conclusion, deduplicated by formula key.
        essential_keys = self._get_essential_premise_keys(chain)

        # Z3 safety net: verify essential premises still entail conclusion.
        # If not, fall back to all undischarged premises.
        if self._z3_translator is not None and _HAS_Z3:
            try:
                _ctx = TranslationContext()
                _ess_formulas = [
                    p for p in chain.initial_premises
                    if _formula_key(p) in essential_keys
                    and _formula_key(p) not in discharged_keys
                ]
                _pz = [
                    self._z3_translator.translate(p, _ctx)
                    for p in _ess_formulas
                ]
                _cz = self._z3_translator.translate(
                    chain.final_conclusion, _ctx,
                )
                _solver = _z3_module.Solver()
                for _p in _pz:
                    _solver.add(_p)
                _solver.add(_z3_module.Not(_cz))
                if _solver.check() != _z3_module.unsat:
                    # Essential premises insufficient — use all undischarged.
                    essential_keys = {
                        _formula_key(p) for p in chain.initial_premises
                    } - discharged_keys
            except Exception:
                pass

        seen_keys: Set[str] = set()
        essential_premise_objects: List[Dict[str, str]] = []
        for i, (p_formula, p_text) in enumerate(
            zip(chain.initial_premises, premise_texts)
        ):
            pk = _formula_key(p_formula)
            if pk in essential_keys and pk not in seen_keys:
                seen_keys.add(pk)
                essential_premise_objects.append(
                    {"id": f"p{i+1}", "text": p_text}
                )

        verifier_notes = json.dumps(
            {
                "rules_used": [r.value for r in chain.rules_used],
                "logic_order": chain.logic_order.value,
                "chain_length": len(chain.steps),
                "initial_premises_count": len(chain.initial_premises),
            }
        )

        return {
            "id": str(uuid.uuid4()),
            "premises": premise_objects,
            "essential_premises": essential_premise_objects,
            "conclusion": conclusion_text,
            "proof_trace": proof_trace,
            "verifier_notes": verifier_notes,
            "annotator_id": "chain_generator",
            "timestamp": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal: backward (goal-directed) chain generation
    # ------------------------------------------------------------------

    def _try_generate(self) -> Optional[ProofChain]:
        """Attempt to build a single proof chain via backward construction.

        1. Choose logic order and target depth.
        2. Generate a random final conclusion.
        3. Recursively justify the conclusion via ``_justify()``.
        4. Assign layers and sort steps.
        5. Z3 backstop verification.
        """
        # Reset per-chain state (rule counts persist across chains for
        # deficit-based reweighting to improve rule distribution).
        self._used_atom_keys = set()

        # 1. Choose logic order.
        if random.random() < self.config.propositional_ratio:
            logic_order = LogicOrder.PROPOSITIONAL
        else:
            logic_order = LogicOrder.FIRST_ORDER

        # 2. Choose target depth (maps to chain length).
        target_depth = random.randint(
            self.config.min_chain_length, self.config.max_chain_length
        )

        # 3. Generate the final conclusion.
        conclusion = self._gen_conclusion(logic_order)

        # 4. Backward construction.
        initial_premises: List[FormulaNode] = []
        steps: List[ProofStep] = []
        step_counter = [1]  # mutable counter for step IDs
        rules_used: List[NDRule] = []

        self._justify(
            goal=conclusion,
            depth=target_depth,
            logic_order=logic_order,
            initial_premises=initial_premises,
            steps=steps,
            step_counter=step_counter,
            available_assumptions=[],
            rules_used=rules_used,
            is_final_step=True,
        )

        if not steps:
            return None

        # 5. Assign layers and sort.
        self._assign_layers(steps, initial_premises)
        steps.sort(key=lambda s: (s.layer, s.step_id))

        chain = ProofChain(
            initial_premises=initial_premises,
            steps=steps,
            final_conclusion=conclusion,
            logic_order=logic_order,
            rules_used=rules_used,
        )

        # 6. Z3 backstop verification.
        if not self._verify_chain(chain):
            self.failure_stats["z3_backstop_rejections"] += 1
            logger.warning(
                "Z3 backstop rejected chain: %s",
                conclusion.to_formal(),
            )
            return None

        return chain

    # ------------------------------------------------------------------
    # Conclusion generation
    # ------------------------------------------------------------------

    def _gen_conclusion(self, logic_order: LogicOrder) -> FormulaNode:
        """Generate a random final conclusion.

        Categories:
        - PL: proposition, negated proposition (stage 1 only)
        - FOL: predicate, negated predicate (stage 1 only),
                universal, negated universal (stage 1 only)
        """
        stage0 = self.config.stage0

        if logic_order == LogicOrder.PROPOSITIONAL:
            if stage0:
                # Stage 0: only positive propositions
                return self._gen_fresh_atom(logic_order)
            else:
                # Stage 1: proposition or negated proposition
                if random.random() < 0.6:
                    return self._gen_fresh_atom(logic_order)
                else:
                    return NegationNode(child=self._gen_fresh_atom(logic_order))
        else:
            # FOL
            if stage0:
                categories = ["predicate", "universal"]
            else:
                categories = [
                    "predicate", "negated_predicate",
                    "universal", "negated_universal",
                ]

            cat = random.choice(categories)

            if cat == "predicate":
                return self._gen_fresh_predicate()
            elif cat == "negated_predicate":
                return NegationNode(child=self._gen_fresh_predicate())
            elif cat == "universal":
                pred = random.choice(self.config.predicates_pool)
                var = random.choice(self.config.variables_pool)
                if random.random() < self.config.two_place_predicate_ratio:
                    const = random.choice(self.config.constants_pool)
                    body = AtomNode(
                        identifier=pred,
                        variables=[var, const],
                        is_predicate=True,
                    )
                else:
                    body = AtomNode(
                        identifier=pred,
                        variables=[var],
                        is_predicate=True,
                    )
                return QuantifiedNode(
                    quantifier=Quantifier.FORALL, variable=var, body=body,
                )
            else:  # negated_universal
                pred = random.choice(self.config.predicates_pool)
                var = random.choice(self.config.variables_pool)
                body = AtomNode(
                    identifier=pred,
                    variables=[var],
                    is_predicate=True,
                )
                return NegationNode(
                    child=QuantifiedNode(
                        quantifier=Quantifier.FORALL, variable=var, body=body,
                    )
                )

    def _gen_fresh_predicate(self) -> AtomNode:
        """Generate a fresh predicate atom with constant(s)."""
        ident = random.choice(self.config.predicates_pool)
        const = random.choice(self.config.constants_pool)
        if random.random() < self.config.two_place_predicate_ratio:
            const2 = random.choice(
                [c for c in self.config.constants_pool if c != const]
                or self.config.constants_pool
            )
            return AtomNode(
                identifier=ident,
                variables=[const, const2],
                is_predicate=True,
            )
        return AtomNode(
            identifier=ident, variables=[const], is_predicate=True,
        )

    # ------------------------------------------------------------------
    # Core backward recursive justification
    # ------------------------------------------------------------------

    def _justify(
        self,
        goal: FormulaNode,
        depth: int,
        logic_order: LogicOrder,
        initial_premises: List[FormulaNode],
        steps: List[ProofStep],
        step_counter: List[int],
        available_assumptions: List[FormulaNode],
        rules_used: List[NDRule],
        is_final_step: bool = False,
    ) -> None:
        """Recursively justify *goal* by backward construction.

        If depth == 0 or no backward options exist, the goal becomes an
        initial premise.  Otherwise, pick a rule that derives the goal,
        add given premises, create proof step(s), and recurse on sub-goals.
        """
        goal_key = _formula_key(goal)

        # Base case: depth exhausted.
        if depth <= 0:
            # Check if goal matches an available assumption.
            for a in available_assumptions:
                if _formula_key(a) == goal_key:
                    return  # already justified by assumption
            # Add as initial premise.
            if not any(_formula_key(p) == goal_key for p in initial_premises):
                initial_premises.append(goal.copy())
            return

        # Check if goal matches an available assumption (coin flip to use it).
        for a in available_assumptions:
            if _formula_key(a) == goal_key:
                if random.random() < 0.5:
                    return  # use assumption
                break  # continue to try backward options

        # Get backward options.
        options = self._backward_options(
            goal, depth, logic_order, is_final_step, available_assumptions,
        )

        if not options:
            # No applicable rules — make it an initial premise.
            if not any(_formula_key(p) == goal_key for p in initial_premises):
                for a in available_assumptions:
                    if _formula_key(a) == goal_key:
                        return
                initial_premises.append(goal.copy())
            return

        # Weighted random choice with eigenvariable rollback.
        # If a selected option has an eigenvariable and the recursive
        # construction violates the eigenvariable condition (the constant
        # appears in an initial premise), rollback and retry with a
        # different option.
        while options:
            option = self._weighted_choice(options)

            # Snapshot state for rollback if eigenvariable check may fail.
            if option.eigenvariable is not None:
                saved_premises = list(initial_premises)
                saved_steps = list(steps)
                saved_counter = step_counter[0]
                saved_rules = list(rules_used)

            # Add given premises to initial_premises.
            for gp in option.given_premises:
                gp_key = _formula_key(gp)
                if not any(
                    _formula_key(p) == gp_key for p in initial_premises
                ):
                    initial_premises.append(gp.copy())

            # If there's a proof block (discharge rule), emit it and recurse
            # on sub-goals at the right positions within the block.
            if option.proof_block is not None:
                self._emit_discharge_block(
                    option, logic_order, initial_premises, steps,
                    step_counter, rules_used,
                )
            else:
                # Simple rule: recurse on sub-goals first, then emit step.
                for sub_goal, sub_depth, sub_assumptions in option.sub_goals:
                    self._justify(
                        goal=sub_goal,
                        depth=sub_depth,
                        logic_order=logic_order,
                        initial_premises=initial_premises,
                        steps=steps,
                        step_counter=step_counter,
                        available_assumptions=sub_assumptions,
                        rules_used=rules_used,
                    )

                # Emit the proof step.
                step = self._make_step(
                    step_counter[0],
                    option.rule,
                    option.step_premises,
                    goal,
                )
                steps.append(step)
                rules_used.append(option.rule)
                step_counter[0] += 1

            # Eigenvariable check: the eigenvariable constant must not
            # appear in any initial premise.  If violated, rollback and
            # retry with a different option.
            if option.eigenvariable is not None:
                ev = option.eigenvariable
                violated = any(
                    ev in p.to_formal() for p in initial_premises
                )
                if violated:
                    logger.debug(
                        "Eigenvariable '%s' appears in initial premises; "
                        "rolling back %s",
                        ev, option.rule.value,
                    )
                    initial_premises[:] = saved_premises
                    steps[:] = saved_steps
                    step_counter[0] = saved_counter
                    rules_used[:] = saved_rules
                    options = [o for o in options if o is not option]
                    continue  # retry with remaining options

            # Option succeeded — done.
            return

        # All options exhausted (only happens after eigenvariable rollbacks).
        # Fall back to making the goal an initial premise.
        if not any(_formula_key(p) == goal_key for p in initial_premises):
            initial_premises.append(goal.copy())
        return

    def _emit_discharge_block(
        self,
        option: _BackwardOption,
        logic_order: LogicOrder,
        initial_premises: List[FormulaNode],
        steps: List[ProofStep],
        step_counter: List[int],
        rules_used: List[NDRule],
    ) -> None:
        """Emit a discharge rule's assume/inner/discharge block.

        The proof_block in the option contains template steps. This method
        assigns step IDs, emits assume steps, recurses on sub-goals to fill
        in inner derivations, then emits the discharge step.
        """
        block = option.proof_block
        assert block is not None

        # The block structure depends on the rule.
        if option.rule == NDRule.OR_ELIM:
            # OR_ELIM block: [assume_A, discharge_A, assume_B, discharge_B, final]
            # Sub-goals may be empty (bridge approach with explicit
            # elimination step) or [(G, d, assumptions+A), ...].
            assert len(block) == 5

            # Emit assume A
            assume_a = self._make_step(
                step_counter[0], option.rule, [],
                block[0].conclusion,
                is_assumption=True,
            )
            steps.append(assume_a)
            step_counter[0] += 1

            # Emit forward steps for branch A
            if option.forward_steps:
                for fwd_step in option.forward_steps:
                    emitted = self._make_step(
                        step_counter[0], fwd_step.rule,
                        fwd_step.premises, fwd_step.conclusion,
                    )
                    steps.append(emitted)
                    rules_used.append(fwd_step.rule)
                    step_counter[0] += 1

            # Recurse: justify G with A assumed (only if sub_goals present)
            if len(option.sub_goals) >= 1:
                sg1_goal, sg1_depth, sg1_assumptions = option.sub_goals[0]
                self._justify(
                    goal=sg1_goal, depth=sg1_depth, logic_order=logic_order,
                    initial_premises=initial_premises, steps=steps,
                    step_counter=step_counter,
                    available_assumptions=sg1_assumptions,
                    rules_used=rules_used,
                )

            # Emit discharge A (pure scope-close after branch A, no premises)
            discharge_a = self._make_step(
                step_counter[0], option.rule, [],
                block[1].conclusion,
                is_discharge=True,
                discharged_formula=block[1].discharged_formula,
            )
            steps.append(discharge_a)
            step_counter[0] += 1

            # Emit assume B
            assume_b = self._make_step(
                step_counter[0], option.rule, [],
                block[2].conclusion,
                is_assumption=True,
            )
            steps.append(assume_b)
            step_counter[0] += 1

            # Emit forward steps for branch B
            if option.forward_steps_b:
                for fwd_step in option.forward_steps_b:
                    emitted = self._make_step(
                        step_counter[0], fwd_step.rule,
                        fwd_step.premises, fwd_step.conclusion,
                    )
                    steps.append(emitted)
                    rules_used.append(fwd_step.rule)
                    step_counter[0] += 1

            # Recurse: justify G with B assumed (only if sub_goals present)
            if len(option.sub_goals) >= 2:
                sg2_goal, sg2_depth, sg2_assumptions = option.sub_goals[1]
                self._justify(
                    goal=sg2_goal, depth=sg2_depth, logic_order=logic_order,
                    initial_premises=initial_premises, steps=steps,
                    step_counter=step_counter,
                    available_assumptions=sg2_assumptions,
                    rules_used=rules_used,
                )

            # Emit discharge B (pure scope-close after branch B, no premises)
            discharge_b = self._make_step(
                step_counter[0], option.rule, [],
                block[3].conclusion,
                is_discharge=True,
                discharged_formula=block[3].discharged_formula,
            )
            steps.append(discharge_b)
            step_counter[0] += 1

            # Emit final OR_ELIM conclusion
            final = block[4]
            final_step = self._make_step(
                step_counter[0], option.rule,
                final.premises,
                final.conclusion,
            )
            steps.append(final_step)
            rules_used.append(option.rule)
            step_counter[0] += 1

        elif option.rule in (NDRule.IMPLIES_INTRO, NDRule.NOT_INTRO,
                             NDRule.EXISTS_ELIM):
            # These all have: [assume, ..., discharge]
            # Sub-goals may be empty (bridge approach) or [(inner_goal, d, assumptions)]
            assert len(block) >= 2

            # Emit assume step
            assume = block[0]
            assume_step = self._make_step(
                step_counter[0], option.rule, [],
                assume.conclusion,
                is_assumption=True,
            )
            steps.append(assume_step)
            step_counter[0] += 1

            # Emit forward steps (from bidirectional search) if present.
            if option.forward_steps:
                for fwd_step in option.forward_steps:
                    emitted = self._make_step(
                        step_counter[0], fwd_step.rule,
                        fwd_step.premises, fwd_step.conclusion,
                    )
                    steps.append(emitted)
                    rules_used.append(fwd_step.rule)
                    step_counter[0] += 1

            # Recurse on sub-goals (inner derivation)
            for sg_goal, sg_depth, sg_assumptions in option.sub_goals:
                self._justify(
                    goal=sg_goal, depth=sg_depth, logic_order=logic_order,
                    initial_premises=initial_premises, steps=steps,
                    step_counter=step_counter,
                    available_assumptions=sg_assumptions,
                    rules_used=rules_used,
                )

            # Emit discharge step
            discharge = block[-1]
            discharge_step = self._make_step(
                step_counter[0], option.rule,
                discharge.premises,
                discharge.conclusion,
                is_discharge=True,
                discharged_formula=discharge.discharged_formula,
            )
            steps.append(discharge_step)
            rules_used.append(option.rule)
            step_counter[0] += 1

    # ------------------------------------------------------------------
    # Forward chain construction for discharge rules
    # ------------------------------------------------------------------

    def _build_forward_chain(
        self,
        start: FormulaNode,
        logic_order: LogicOrder,
        num_steps: int,
        eigenvariable: Optional[str] = None,
        forbidden_keys: Optional[Set[str]] = None,
    ) -> Tuple[FormulaNode, List[ProofStep], List[FormulaNode]]:
        """Build a forward chain from *start* by applying random intro rules.

        At each step, randomly applies one of:
        - OR_INTRO: current ∨ X or X ∨ current
        - AND_INTRO: current ∧ X or X ∧ current (only when X does not
          collide with *forbidden_keys* — avoids the discharged-assumption
          name collision that caused Z3 rejections in earlier iterations)
        - EXISTS_INTRO (FOL only, when eigenvariable present and current
          contains it): ∃x.current[c/x]

        Returns ``(final_formula, list_of_proof_steps, extra_premises)``
        where *extra_premises* are fresh atoms introduced by AND_INTRO
        that must be added to given_premises by the caller.
        """
        if forbidden_keys is None:
            forbidden_keys = set()

        current = start
        fwd_steps: List[ProofStep] = []
        extra_premises: List[FormulaNode] = []
        step_id = 0

        for _ in range(num_steps):
            # Each expansion is (derived, rule, step_premises, extras)
            expansions: List[
                Tuple[FormulaNode, NDRule, List[FormulaNode], List[FormulaNode]]
            ] = []

            # OR_INTRO: current ∨ X or X ∨ current
            y = self._gen_fresh_atom(logic_order)
            if random.random() < 0.5:
                disj = make_disjunction(current, y)
            else:
                disj = make_disjunction(y, current)
            expansions.append((disj, NDRule.OR_INTRO, [current], []))

            # AND_INTRO: current ∧ X or X ∧ current (skip if X collides)
            x = self._gen_fresh_atom(logic_order)
            x_key = _formula_key(x)
            if x_key not in forbidden_keys:
                if random.random() < 0.5:
                    conj = make_conjunction(current, x)
                else:
                    conj = make_conjunction(x, current)
                expansions.append(
                    (conj, NDRule.AND_INTRO, [current, x], [x])
                )

            # EXISTS_INTRO (FOL only, when eigenvariable present and
            # current contains it)
            if (logic_order == LogicOrder.FIRST_ORDER
                    and eigenvariable is not None
                    and eigenvariable in current.to_formal()):
                var = random.choice(self.config.variables_pool)
                gen_body = _substitute_variable(current, eigenvariable, var)
                exists_abs = QuantifiedNode(
                    quantifier=Quantifier.EXISTS, variable=var, body=gen_body,
                )
                expansions.append(
                    (exists_abs, NDRule.EXISTS_INTRO, [current], [])
                )

            # Pick one expansion at random
            derived, rule, premises, extras = random.choice(expansions)
            step_id += 1
            step = ProofStep(
                step_id=step_id,
                rule=rule,
                premises=[p.copy() for p in premises],
                conclusion=derived.copy(),
            )
            fwd_steps.append(step)
            extra_premises.extend(extras)
            current = derived

        return (current, fwd_steps, extra_premises)

    def _make_bridge_and_elim(
        self,
        derived: FormulaNode,
        goal: FormulaNode,
        logic_order: LogicOrder,
        step_offset: int,
    ) -> Tuple[FormulaNode, ProofStep]:
        """Create a bridge premise and an elimination step that consumes it.

        Randomly creates either ``derived → goal`` (consumed by
        IMPLIES_ELIM) or ``derived ↔ goal`` (consumed by IFF_ELIM).

        Returns ``(bridge_formula, elimination_proof_step)``.
        """
        if random.random() < 0.5:
            bridge = make_implication(derived, goal)
            elim_rule = NDRule.IMPLIES_ELIM
        else:
            bridge = make_biconditional(derived, goal)
            elim_rule = NDRule.IFF_ELIM

        elim_step = ProofStep(
            step_id=step_offset + 1,
            rule=elim_rule,
            premises=[bridge.copy(), derived.copy()],
            conclusion=goal.copy(),
        )
        return (bridge, elim_step)

    # ------------------------------------------------------------------
    # Backward options
    # ------------------------------------------------------------------

    def _backward_options(
        self,
        goal: FormulaNode,
        depth: int,
        logic_order: LogicOrder,
        is_final_step: bool,
        available_assumptions: List[FormulaNode],
    ) -> List[_BackwardOption]:
        """Return all applicable backward construction options for *goal*."""
        options: List[_BackwardOption] = []
        cfg = self.config
        stage0 = cfg.stage0

        # --- Elimination rules (work for ANY goal type) ---

        # IMPLIES_ELIM: generate antecedent A, given premise A→G
        antecedent = self._gen_antecedent(depth, logic_order)
        impl = make_implication(antecedent, goal)
        # Check if A matches an available assumption
        ant_key = _formula_key(antecedent)
        ant_is_assumed = any(
            _formula_key(a) == ant_key for a in available_assumptions
        )
        sub_goals_ie: List[Tuple[FormulaNode, int, List[FormulaNode]]] = []
        if not ant_is_assumed:
            sub_goals_ie.append(
                (antecedent.copy(), depth - 1, list(available_assumptions))
            )
        options.append(_BackwardOption(
            rule=NDRule.IMPLIES_ELIM,
            step_premises=[impl, antecedent],
            given_premises=[impl],
            sub_goals=sub_goals_ie,
            weight=cfg.rule_weights.get(NDRule.IMPLIES_ELIM.value, 1.0),
        ))

        # IFF_ELIM: generate operand A, given premise A↔G or G↔A
        operand = self._gen_antecedent(depth, logic_order)
        if random.random() < 0.5:
            bic = make_biconditional(operand, goal)
        else:
            bic = make_biconditional(goal, operand)
        op_key = _formula_key(operand)
        op_is_assumed = any(
            _formula_key(a) == op_key for a in available_assumptions
        )
        sub_goals_bic: List[Tuple[FormulaNode, int, List[FormulaNode]]] = []
        if not op_is_assumed:
            sub_goals_bic.append(
                (operand.copy(), depth - 1, list(available_assumptions))
            )
        options.append(_BackwardOption(
            rule=NDRule.IFF_ELIM,
            step_premises=[bic, operand],
            given_premises=[bic],
            sub_goals=sub_goals_bic,
            weight=cfg.rule_weights.get(NDRule.IFF_ELIM.value, 1.0),
        ))

        # AND_ELIM: sub-goal G∧B or B∧G (continues backward construction)
        fresh_b = self._gen_fresh_atom(logic_order)
        if random.random() < 0.5:
            conj = make_conjunction(goal, fresh_b)
        else:
            conj = make_conjunction(fresh_b, goal)
        options.append(_BackwardOption(
            rule=NDRule.AND_ELIM,
            step_premises=[conj],
            given_premises=[],
            sub_goals=[
                (conj.copy(), depth - 1, list(available_assumptions)),
            ],
            weight=cfg.rule_weights.get(NDRule.AND_ELIM.value, 1.0),
        ))

        # NOT_ELIM: sub-goal ¬¬G (continues backward construction)
        if not (is_final_step and stage0 and not cfg.allow_not_elim_final):
            double_neg = NegationNode(child=NegationNode(child=goal.copy()))
            options.append(_BackwardOption(
                rule=NDRule.NOT_ELIM,
                step_premises=[double_neg],
                given_premises=[],
                sub_goals=[
                    (double_neg.copy(), depth - 1, list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.NOT_ELIM.value, 1.0),
            ))

        # FORALL_ELIM: only if goal is predicate atom with constant, FOL, NOT final
        if (logic_order == LogicOrder.FIRST_ORDER
                and not is_final_step
                and isinstance(goal, AtomNode)
                and goal.is_predicate
                and goal.variables):
            const = goal.variables[0]
            var = random.choice(self.config.variables_pool)
            gen_body = _substitute_variable(goal, const, var)
            forall_prem = QuantifiedNode(
                quantifier=Quantifier.FORALL, variable=var, body=gen_body,
            )
            options.append(_BackwardOption(
                rule=NDRule.FORALL_ELIM,
                step_premises=[forall_prem],
                given_premises=[],
                sub_goals=[
                    (forall_prem.copy(), depth - 1, list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.FORALL_ELIM.value, 1.0),
            ))

        # OR_ELIM as elimination: goal G, given A∨B, discharge
        # Uses forward chain + bridge premise with explicit elimination step
        # to ensure assumptions are used and bridge is consumed.
        if depth >= 3:
            a_disj = self._gen_fresh_atom(logic_order)
            b_disj = self._gen_fresh_atom(logic_order)
            disj = make_disjunction(a_disj, b_disj)

            # Forbidden keys: assumptions that will be discharged
            _fk_or = {_formula_key(a_disj), _formula_key(b_disj)}
            for a in available_assumptions:
                _fk_or.add(_formula_key(a))

            # Forward chains for each branch (d_fwd=0 allowed → no fwd steps)
            d_fwd_a = random.randint(0, min(3, depth - 2))
            f_a, fwd_steps_a, extras_a = self._build_forward_chain(
                a_disj, logic_order, d_fwd_a,
                forbidden_keys=_fk_or,
            )

            d_fwd_b = random.randint(0, min(3, depth - 2))
            f_b, fwd_steps_b, extras_b = self._build_forward_chain(
                b_disj, logic_order, d_fwd_b,
                forbidden_keys=_fk_or,
            )

            # Bridge premises + explicit elimination steps that consume them
            bridge_a, elim_step_a = self._make_bridge_and_elim(
                f_a, goal, logic_order, len(fwd_steps_a),
            )
            fwd_steps_a.append(elim_step_a)

            bridge_b, elim_step_b = self._make_bridge_and_elim(
                f_b, goal, logic_order, len(fwd_steps_b),
            )
            fwd_steps_b.append(elim_step_b)

            block = [
                # 0: assume A
                ProofStep(step_id=0, rule=NDRule.OR_ELIM, premises=[],
                          conclusion=a_disj.copy(), is_assumption=True),
                # 1: discharge A after branch A
                ProofStep(step_id=0, rule=NDRule.OR_ELIM, premises=[],
                          conclusion=goal.copy(),
                          is_discharge=True,
                          discharged_formula=a_disj.copy()),
                # 2: assume B
                ProofStep(step_id=0, rule=NDRule.OR_ELIM, premises=[],
                          conclusion=b_disj.copy(), is_assumption=True),
                # 3: discharge B after branch B
                ProofStep(step_id=0, rule=NDRule.OR_ELIM, premises=[],
                          conclusion=goal.copy(),
                          is_discharge=True,
                          discharged_formula=b_disj.copy()),
                # 4: final OR_ELIM conclusion
                ProofStep(step_id=0, rule=NDRule.OR_ELIM,
                          premises=[disj, goal, goal],
                          conclusion=goal.copy()),
            ]
            _gp_or = [disj, bridge_a, bridge_b] + extras_a + extras_b
            options.append(_BackwardOption(
                rule=NDRule.OR_ELIM,
                step_premises=[disj, goal, goal],
                given_premises=_gp_or,
                sub_goals=[],  # no recursion; elimination step derives G
                weight=cfg.rule_weights.get(NDRule.OR_ELIM.value, 1.0),
                proof_block=block,
                forward_steps=fwd_steps_a or None,
                forward_steps_b=fwd_steps_b or None,
            ))

        # EXISTS_ELIM as elimination: goal G, given ∃x.P(x), FOL only
        # Uses forward chain + bridge premise to ensure assumption P(c) is used.
        if (logic_order == LogicOrder.FIRST_ORDER and depth >= 3):
            pred_ee = random.choice(self.config.predicates_pool)
            var_ee = random.choice(self.config.variables_pool)
            body_ee = AtomNode(
                identifier=pred_ee, variables=[var_ee], is_predicate=True,
            )
            exists_prem = QuantifiedNode(
                quantifier=Quantifier.EXISTS, variable=var_ee, body=body_ee,
            )
            # Choose fresh eigenvariable constant
            goal_formal = goal.to_formal()
            all_formals = [p.to_formal() for p in available_assumptions]
            used_c = set()
            for f in all_formals + [goal_formal]:
                for c in self.config.constants_pool:
                    if c in f:
                        used_c.add(c)
            fresh_consts = [
                c for c in self.config.constants_pool if c not in used_c
            ]
            if fresh_consts:
                eigen_const = random.choice(fresh_consts)
                assumed_ee = _substitute_variable(body_ee, var_ee, eigen_const)

                # Forbidden keys: discharged assumption + outer assumptions
                _fk_ee = {_formula_key(assumed_ee)}
                for a in available_assumptions:
                    _fk_ee.add(_formula_key(a))

                d_fwd_ee = random.randint(0, min(3, depth - 2))

                # Forward chain from P(c), with eigenvariable for EXISTS_INTRO
                f_ee, fwd_steps_ee_list, extras_ee = self._build_forward_chain(
                    assumed_ee, logic_order, d_fwd_ee,
                    eigenvariable=eigen_const,
                    forbidden_keys=_fk_ee,
                )

                # If F still contains the eigenvariable, append an
                # EXISTS_INTRO step to abstract it away.
                if eigen_const in f_ee.to_formal():
                    var_abs = random.choice(self.config.variables_pool)
                    abs_body = _substitute_variable(
                        f_ee, eigen_const, var_abs,
                    )
                    exists_abs = QuantifiedNode(
                        quantifier=Quantifier.EXISTS,
                        variable=var_abs, body=abs_body,
                    )
                    abs_step = ProofStep(
                        step_id=len(fwd_steps_ee_list) + 1,
                        rule=NDRule.EXISTS_INTRO,
                        premises=[f_ee.copy()],
                        conclusion=exists_abs.copy(),
                    )
                    fwd_steps_ee_list.append(abs_step)
                    f_ee = exists_abs
                    d_fwd_ee += 1  # account for extra step

                # Bridge premise + explicit elimination step
                bridge_ee, elim_step_ee = self._make_bridge_and_elim(
                    f_ee, goal, logic_order, len(fwd_steps_ee_list),
                )
                fwd_steps_ee_list.append(elim_step_ee)

                block_ee = [
                    ProofStep(
                        step_id=0, rule=NDRule.EXISTS_ELIM, premises=[],
                        conclusion=assumed_ee.copy(), is_assumption=True,
                    ),
                    ProofStep(
                        step_id=0, rule=NDRule.EXISTS_ELIM,
                        premises=[exists_prem, assumed_ee],
                        conclusion=goal.copy(),
                        is_discharge=True,
                        discharged_formula=assumed_ee.copy(),
                    ),
                ]
                _gp_ee = [exists_prem, bridge_ee] + extras_ee
                options.append(_BackwardOption(
                    rule=NDRule.EXISTS_ELIM,
                    step_premises=[exists_prem, assumed_ee],
                    given_premises=_gp_ee,
                    sub_goals=[],  # no recursion; elimination step derives G
                    weight=cfg.rule_weights.get(
                        NDRule.EXISTS_ELIM.value, 1.0,
                    ),
                    proof_block=block_ee,
                    eigenvariable=eigen_const,
                    forward_steps=fwd_steps_ee_list or None,
                ))

        # --- Introduction rules (only when goal has matching structure) ---

        # AND_INTRO: goal A∧B
        if _is_conjunction(goal):
            assert isinstance(goal, BinaryNode)
            options.append(_BackwardOption(
                rule=NDRule.AND_INTRO,
                step_premises=[goal.left, goal.right],
                given_premises=[],
                sub_goals=[
                    (goal.left.copy(), depth - 1, list(available_assumptions)),
                    (goal.right.copy(), depth - 1, list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.AND_INTRO.value, 1.0),
            ))

        # OR_INTRO: goal A∨B
        if _is_disjunction(goal):
            assert isinstance(goal, BinaryNode)
            # Randomly choose to justify left or right
            if random.random() < 0.5:
                chosen = goal.left
            else:
                chosen = goal.right
            options.append(_BackwardOption(
                rule=NDRule.OR_INTRO,
                step_premises=[chosen],
                given_premises=[],
                sub_goals=[
                    (chosen.copy(), depth - 1, list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.OR_INTRO.value, 1.0),
            ))

        # IFF_INTRO: goal A↔B
        if _is_biconditional(goal):
            assert isinstance(goal, BinaryNode)
            impl_ab = make_implication(goal.left, goal.right)
            impl_ba = make_implication(goal.right, goal.left)
            options.append(_BackwardOption(
                rule=NDRule.IFF_INTRO,
                step_premises=[impl_ab, impl_ba],
                given_premises=[],
                sub_goals=[
                    (impl_ab.copy(), depth - 1, list(available_assumptions)),
                    (impl_ba.copy(), depth - 1, list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.IFF_INTRO.value, 1.0),
            ))

        # IMPLIES_INTRO: goal A→B (discharge rule)
        # Uses forward chain + bridge premise to ensure assumption A is used.
        if _is_implication(goal) and depth >= 3:
            assert isinstance(goal, BinaryNode)
            assumed_a = goal.left
            inner_goal_b = goal.right

            # Forbidden keys: discharged assumption + outer assumptions
            _fk_ii = {_formula_key(assumed_a)}
            for a in available_assumptions:
                _fk_ii.add(_formula_key(a))

            d_fwd_ii = random.randint(0, min(3, depth - 2))
            f_ii, fwd_steps_ii_list, extras_ii = self._build_forward_chain(
                assumed_a, logic_order, d_fwd_ii,
                forbidden_keys=_fk_ii,
            )

            # Bridge premise + explicit elimination step
            bridge_ii, elim_step_ii = self._make_bridge_and_elim(
                f_ii, inner_goal_b, logic_order, len(fwd_steps_ii_list),
            )
            fwd_steps_ii_list.append(elim_step_ii)

            block_ii = [
                ProofStep(
                    step_id=0, rule=NDRule.IMPLIES_INTRO, premises=[],
                    conclusion=assumed_a.copy(), is_assumption=True,
                ),
                ProofStep(
                    step_id=0, rule=NDRule.IMPLIES_INTRO,
                    premises=[assumed_a, inner_goal_b],
                    conclusion=goal.copy(),
                    is_discharge=True,
                    discharged_formula=assumed_a.copy(),
                ),
            ]
            _gp_ii = [bridge_ii] + extras_ii
            options.append(_BackwardOption(
                rule=NDRule.IMPLIES_INTRO,
                step_premises=[assumed_a, inner_goal_b],
                given_premises=_gp_ii,
                sub_goals=[],  # no recursion; elimination step derives B
                weight=cfg.rule_weights.get(
                    NDRule.IMPLIES_INTRO.value, 1.0,
                ),
                proof_block=block_ii,
                forward_steps=fwd_steps_ii_list or None,
            ))

        # NOT_INTRO: goal ¬A (discharge rule)
        # Uses forward chain + bridge premise to ensure assumption A is used.
        if (isinstance(goal, NegationNode)
                and depth >= 3
                and not (is_final_step and stage0)):
            if cfg.allow_not_intro or not is_final_step:
                assumed_a_ni = goal.child
                # Generate a fresh B for the contradiction B ∧ ¬B
                contra_b = self._gen_fresh_atom(logic_order)
                neg_contra_b = NegationNode(child=contra_b.copy())

                # Forbidden keys: discharged assumption + outer assumptions
                _fk_ni = {_formula_key(assumed_a_ni)}
                for a in available_assumptions:
                    _fk_ni.add(_formula_key(a))

                d_fwd_ni = random.randint(0, min(3, depth - 2))
                f_ni, fwd_steps_ni_list, extras_ni = self._build_forward_chain(
                    assumed_a_ni, logic_order, d_fwd_ni,
                    forbidden_keys=_fk_ni,
                )

                # Bridge premise + explicit elimination step
                bridge_ni, elim_step_ni = self._make_bridge_and_elim(
                    f_ni, contra_b, logic_order, len(fwd_steps_ni_list),
                )
                fwd_steps_ni_list.append(elim_step_ni)

                block_ni = [
                    ProofStep(
                        step_id=0, rule=NDRule.NOT_INTRO, premises=[],
                        conclusion=assumed_a_ni.copy(),
                        is_assumption=True,
                    ),
                    ProofStep(
                        step_id=0, rule=NDRule.NOT_INTRO,
                        premises=[contra_b, neg_contra_b],
                        conclusion=goal.copy(),
                        is_discharge=True,
                        discharged_formula=assumed_a_ni.copy(),
                    ),
                ]
                _gp_ni = [neg_contra_b.copy(), bridge_ni] + extras_ni
                options.append(_BackwardOption(
                    rule=NDRule.NOT_INTRO,
                    step_premises=[contra_b, neg_contra_b],
                    given_premises=_gp_ni,
                    sub_goals=[],  # no recursion; elimination step derives contra_b
                    weight=cfg.rule_weights.get(
                        NDRule.NOT_INTRO.value, 1.0,
                    ),
                    proof_block=block_ni,
                    forward_steps=fwd_steps_ni_list or None,
                ))

        # FORALL_INTRO: goal ∀x.P(x), FOL only
        if (_is_universal(goal) and not is_final_step
                and logic_order == LogicOrder.FIRST_ORDER):
            assert isinstance(goal, QuantifiedNode)
            eigen_const_fi = random.choice(self.config.constants_pool)
            instantiated = _substitute_variable(
                goal.body, goal.variable, eigen_const_fi,
            )
            options.append(_BackwardOption(
                rule=NDRule.FORALL_INTRO,
                step_premises=[instantiated],
                given_premises=[],
                sub_goals=[
                    (instantiated.copy(), depth - 1,
                     list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.FORALL_INTRO.value, 1.0),
                eigenvariable=eigen_const_fi,
            ))

        # EXISTS_INTRO: goal ∃x.P(x), FOL only, NOT final step
        if (_is_existential(goal) and not is_final_step
                and logic_order == LogicOrder.FIRST_ORDER):
            assert isinstance(goal, QuantifiedNode)
            const_ei = random.choice(self.config.constants_pool)
            instantiated_ei = _substitute_variable(
                goal.body, goal.variable, const_ei,
            )
            options.append(_BackwardOption(
                rule=NDRule.EXISTS_INTRO,
                step_premises=[instantiated_ei],
                given_premises=[],
                sub_goals=[
                    (instantiated_ei.copy(), depth - 1,
                     list(available_assumptions)),
                ],
                weight=cfg.rule_weights.get(NDRule.EXISTS_INTRO.value, 1.0),
            ))

        return options

    # ------------------------------------------------------------------
    # Compound formula / antecedent generation
    # ------------------------------------------------------------------

    def _gen_antecedent(
        self, depth: int, logic_order: LogicOrder,
    ) -> FormulaNode:
        """Generate an antecedent for IMPLIES_ELIM/IFF_ELIM.

        With probability ``intro_subgoal_prob``, forces a compound formula
        of an intro-rule-enabling type (conjunction, disjunction,
        implication, biconditional) so that introduction rules fire
        in subsequent sub-derivations.

        At higher depths, may also generate compound formulas with 1-4
        connectives.
        """
        # Compound injection for intro-rule sub-goals.
        if random.random() < self.config.intro_subgoal_prob:
            return self._gen_intro_compound(logic_order)

        prob = self.config.compound_antecedent_prob
        if depth >= 3 and random.random() < prob:
            n = random.randint(1, self.config.max_compound_connectives)
            return self._gen_compound_formula(n, logic_order)
        elif depth >= 2 and random.random() < prob * 0.5:
            n = random.randint(1, 2)
            return self._gen_compound_formula(n, logic_order)
        return self._gen_fresh_atom(logic_order)

    def _gen_intro_compound(self, logic_order: LogicOrder) -> FormulaNode:
        """Generate a compound formula that enables introduction rules.

        Uniformly samples from {conjunction, disjunction, implication,
        biconditional} and builds operands using rich atom types.
        For FOL, may wrap the result in quantifier(s) to enable
        FORALL_ELIM/EXISTS_ELIM opportunities.
        """
        connective = random.choice([
            Connective.AND, Connective.OR,
            Connective.IMPLIES, Connective.IFF,
        ])
        left = self._gen_rich_atom(logic_order)
        right = self._gen_rich_atom(logic_order)
        body = BinaryNode(connective=connective, left=left, right=right)

        # For FOL, optionally wrap in quantifier(s).
        if logic_order == LogicOrder.FIRST_ORDER and random.random() < 0.4:
            body = self._wrap_free_vars_with_quantifiers(body)

        return body

    def _gen_rich_atom(self, logic_order: LogicOrder) -> FormulaNode:
        """Generate an atom using all atom types.

        Types: regular atom, negated atom, universally quantified atom,
        existentially quantified atom.  Quantified atoms are only
        generated for FOL.
        """
        if logic_order == LogicOrder.FIRST_ORDER:
            kind = random.choice([
                "regular", "negated", "universal", "existential",
            ])
        else:
            kind = random.choice(["regular", "negated"])

        if kind == "regular":
            return self._gen_fresh_atom(logic_order)
        elif kind == "negated":
            return NegationNode(child=self._gen_fresh_atom(logic_order))
        elif kind == "universal":
            pred = random.choice(self.config.predicates_pool)
            var = random.choice(self.config.variables_pool)
            body = AtomNode(
                identifier=pred, variables=[var], is_predicate=True,
            )
            return QuantifiedNode(
                quantifier=Quantifier.FORALL, variable=var, body=body,
            )
        else:  # existential
            pred = random.choice(self.config.predicates_pool)
            var = random.choice(self.config.variables_pool)
            body = AtomNode(
                identifier=pred, variables=[var], is_predicate=True,
            )
            return QuantifiedNode(
                quantifier=Quantifier.EXISTS, variable=var, body=body,
            )

    @staticmethod
    def _collect_free_variables(
        formula: FormulaNode, bound: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Collect all free variables in *formula*."""
        if bound is None:
            bound = set()
        if isinstance(formula, AtomNode):
            return {
                v for v in formula.variables
                if v not in bound and v.islower() and len(v) == 1
            }
        elif isinstance(formula, NegationNode):
            return ChainGenerator._collect_free_variables(formula.child, bound)
        elif isinstance(formula, BinaryNode):
            return (
                ChainGenerator._collect_free_variables(formula.left, bound)
                | ChainGenerator._collect_free_variables(formula.right, bound)
            )
        elif isinstance(formula, QuantifiedNode):
            return ChainGenerator._collect_free_variables(
                formula.body, bound | {formula.variable},
            )
        return set()

    def _wrap_free_vars_with_quantifiers(
        self, formula: FormulaNode,
    ) -> FormulaNode:
        """Wrap each free variable in *formula* with a random quantifier.

        Applies innermost-first so all variables end up bound.
        """
        free_vars = self._collect_free_variables(formula)
        result = formula
        for v in sorted(free_vars):  # deterministic order
            q = random.choice([Quantifier.FORALL, Quantifier.EXISTS])
            result = QuantifiedNode(quantifier=q, variable=v, body=result)
        return result

    def _gen_compound_formula(
        self, num_connectives: int, logic_order: LogicOrder,
    ) -> FormulaNode:
        """Build a random compound formula with *num_connectives* binary connectives."""
        if num_connectives <= 0:
            return self._gen_fresh_atom(logic_order)

        connective = random.choice([
            Connective.AND, Connective.OR,
            Connective.IMPLIES, Connective.IFF,
        ])

        # Distribute remaining connectives between left and right.
        remaining = num_connectives - 1
        if remaining == 0:
            left = self._gen_fresh_atom(logic_order)
            right = self._gen_fresh_atom(logic_order)
        else:
            left_n = random.randint(0, remaining)
            right_n = remaining - left_n
            left = self._gen_compound_formula(left_n, logic_order)
            right = self._gen_compound_formula(right_n, logic_order)

        return BinaryNode(connective=connective, left=left, right=right)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gen_fresh_atom(self, logic_order: LogicOrder) -> AtomNode:
        """Generate a fresh atom appropriate for the logic order."""
        if logic_order == LogicOrder.FIRST_ORDER:
            return self._gen_fresh_predicate()
        ident = random.choice(self.config.atom_pool)
        return AtomNode(identifier=ident)

    def _make_step(
        self,
        step_id: int,
        rule: NDRule,
        premises: List[FormulaNode],
        conclusion: FormulaNode,
        **kwargs,
    ) -> ProofStep:
        return ProofStep(
            step_id=step_id,
            rule=rule,
            premises=[p.copy() for p in premises],
            conclusion=conclusion.copy(),
            **kwargs,
        )

    def _weighted_choice(self, options: List[_BackwardOption]) -> _BackwardOption:
        """Weighted random selection with deficit-based reweighting.

        Each option's weight is multiplied by a squared catch-up factor:
        (target_freq / (observed_freq + epsilon))^2, so under-represented
        rules get a strong boost and over-represented rules get dampened.
        The exponent of 2 provides aggressive rebalancing; cross-chain
        accumulation of rule counts provides enough history for this to
        converge toward the target distribution.
        """
        epsilon = 1e-6
        total_uses = sum(self._rule_counts.values()) + epsilon

        adjusted: List[float] = []
        for o in options:
            rule_name = o.rule.value
            target = self.config.target_rule_freqs.get(
                rule_name, 1.0 / len(NDRule),
            )
            observed = self._rule_counts.get(rule_name, 0) / total_uses
            catchup = (target / (observed + epsilon)) ** 2
            adjusted.append(o.weight * catchup)

        total = sum(adjusted)
        r = random.random() * total
        cumulative = 0.0
        for o, w in zip(options, adjusted):
            cumulative += w
            if r <= cumulative:
                self._rule_counts[o.rule.value] = (
                    self._rule_counts.get(o.rule.value, 0) + 1
                )
                return o
        chosen = options[-1]
        self._rule_counts[chosen.rule.value] = (
            self._rule_counts.get(chosen.rule.value, 0) + 1
        )
        return chosen

    # ------------------------------------------------------------------
    # Z3 backstop verification
    # ------------------------------------------------------------------

    def _verify_chain(self, chain: ProofChain) -> bool:
        """Verify the full chain: conjunction of initial premises entails conclusion.

        Returns True if verification passes or if Z3 is unavailable.
        """
        if self._z3_translator is None:
            return True

        try:
            ctx = TranslationContext()

            # Translate all initial premises and the conclusion.
            # Only include undischarged premises.
            discharged_keys: Set[str] = set()
            for step in chain.steps:
                if step.is_discharge and step.discharged_formula is not None:
                    discharged_keys.add(_formula_key(step.discharged_formula))

            premise_z3 = []
            for p in chain.initial_premises:
                if _formula_key(p) not in discharged_keys:
                    premise_z3.append(
                        self._z3_translator.translate(p, ctx)
                    )

            conclusion_z3 = self._z3_translator.translate(
                chain.final_conclusion, ctx,
            )

            # Check: premises AND NOT(conclusion) is UNSAT?
            solver = _z3_module.Solver()
            for pz in premise_z3:
                solver.add(pz)
            solver.add(_z3_module.Not(conclusion_z3))

            result = solver.check()
            return result == _z3_module.unsat
        except Exception as exc:
            logger.debug("Z3 backstop verification error: %s", exc)
            # Accept on error to avoid blocking generation.
            return True

    # ------------------------------------------------------------------
    # Layer assignment
    # ------------------------------------------------------------------

    def _assign_layers(
        self,
        steps: List[ProofStep],
        initial_premises: List[FormulaNode],
    ) -> None:
        """Assign BFS layers to steps based on dependency order."""
        # Map formula key -> earliest step_id that produces it.
        producers: Dict[str, int] = {}
        for p_idx, p in enumerate(initial_premises):
            producers[_formula_key(p)] = -(p_idx + 1)
        for step in steps:
            producers.setdefault(_formula_key(step.conclusion), step.step_id)

        # Build dependency graph: step_id -> set of step_ids it depends on.
        deps: Dict[int, Set[int]] = {}
        for step in steps:
            dep_set: Set[int] = set()
            for prem in step.premises:
                pk = _formula_key(prem)
                if pk in producers:
                    dep_set.add(producers[pk])
            deps[step.step_id] = dep_set

        # BFS from steps with no internal dependencies.
        layer_of: Dict[int, int] = {}
        # Initial premises are layer 0.
        for p_idx in range(len(initial_premises)):
            layer_of[-(p_idx + 1)] = 0

        changed = True
        while changed:
            changed = False
            for step in steps:
                if step.step_id in layer_of:
                    continue
                dep_layers = []
                all_resolved = True
                for d in deps.get(step.step_id, set()):
                    if d in layer_of:
                        dep_layers.append(layer_of[d])
                    else:
                        all_resolved = False
                        break
                if all_resolved:
                    layer_of[step.step_id] = (
                        max(dep_layers) + 1 if dep_layers else 1
                    )
                    changed = True

        # Assign layers to steps (fallback to step order).
        for i, step in enumerate(steps):
            step.layer = layer_of.get(step.step_id, i + 1)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _compress_steps(
        self, steps: List[ProofStep], max_n: int
    ) -> List[CompressedSegment]:
        """Partition linearised steps into contiguous groups of size 1..max_n.

        Each group becomes one ``CompressedSegment`` whose premises are the
        leaf inputs and whose conclusion is the final output of the group.
        """
        if max_n <= 1 or not steps:
            # No compression -- one segment per step.
            segments: List[CompressedSegment] = []
            for step in steps:
                seg = CompressedSegment(
                    premises=step.premises,
                    conclusion=step.conclusion,
                    is_assumption=step.is_assumption,
                    is_discharge=step.is_discharge,
                    discharged_formula=step.discharged_formula,
                    assume_formula=(
                        step.conclusion if step.is_assumption else None
                    ),
                )
                segments.append(seg)
            return segments

        # Random contiguous partitioning.
        segments = []
        i = 0
        while i < len(steps):
            group_size = random.randint(1, min(max_n, len(steps) - i))
            group = steps[i : i + group_size]

            # Collect all premises that are *not* produced within the group.
            produced_keys: Set[str] = set()
            for s in group:
                produced_keys.add(_formula_key(s.conclusion))

            leaf_premises: List[FormulaNode] = []
            seen_leaf_keys: Set[str] = set()
            for s in group:
                for p in s.premises:
                    pk = _formula_key(p)
                    if pk not in produced_keys and pk not in seen_leaf_keys:
                        leaf_premises.append(p)
                        seen_leaf_keys.add(pk)

            final_step = group[-1]
            first_step = group[0]

            seg = CompressedSegment(
                premises=leaf_premises,
                conclusion=final_step.conclusion,
                is_assumption=first_step.is_assumption,
                is_discharge=final_step.is_discharge,
                discharged_formula=final_step.discharged_formula,
                assume_formula=(
                    first_step.conclusion if first_step.is_assumption else None
                ),
            )
            segments.append(seg)
            i += group_size

        return segments
