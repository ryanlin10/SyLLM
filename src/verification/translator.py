"""Translate FormulaNode ASTs to Z3 expressions for symbolic verification."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..data.syntax_tree import (
    FormulaNode, AtomNode, NegationNode, BinaryNode, QuantifiedNode,
    Connective, Quantifier,
)

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@dataclass
class TranslationContext:
    """Shared context that maps logical names to Z3 objects.

    A single context should be passed across all premises **and** the
    conclusion so that the same atom text always maps to the same Z3
    variable / predicate / constant.
    """

    bool_vars: Dict[str, Any] = field(default_factory=dict)    # name -> z3.Bool
    predicates: Dict[str, Any] = field(default_factory=dict)   # name -> z3.Function
    constants: Dict[str, Any] = field(default_factory=dict)    # name -> z3.Const
    bound_vars: Dict[str, Any] = field(default_factory=dict)   # name -> z3.Const (quantifier scope)
    is_fol: bool = False


class Z3Translator:
    """Translate a :class:`FormulaNode` AST into a Z3 expression."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, formula: FormulaNode, ctx: TranslationContext) -> Any:
        """Translate *formula* to a Z3 expression.

        Parameters
        ----------
        formula:
            The root of a FormulaNode AST produced by the syntax-tree
            module.
        ctx:
            A :class:`TranslationContext` that accumulates variable /
            predicate / constant mappings.  Pass the **same** context for
            every premise and the conclusion so that identical atom text
            maps to the same Z3 object.

        Returns
        -------
        z3.BoolRef
            The translated Z3 boolean expression.

        Raises
        ------
        ImportError
            If the ``z3-solver`` package is not installed.
        TypeError
            If *formula* is an unrecognised :class:`FormulaNode` subtype.
        """
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver not installed")

        if isinstance(formula, AtomNode):
            return self._translate_atom(formula, ctx)
        elif isinstance(formula, NegationNode):
            return self._translate_negation(formula, ctx)
        elif isinstance(formula, BinaryNode):
            return self._translate_binary(formula, ctx)
        elif isinstance(formula, QuantifiedNode):
            return self._translate_quantified(formula, ctx)
        else:
            raise TypeError(f"Unsupported FormulaNode type: {type(formula).__name__}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _translate_atom(self, atom: AtomNode, ctx: TranslationContext) -> Any:
        """Translate an :class:`AtomNode`.

        Propositional atoms (``is_predicate=False``) become ``z3.Bool``
        variables.  FOL predicates (``is_predicate=True``) become
        applications of a ``z3.Function`` to ``IntSort`` arguments.
        Supports N-place predicates (e.g., 1-place ``P(a)`` and
        2-place ``R(a, b)``).
        """
        key = atom.identifier

        if not atom.is_predicate:
            # ---- Propositional atom ----
            if key not in ctx.bool_vars:
                ctx.bool_vars[key] = z3.Bool(key)
            return ctx.bool_vars[key]

        # ---- FOL predicate ----
        ctx.is_fol = True
        arity = len(atom.variables)

        # Key predicates by (name, arity) to support different arities.
        pred_key = f"{key}/{arity}"
        if pred_key not in ctx.predicates:
            ctx.predicates[pred_key] = z3.Function(
                key, *([z3.IntSort()] * arity), z3.BoolSort()
            )
        pred_func = ctx.predicates[pred_key]

        # Resolve each entity / variable argument.
        args = []
        for var_name in atom.variables:
            if var_name in ctx.bound_vars:
                args.append(ctx.bound_vars[var_name])
            else:
                if var_name not in ctx.constants:
                    ctx.constants[var_name] = z3.Const(var_name, z3.IntSort())
                args.append(ctx.constants[var_name])

        return pred_func(*args)

    def _translate_negation(self, node: NegationNode, ctx: TranslationContext) -> Any:
        """Translate ``NOT child``."""
        child_z3 = self.translate(node.child, ctx)
        return z3.Not(child_z3)

    def _translate_binary(self, node: BinaryNode, ctx: TranslationContext) -> Any:
        """Translate a binary connective (AND, OR, IMPLIES, IFF)."""
        left_z3 = self.translate(node.left, ctx)
        right_z3 = self.translate(node.right, ctx)

        if node.connective == Connective.AND:
            return z3.And(left_z3, right_z3)
        elif node.connective == Connective.OR:
            return z3.Or(left_z3, right_z3)
        elif node.connective == Connective.IMPLIES:
            return z3.Implies(left_z3, right_z3)
        elif node.connective == Connective.IFF:
            return left_z3 == right_z3
        else:
            raise ValueError(f"Unknown connective: {node.connective}")

    def _translate_quantified(self, node: QuantifiedNode, ctx: TranslationContext) -> Any:
        """Translate ``FORALL x. body`` or ``EXISTS x. body``.

        The bound variable is added to *ctx.bound_vars* for the duration
        of the body translation and removed afterwards to restore the
        enclosing scope.
        """
        var = z3.Const(node.variable, z3.IntSort())

        # Save any previous binding so nested quantifiers over the same
        # variable name are handled correctly.
        prev = ctx.bound_vars.get(node.variable)
        ctx.bound_vars[node.variable] = var

        body_z3 = self.translate(node.body, ctx)

        # Restore previous scope.
        if prev is not None:
            ctx.bound_vars[node.variable] = prev
        else:
            del ctx.bound_vars[node.variable]

        if node.quantifier == Quantifier.FORALL:
            return z3.ForAll([var], body_z3)
        elif node.quantifier == Quantifier.EXISTS:
            return z3.Exists([var], body_z3)
        else:
            raise ValueError(f"Unknown quantifier: {node.quantifier}")
