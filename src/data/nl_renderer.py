"""
Natural language renderer for logic formula trees.

Converts syntax trees into natural language sentences using atomic propositions
from a proposition pool.

Only uses approved logical terms:
- 'if ... then ...' for implication
- 'and' for conjunction
- 'or' for disjunction
- 'it is not the case that' for negation
- 'for all' for universal quantification
- 'there exist' for existential quantification

Uses curly brackets {} to disambiguate compound formulas.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import random

from .syntax_tree import (
    FormulaNode, AtomNode, NegationNode, BinaryNode, QuantifiedNode,
    Connective, Quantifier, LogicOrder
)
from .atomic_proposition_generator import PropositionPool, AtomicProposition


@dataclass
class RenderConfig:
    """Configuration for natural language rendering."""
    # Connective templates (with {left} and {right} placeholders)
    # Only use approved terms: 'if ... then ...', 'and', 'or', 'it is not the case that', 'for all', 'there exist'
    # Note: Brackets are added programmatically in the render functions
    and_templates: List[str] = field(default_factory=lambda: [
        "{left} and {right}",
    ])
    or_templates: List[str] = field(default_factory=lambda: [
        "{left} or {right}",
    ])
    implies_templates: List[str] = field(default_factory=lambda: [
        "if {left}, then {right}",
    ])
    iff_templates: List[str] = field(default_factory=lambda: [
        "{left} if and only if {right}",
    ])
    not_templates: List[str] = field(default_factory=lambda: [
        "it is not the case that {child}",
    ])

    # FOL-specific negation templates for predicates
    not_predicate_templates: List[str] = field(default_factory=lambda: [
        "it is not the case that {entity} is {predicate}",
    ])

    # Quantifier templates (with {var} and {body} placeholders)
    # Use explicit variable names for clarity with nested quantifiers
    forall_templates: List[str] = field(default_factory=lambda: [
        "for all {var}, {body}",
    ])
    exists_templates: List[str] = field(default_factory=lambda: [
        "there exist {var} such that {body}",
    ])

    # FOL predicate templates
    predicate_templates: List[str] = field(default_factory=lambda: [
        "{entity} is {predicate}",
    ])

    # FOL predicate templates with explicit variable reference
    predicate_with_var_templates: List[str] = field(default_factory=lambda: [
        "{var} is {predicate}",
    ])

    # Whether to vary templates or use consistent ones
    vary_templates: bool = False  # Set to False for consistency

    # Capitalize first letter of output
    capitalize_output: bool = False


class NaturalLanguageRenderer:
    """Convert formula trees to natural language using a proposition pool."""

    def __init__(
        self,
        pool: PropositionPool,
        config: Optional[RenderConfig] = None
    ):
        self.pool = pool
        self.config = config or RenderConfig()

        # Mapping from atom identifiers to natural language
        self._atom_mapping: Dict[str, str] = {}
        self._entity_mapping: Dict[str, str] = {}
        self._predicate_mapping: Dict[str, str] = {}

        # Track bound variables in current scope
        self._bound_variables: Set[str] = set()
        # Map variable names to descriptive names for natural language
        self._variable_names: Dict[str, str] = {}

        # Track if we're rendering FOL
        self._is_fol_mode = False

    def render(self, formula: FormulaNode) -> str:
        """
        Render a formula tree to natural language.

        Resets all atom-to-NL mappings before rendering, so each call
        produces independent mappings.  For rendering multiple formulas
        with *consistent* mappings (e.g. a proof chain), use
        :meth:`register_formulas` followed by :meth:`render_formula`.

        Args:
            formula: The formula tree to render

        Returns:
            Natural language string
        """
        # Reset mappings for fresh render
        self._atom_mapping = {}
        self._entity_mapping = {}
        self._predicate_mapping = {}
        self._bound_variables = set()
        self._variable_names = {}

        # Detect FOL mode
        self._is_fol_mode = self._contains_fol(formula)

        # Pre-assign atoms/predicates to natural language
        if self._is_fol_mode:
            self._assign_fol_mappings(formula)
        else:
            atoms = formula.get_atoms()
            self._assign_atoms(atoms)

        # Render the formula
        result = self._render_node(formula)

        # Capitalize if configured
        if self.config.capitalize_output and result:
            result = result[0].upper() + result[1:]

        return result

    def register_formulas(self, formulas: List[FormulaNode]) -> None:
        """Pre-register atom/predicate/entity mappings for a batch of formulas.

        Call this once with **all** formulas that will appear in a proof
        chain (initial premises, step premises, step conclusions, etc.),
        then use :meth:`render_formula` for each individual formula.
        This guarantees that the same internal atom always maps to the
        same natural-language text.
        """
        self._atom_mapping = {}
        self._entity_mapping = {}
        self._predicate_mapping = {}
        self._bound_variables = set()
        self._variable_names = {}

        self._is_fol_mode = any(self._contains_fol(f) for f in formulas)

        if self._is_fol_mode:
            for f in formulas:
                self._assign_fol_mappings(f)
        else:
            all_atoms: List[str] = []
            for f in formulas:
                all_atoms.extend(f.get_atoms())
            self._assign_atoms(list(set(all_atoms)))

    def render_formula(self, formula: FormulaNode) -> str:
        """Render a formula using previously registered mappings.

        Must call :meth:`register_formulas` first.  Does **not** reset
        mappings, so all formulas rendered with this method share the
        same atom-to-NL assignments.
        """
        self._bound_variables = set()
        result = self._render_node(formula)
        if self.config.capitalize_output and result:
            result = result[0].upper() + result[1:]
        return result

    def render_inference(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode
    ) -> Dict[str, Any]:
        """
        Render a complete inference to natural language.

        Returns dict with 'premises' list and 'conclusion' string.
        """
        # Reset mappings
        self._atom_mapping = {}
        self._entity_mapping = {}
        self._predicate_mapping = {}
        self._bound_variables = set()
        self._variable_names = {}

        # Detect FOL mode from any formula
        self._is_fol_mode = any(self._contains_fol(p) for p in premises) or self._contains_fol(conclusion)

        if self._is_fol_mode:
            # Collect FOL info from all formulas
            for premise in premises:
                self._assign_fol_mappings(premise)
            self._assign_fol_mappings(conclusion)
        else:
            # Collect all atoms from premises and conclusion
            all_atoms = []
            for premise in premises:
                all_atoms.extend(premise.get_atoms())
            all_atoms.extend(conclusion.get_atoms())
            self._assign_atoms(list(set(all_atoms)))

        # Render premises
        rendered_premises = []
        for premise in premises:
            # Reset bound variables for each premise
            self._bound_variables = set()
            text = self._render_node(premise)
            if self.config.capitalize_output and text:
                text = text[0].upper() + text[1:]
            rendered_premises.append(text)

        # Render conclusion
        self._bound_variables = set()
        conclusion_text = self._render_node(conclusion)
        if self.config.capitalize_output and conclusion_text:
            conclusion_text = conclusion_text[0].upper() + conclusion_text[1:]

        return {
            "premises": rendered_premises,
            "conclusion": conclusion_text
        }

    def _contains_fol(self, formula: FormulaNode) -> bool:
        """Check if formula contains FOL elements (quantifiers or predicates)."""
        if isinstance(formula, QuantifiedNode):
            return True
        if isinstance(formula, AtomNode) and formula.is_predicate:
            return True
        if isinstance(formula, NegationNode):
            return self._contains_fol(formula.child)
        if isinstance(formula, BinaryNode):
            return self._contains_fol(formula.left) or self._contains_fol(formula.right)
        return False

    def _assign_fol_mappings(self, formula: FormulaNode) -> None:
        """Assign natural language to FOL predicates and entities."""
        self._collect_fol_elements(formula)

    def _collect_fol_elements(self, formula: FormulaNode) -> None:
        """Recursively collect and map FOL predicates and entities."""
        if isinstance(formula, AtomNode):
            # Map predicate identifier
            if formula.identifier not in self._predicate_mapping:
                if self.pool.predicates:
                    available = [p for p in self.pool.predicates
                                 if p.text not in self._predicate_mapping.values()]
                    if available:
                        pred = random.choice(available)
                        self._predicate_mapping[formula.identifier] = pred.text
                    else:
                        pred = random.choice(self.pool.predicates)
                        self._predicate_mapping[formula.identifier] = pred.text
                else:
                    self._predicate_mapping[formula.identifier] = formula.identifier.lower()

            # Map entity/variable names (constants only)
            for var in formula.variables:
                if var not in self._entity_mapping:
                    # Check if it's a constant (a, b, c, d) - not a bound variable
                    if var in ["a", "b", "c", "d"]:
                        # It's a constant - map to an entity name
                        if self.pool.entities and self.pool.entities.names:
                            available = [e for e in self.pool.entities.names
                                         if e not in self._entity_mapping.values()]
                            if available:
                                self._entity_mapping[var] = random.choice(available)
                            else:
                                self._entity_mapping[var] = random.choice(self.pool.entities.names)
                        else:
                            self._entity_mapping[var] = var.upper()
                    # Variables (x, y, z, w) will be handled during rendering based on scope

        elif isinstance(formula, NegationNode):
            self._collect_fol_elements(formula.child)
        elif isinstance(formula, BinaryNode):
            self._collect_fol_elements(formula.left)
            self._collect_fol_elements(formula.right)
        elif isinstance(formula, QuantifiedNode):
            # Assign a descriptive name to this variable if not already assigned
            if formula.variable not in self._variable_names:
                var_descriptions = ["x", "y", "z", "w", "v", "u"]
                self._variable_names[formula.variable] = formula.variable
            self._collect_fol_elements(formula.body)

    def _assign_atoms(self, atom_ids: List[str]) -> None:
        """Assign natural language to atom identifiers (propositional logic)."""
        unique_ids = list(set(atom_ids))

        # Sample propositions from pool
        if len(unique_ids) <= len(self.pool.propositions):
            sampled = self.pool.sample_propositions(len(unique_ids))
        else:
            # If we need more than available, reuse
            sampled = []
            pool_size = len(self.pool.propositions)
            for i in range(len(unique_ids)):
                idx = i % pool_size
                sampled.append(self.pool.propositions[idx])

        for atom_id, prop in zip(unique_ids, sampled):
            self._atom_mapping[atom_id] = prop.text

    def _render_node(self, node: FormulaNode) -> str:
        """Render a single node to natural language."""
        if isinstance(node, AtomNode):
            return self._render_atom(node)
        elif isinstance(node, NegationNode):
            return self._render_negation(node)
        elif isinstance(node, BinaryNode):
            return self._render_binary(node)
        elif isinstance(node, QuantifiedNode):
            return self._render_quantified(node)
        else:
            return str(node)

    def _render_atom(self, node: AtomNode) -> str:
        """Render an atomic proposition or FOL predicate."""
        # FOL predicate rendering
        if self._is_fol_mode and node.is_predicate:
            predicate = self._predicate_mapping.get(node.identifier, node.identifier.lower())

            if node.variables:
                # Get entity/variable for the variable/constant
                var = node.variables[0]

                # Check if it's a bound variable or a constant
                if var in self._bound_variables:
                    # It's a bound variable - use the variable name directly
                    template = random.choice(self.config.predicate_with_var_templates) if self.config.vary_templates else self.config.predicate_with_var_templates[0]
                    return template.format(var=var, predicate=predicate)
                elif var in ["a", "b", "c", "d"]:
                    # It's a constant - use the mapped entity name
                    entity = self._entity_mapping.get(var, var.upper())
                    template = random.choice(self.config.predicate_templates) if self.config.vary_templates else self.config.predicate_templates[0]
                    return template.format(entity=entity, predicate=predicate)
                else:
                    # It's an unbound variable (shouldn't happen in well-formed formulas)
                    # but handle gracefully
                    template = random.choice(self.config.predicate_with_var_templates) if self.config.vary_templates else self.config.predicate_with_var_templates[0]
                    return template.format(var=var, predicate=predicate)
            else:
                return f"is {predicate}"

        # Propositional logic rendering
        if node.identifier in self._atom_mapping:
            return self._atom_mapping[node.identifier]

        # Fallback: use identifier directly
        if node.variables:
            return f"{node.identifier}({', '.join(node.variables)})"
        return node.identifier

    def _render_negation(self, node: NegationNode) -> str:
        """Render a negation with brackets for disambiguation."""
        # Special handling for FOL predicate negation
        if self._is_fol_mode and isinstance(node.child, AtomNode) and node.child.is_predicate:
            atom = node.child
            predicate = self._predicate_mapping.get(atom.identifier, atom.identifier.lower())

            if atom.variables:
                var = atom.variables[0]

                if var in self._bound_variables:
                    # Bound variable
                    return "{it is not the case that " + var + " is " + predicate + "}"
                elif var in ["a", "b", "c", "d"]:
                    # Constant
                    entity = self._entity_mapping.get(var, var.upper())
                    template = random.choice(self.config.not_predicate_templates) if self.config.vary_templates else self.config.not_predicate_templates[0]
                    inner = template.format(entity=entity, predicate=predicate)
                    return "{" + inner + "}"
                else:
                    return "{it is not the case that " + var + " is " + predicate + "}"

        child_text = self._render_node(node.child)

        templates = self.config.not_templates
        if self.config.vary_templates:
            template = random.choice(templates)
        else:
            template = templates[0]

        inner = template.format(child=child_text)
        return "{" + inner + "}"

    def _render_binary(self, node: BinaryNode) -> str:
        """Render a binary connective with brackets for disambiguation."""
        left_text = self._render_node(node.left)
        right_text = self._render_node(node.right)

        # Choose template based on connective
        if node.connective == Connective.AND:
            templates = self.config.and_templates
        elif node.connective == Connective.OR:
            templates = self.config.or_templates
        elif node.connective == Connective.IMPLIES:
            templates = self.config.implies_templates
        elif node.connective == Connective.IFF:
            templates = self.config.iff_templates
        else:
            templates = ["{left} {connective} {right}"]

        if self.config.vary_templates:
            template = random.choice(templates)
        else:
            template = templates[0]

        inner = template.format(
            left=left_text,
            right=right_text,
            connective=node.connective.value
        )
        return "{" + inner + "}"

    def _render_quantified(self, node: QuantifiedNode) -> str:
        """Render a quantified formula with brackets for disambiguation."""
        # Add the variable to bound variables before rendering body
        self._bound_variables.add(node.variable)

        # Render the body with the variable now in scope
        body_text = self._render_node(node.body)

        # Choose quantifier template
        if node.quantifier == Quantifier.FORALL:
            templates = self.config.forall_templates
        else:
            templates = self.config.exists_templates

        template = random.choice(templates) if self.config.vary_templates else templates[0]

        inner = template.format(
            var=node.variable,
            body=body_text
        )

        # Note: we don't remove the variable from bound_variables here
        # because in a single formula rendering, once bound it stays bound
        # (this matches standard variable scoping semantics)

        return "{" + inner + "}"


class InferenceRenderer:
    """Convenience class for rendering complete inferences."""

    def __init__(
        self,
        pool: PropositionPool,
        config: Optional[RenderConfig] = None
    ):
        self.renderer = NaturalLanguageRenderer(pool, config)

    def render(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode,
        include_therefore: bool = True
    ) -> str:
        """
        Render an inference as a complete text.

        Args:
            premises: List of premise formula trees
            conclusion: Conclusion formula tree
            include_therefore: Whether to add "Therefore" before conclusion

        Returns:
            Complete inference as text
        """
        result = self.renderer.render_inference(premises, conclusion)

        # Build text
        premise_texts = result["premises"]
        conclusion_text = result["conclusion"]

        parts = []
        for i, premise in enumerate(premise_texts):
            parts.append(premise + ".")

        if include_therefore:
            parts.append(f"Therefore, {conclusion_text.lower()}.")
        else:
            parts.append(f"{conclusion_text}.")

        return " ".join(parts)

    def render_structured(
        self,
        premises: List[FormulaNode],
        conclusion: FormulaNode
    ) -> Dict[str, Any]:
        """
        Render to structured format for data generation.

        Returns:
            Dict with 'premises' list and 'conclusion' string
        """
        return self.renderer.render_inference(premises, conclusion)
