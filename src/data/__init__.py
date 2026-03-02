"""Data generation, curation, and schema modules."""

from .schema import (
    Annotation,
    Premise,
    EvidenceSpan,
    validate_annotation,
    safe_parse_model_output,
    format_prompt,
    ANNOTATION_SCHEMA
)

# Optional imports for modules that depend on transformers
try:
    from .generator import SyntheticDataGenerator, GenerationConfig, DataAugmenter
    from .curation import DataCurator
    _HAS_TRANSFORMERS = True
except ImportError:
    SyntheticDataGenerator = None
    GenerationConfig = None
    DataAugmenter = None
    DataCurator = None
    _HAS_TRANSFORMERS = False
from .logic_templates import (
    LogicTemplate,
    LogicType,
    get_all_templates,
    get_templates_by_type,
    sample_template,
    get_template_by_name,
    PROPOSITIONAL_TEMPLATES,
    FIRST_ORDER_TEMPLATES
)
from .atomic_proposition_generator import (
    AtomicPropositionGenerator,
    AtomicProposition,
    PropositionPool,
    EntityPool,
    GeneratorConfig as PropositionGeneratorConfig,
    create_fallback_pool,
    TOPIC_CATEGORIES
)
from .syntax_tree import (
    FormulaNode,
    AtomNode,
    NegationNode,
    BinaryNode,
    QuantifiedNode,
    LogicOrder,
    Connective,
    Quantifier,
    TreeGeneratorConfig,
    RandomTreeGenerator,
    negate,
    make_implication,
    make_conjunction,
    make_disjunction,
    make_biconditional
)
from .inference_generator import (
    Inference,
    InferencePattern,
    InferenceGenerator,
    InferenceGeneratorConfig,
    generate_inference,
    generate_inferences,
    PROPOSITIONAL_PATTERNS,
    FOL_PATTERNS
)
from .nl_renderer import (
    NaturalLanguageRenderer,
    InferenceRenderer,
    RenderConfig
)
from .chain_generator import (
    ChainGenerator,
    ChainGeneratorConfig,
    NDRule,
    ProofStep,
    ProofChain,
    CompressedSegment,
)
from .applied_chain_generator import (
    AppliedChainGenerator,
    AppliedGeneratorConfig,
    AppliedExample,
    APPLIED_DOMAINS,
)
from .stage2_generator import (
    Stage2Generator,
    Stage2Config,
    Stage2Example,
)

__all__ = [
    # Schema
    "Annotation",
    "Premise",
    "EvidenceSpan",
    "validate_annotation",
    "safe_parse_model_output",
    "format_prompt",
    "ANNOTATION_SCHEMA",
    # Data generation
    "SyntheticDataGenerator",
    "GenerationConfig",
    "DataAugmenter",
    "DataCurator",
    # Logic templates
    "LogicTemplate",
    "LogicType",
    "get_all_templates",
    "get_templates_by_type",
    "sample_template",
    "get_template_by_name",
    "PROPOSITIONAL_TEMPLATES",
    "FIRST_ORDER_TEMPLATES",
    # Atomic proposition generation (pooled)
    "AtomicPropositionGenerator",
    "AtomicProposition",
    "PropositionPool",
    "EntityPool",
    "PropositionGeneratorConfig",
    "create_fallback_pool",
    "TOPIC_CATEGORIES",
    # Syntax tree (random formula generation)
    "FormulaNode",
    "AtomNode",
    "NegationNode",
    "BinaryNode",
    "QuantifiedNode",
    "LogicOrder",
    "Connective",
    "Quantifier",
    "TreeGeneratorConfig",
    "RandomTreeGenerator",
    "negate",
    "make_implication",
    "make_conjunction",
    "make_disjunction",
    "make_biconditional",
    # Inference generation
    "Inference",
    "InferencePattern",
    "InferenceGenerator",
    "InferenceGeneratorConfig",
    "generate_inference",
    "generate_inferences",
    "PROPOSITIONAL_PATTERNS",
    "FOL_PATTERNS",
    # Natural language rendering
    "NaturalLanguageRenderer",
    "InferenceRenderer",
    "RenderConfig",
    # Chain generation (natural deduction proof chains)
    "ChainGenerator",
    "ChainGeneratorConfig",
    "NDRule",
    "ProofStep",
    "ProofChain",
    "CompressedSegment",
    # Applied chain generation (domain-specific examples)
    "AppliedChainGenerator",
    "AppliedGeneratorConfig",
    "AppliedExample",
    "APPLIED_DOMAINS",
    # Stage 2 generation (semi-formal CoT on logic benchmarks)
    "Stage2Generator",
    "Stage2Config",
    "Stage2Example",
]
