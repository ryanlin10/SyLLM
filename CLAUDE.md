# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run tests
pytest scripts/test_lora_finetune.py -v

# Run a single test
pytest scripts/test_lora_finetune.py::TestClassName::test_method -v

# Generate logic training data
python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl

# Fine-tune a model (main training entry point)
python scripts/lora_finetune.py \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --train-path ./data/train.jsonl \
  --output-dir ./outputs

# Run inference
python scripts/inference_demo.py "prompt text"

# Switch models
python scripts/switch_model.py --list
python scripts/switch_model.py mistralai/Mistral-Small-3.2-24B-Instruct-2506

# Evaluate
python scripts/evaluate.py --model_outputs outputs.jsonl --ground_truth data/test.jsonl
```

## Architecture

### Data Flow

Data generation (`src/data/`) → Training (`src/training/`) → Inference (`src/inference/`) → Verification (`src/verification/`)

### Logic Formula System (`src/data/`)

The core data generation builds on a polymorphic AST for logic formulas:

- **`syntax_tree.py`**: `FormulaNode` ABC with subtypes `AtomNode`, `NegationNode`, `BinaryNode`, `QuantifiedNode`. `RandomTreeGenerator` creates random formulas with configurable depth/weights. Supports both propositional and first-order logic.
- **`inference_generator.py`**: 11 propositional patterns (Modus Ponens, Modus Tollens, etc.) and 5 FOL patterns (Universal Instantiation, etc.). Each pattern is instantiated with random subformulas to produce `Inference` objects (premises + conclusion + pattern name).
- **`nl_renderer.py`**: Converts formula ASTs to natural language. Uses curly brackets `{}` to disambiguate nested compound formulas.
- **`atomic_proposition_generator.py`**: Calls OpenAI API to generate diverse atomic propositions. Maintains `PropositionPool`/`EntityPool` for reuse across examples.

### Chain Generation (`src/data/chain_generator.py`)

Generates multi-step natural deduction proof chains using **backward (goal-directed)
construction**. Starts with a final conclusion and recursively determines which rule
derives it and what premises are needed.

- **Backward construction**: `_justify(goal, depth)` picks a rule that can derive the
  goal, adds required premises, and recurses on sub-goals. Sound by construction —
  every premise is essential, no fabrication, no redundancy.
- **All 13 ND rules**: AND/OR/IMPLIES/IFF intro+elim, NOT intro+elim, FORALL intro+elim,
  EXISTS intro+elim, OR_ELIM (discharge), EXISTS_ELIM (discharge).
- **Discharge rules**: IMPLIES_INTRO, NOT_INTRO, OR_ELIM, EXISTS_ELIM create
  assume/derive/discharge proof blocks. Discharged assumptions are used in 50% of cases.
- **Compound goals**: Antecedents can be compound formulas with 1-4 connectives, forcing
  introduction rules (AND_INTRO, OR_INTRO, etc.) to fire in sub-derivations.
- **Stage 0 restrictions**: No negated conclusions, no NOT_INTRO/NOT_ELIM as final step.
- **Z3 backstop**: After generation, verifies the full chain via Z3 (should never reject).
- **Two-stage training**: Stage 0 = premises + conclusion only; Stage 1 = full proof trace.

### Staged Verification (`src/verification/`)

Verification uses a three-stage fallback pipeline in `dag_verifier.py`:
1. **NLI check** (`verifier.py`): Fast approximate premise/inference classification
2. **Semantic parsing** (`semantic_parser.py`): Converts NL to formal logic (Datalog/Z3/FOL targets)
3. **Symbolic solving**: Z3 SMT solver or Datalog for provably correct entailment checking

Each stage can produce a final verdict (`accept`/`review`/`reject`). Later stages only run if earlier stages are inconclusive. The verifier gracefully degrades to rule-based fallbacks if models are unavailable.

### Training (`src/training/`, `scripts/lora_finetune.py`)

- `scripts/lora_finetune.py` is the main training entry point (~1100 lines). It contains `MODEL_REGISTRY` mapping model keys to HuggingFace paths, LoRA target modules, and max context lengths. Models can be referenced by registry key (e.g., `deepseek-v3`) or full HuggingFace path.
- **Prompt token masking**: Labels are set to -100 for prompt tokens so only conclusion/target tokens contribute to loss.
- Supports multiple dataset formats: JSONL, HuggingFace datasets, and logic benchmark format.

### Configuration System (`src/utils/config_loader.py`)

Config priority: `.env` file > system environment variables > `config.yaml`. Key env vars: `MODEL_NAME`, `PREMISE_VERIFIER_MODEL`, `INFERENCE_VERIFIER_MODEL`, `DATA_DIR`, `OUTPUT_DIR`.

### Inference (`src/inference/predictor.py`)

Uses vLLM for efficient batched inference with optional LoRA adapter loading and tensor parallelism.

## Key Conventions

- **Config objects**: Dataclasses with `*Config` suffix (e.g., `TreeGeneratorConfig`, `VerifierConfig`, `RetrievalConfig`)
- **Enums for types**: `Connective`, `Quantifier`, `InferencePattern`, `LogicOrder`
- **Optional imports**: Heavy dependencies (torch, transformers, vllm) use try/except with graceful fallbacks
- **Tests use mocks**: `MockTokenizer` and `MagicMock` avoid loading real models during testing
- **Finetuned model outputs** are gitignored by provider prefix pattern (e.g., `mistralai_*/`, `meta-llama_*/`)
