# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run tests
pytest scripts/test_lora_finetune.py -v

# Run a single test
pytest scripts/test_lora_finetune.py::TestClassName::test_method -v

# Generate Stage 0 proof chain data (premises + conclusion only)
python3 scripts/generate_chain_data.py --stage 0 -n 10000 -o ./data/

# Generate Stage 1 proof chain data (full proof traces)
python3 scripts/generate_chain_data.py --stage 1 -n 10000 -o ./data/

# Generate simple logic inference examples
python3 scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl

# Stage 0 SFT (teach conclusions)
python3 scripts/lora_finetune.py \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --train-path ./data/chain_stage0_n10000.jsonl \
  --output-dir ./outputs/stage0 \
  --bf16 --4bit

# Stage 1 SFT (teach full proof traces; continues from Stage 0 adapter)
python3 scripts/lora_finetune.py \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --lora-adapter ./outputs/stage0/final \
  --train-path ./data/chain_stage1_n10000.jsonl \
  --output-dir ./outputs/stage1 \
  --bf16 --4bit

# Prepare FOLIO data for GRPO
python3 scripts/prepare_grpo_data.py --output-path data/grpo_folio_prompted.jsonl

# Run GRPO reinforcement learning
python3 scripts/train_grpo.py \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --lora-adapter ./outputs/stage1/final \
  --train-path data/grpo_folio_prompted.jsonl \
  --output-dir ./outputs/grpo_folio \
  --task-type verdict --verifier-weight 0.3 --outcome-weight 0.7 \
  --group-size 8 --batch-size 2 --grad-accum 1 \
  --max-prompt-length 2048 --max-gen-length 300 \
  --n-verify-workers 16 \
  --kl-coeff 0.1 --lr 5e-6 --epochs 3 \
  --bf16 --4bit --temperature 0.8

# Run inference
python3 scripts/inference_demo.py "prompt text"

# Switch models
python3 scripts/switch_model.py --list
python3 scripts/switch_model.py mistralai/Mistral-Small-3.2-24B-Instruct-2506

# Evaluate
python3 scripts/evaluate.py --model_outputs outputs.jsonl --ground_truth data/test.jsonl
```

Note: use `python3` (not `python`) on this system.

## Architecture

### Data Flow

Data generation (`src/data/`) → SFT Training (`src/training/`) → GRPO RL (`src/training/grpo.py`) → Inference (`src/inference/`) → Verification (`src/verification/`)

### Logic Formula System (`src/data/`)

The core data generation builds on a polymorphic AST for logic formulas:

- **`syntax_tree.py`**: `FormulaNode` ABC with subtypes `AtomNode`, `NegationNode`, `BinaryNode`, `QuantifiedNode`. `RandomTreeGenerator` creates random formulas with configurable depth/weights. Supports both propositional and first-order logic.
- **`inference_generator.py`**: 11 propositional patterns (Modus Ponens, Modus Tollens, etc.) and 5 FOL patterns (Universal Instantiation, etc.). Each pattern is instantiated with random subformulas to produce `Inference` objects (premises + conclusion + pattern name).
- **`nl_renderer.py`**: Converts formula ASTs to natural language. Uses curly brackets `{}` to disambiguate nested compound formulas.
- **`atomic_proposition_generator.py`**: Calls OpenAI API to generate diverse atomic propositions. Maintains `PropositionPool`/`EntityPool` for reuse across examples.

### Chain Generation (`src/data/chain_generator.py`)

Generates multi-step natural deduction proof chains using **backward (goal-directed) construction**. Starts with a final conclusion and recursively determines which rule derives it and what premises are needed.

- **Backward construction**: `_justify(goal, depth)` picks a rule that can derive the goal, adds required premises, and recurses on sub-goals. Sound by construction — every premise is essential, no fabrication, no redundancy.
- **All 13 ND rules**: AND/OR/IMPLIES/IFF intro+elim, NOT intro+elim, FORALL intro+elim, EXISTS intro+elim, OR_ELIM (discharge), EXISTS_ELIM (discharge).
- **Discharge rules**: IMPLIES_INTRO, NOT_INTRO, OR_ELIM, EXISTS_ELIM create assume/derive/discharge proof blocks. Discharged assumptions are used in 50% of cases.
- **Compound goals**: Antecedents can be compound formulas with 1-4 connectives, forcing introduction rules (AND_INTRO, OR_INTRO, etc.) to fire in sub-derivations.
- **Stage 0 restrictions**: No negated conclusions, no NOT_INTRO/NOT_ELIM as final step.
- **Z3 backstop**: After generation, verifies the full chain via Z3 (should never reject).
- **Two-stage training**: Stage 0 = premises + conclusion only; Stage 1 = full proof trace.

### Staged Verification (`src/verification/`)

- **`verifier.py`** + **`translator.py`**: Z3-based symbolic entailment check. Semi-formal NL → `FormulaNode` AST (via recursive descent parser) → Z3 expressions → `solver.check() == unsat`.
- **`parser.py`**: Handles `{if P, then Q}`, `{P and Q}`, `{for all x, ...}` syntax. Input can be free text or structured `<PREMISE>`/`<CONCLUSION>` tags.
- **`dag_verifier.py`**: Legacy three-stage fallback pipeline (NLI → semantic parsing → Z3/Datalog). Still used for graceful degradation.

### GRPO Training (`src/training/grpo.py`, `src/training/soundness_reward.py`)

- **Algorithm**: Sample G=8 responses per prompt, score with outcome-gated process reward, compute group-relative advantages, update with importance-weighted surrogate loss + KL penalty.
- **Reward formula**: `correct → 0.7×(+1) + 0.3×log(1+sound_steps)`; `wrong → 0.7×(−1)` (no process offset); `no_verdict → 0.3×log(1+sound_steps)`.
- **Parallel Z3**: `ProcessPoolExecutor` with `spawn` start method. Z3 is NOT thread-safe — must use processes. Per-call context manager prevents idle worker OS kills. Workers initialized fresh each scoring call (~1–2s), torn down immediately.
- **Flash Attention 2**: Enabled by default; `--no-flash-attn` to disable.
- **Speed on H200**: ~70–80s/step with `group_size=8`, `batch_size=2`, parallel Z3.

### Training (`src/training/`, `scripts/lora_finetune.py`)

- `scripts/lora_finetune.py` is the main SFT entry point (~1100 lines). Contains `MODEL_REGISTRY` mapping model keys to HuggingFace paths, LoRA target modules, and max context lengths. Models can be referenced by registry key (e.g., `deepseek-v3`) or full HuggingFace path.
- **Prompt token masking**: Labels are set to -100 for prompt tokens so only conclusion/target tokens contribute to loss.
- Supports multiple dataset formats: JSONL, HuggingFace datasets, and logic benchmark format.
- **LoRA config**: r=16, alpha=32, target modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 modules, 101M/24B trainable = 0.42%).

### Configuration System (`src/utils/config_loader.py`)

Config priority: `.env` file > system environment variables > `config.yaml`. Key env vars: `MODEL_NAME`, `PREMISE_VERIFIER_MODEL`, `INFERENCE_VERIFIER_MODEL`, `DATA_DIR`, `OUTPUT_DIR`.

### Inference (`src/inference/predictor.py`)

Uses vLLM for efficient batched inference with optional LoRA adapter loading and tensor parallelism. Patches Mistral 3.x tokenizer compatibility.

## Experiments

Five experiments in `experiments/`:

- **exp1_premise_perturbation**: Perturbs one premise per example, measures accuracy drop and sensitivity (16% sensitivity, 86% → 74% accuracy).
- **exp2_positional_permutation**: Shuffles premise order, measures stability.
- **exp3_monotonicity**: Adds supporting/contradicting premises, measures monotonicity violations.
- **exp4_lora_svd**: SVD analysis of LoRA matrices (Frobenius norms, singular value spectra, effective rank) across Stage 0 and Stage 1 adapters. `run.py` loads safetensors, computes `||BA||_F` efficiently via trace trick, generates heatmaps and spectra plots.
- **exp5_probing**: Linear probing classifiers on hidden states at each layer.

## Key Conventions

- **Config objects**: Dataclasses with `*Config` suffix (e.g., `TreeGeneratorConfig`, `VerifierConfig`, `RetrievalConfig`)
- **Enums for types**: `Connective`, `Quantifier`, `InferencePattern`, `LogicOrder`
- **Optional imports**: Heavy dependencies (torch, transformers, vllm) use try/except with graceful fallbacks
- **Tests use mocks**: `MockTokenizer` and `MagicMock` avoid loading real models during testing
- **Finetuned model outputs** are gitignored by provider prefix pattern (e.g., `mistralai_*/`, `meta-llama_*/`)
- **`data/` directory** is gitignored; generated JSONL files live there at runtime
- **Z3 threading**: Always use `ProcessPoolExecutor` (spawn), never `ThreadPoolExecutor` for Z3 work
