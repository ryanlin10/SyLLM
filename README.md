# SyLLM: Structured Reasoning LLM with Automated Verification

A comprehensive system for fine-tuning LLMs to output structured reasoning in premise-conclusion format with automated verification, supporting propositional and first-order logic.

## Overview

SyLLM enables LLMs to:
- Output structured reasoning chains with explicit premises and conclusions
- Automatically verify premise factuality and logical inference using a staged pipeline (NLI, semantic parsing, symbolic solvers)
- Ground claims in retrieved evidence via RAG
- Generate synthetic logic reasoning datasets (propositional and first-order logic)
- Improve reasoning quality through reinforcement learning with verifier-based rewards

## Features

- **Structured Output**: Enforces JSON format with premises and conclusions
- **Multi-Model Support**: Model registry with support for Mistral, DeepSeek, Llama, and more — switch models via CLI or `.env`
- **Logic Reasoning**: Propositional logic (`P ^ Q -> R`) and first-order logic (`forall x. Human(x) -> Mortal(x)`)
- **Staged Verification**: Three-stage verifier pipeline — NLI classification, semantic parsing to formal logic, and symbolic solving (Z3/Datalog)
- **RAG Integration**: Retrieval-augmented generation with FAISS for evidence grounding
- **Data Generation**: Synthetic data pipeline using random syntax trees and OpenAI API for diverse atomic propositions
- **Fine-tuning**: LoRA-based efficient fine-tuning with automatic target module detection, mixed precision, and gradient checkpointing
- **Reinforcement Learning**: GRPO training with soundness and outcome-based rewards
- **Evaluation**: Comprehensive metrics for premise accuracy, entailment, verifier calibration, and parse success rate
- **Inference**: Efficient serving with vLLM and tensor parallelism support

## Project Structure

```
SyLLM/
├── config.yaml                  # Main configuration file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── src/
│   ├── data/                    # Data generation and curation
│   │   ├── schema.py            # Data schema definitions
│   │   ├── generator.py         # Synthetic data generation
│   │   ├── curation.py          # Data validation and splitting
│   │   ├── syntax_tree.py       # Logic formula AST (propositional/FOL)
│   │   ├── atomic_proposition_generator.py  # Atomic propositions via OpenAI API
│   │   ├── inference_generator.py  # Logic inference example generation
│   │   ├── nl_renderer.py       # Logic formulas to natural language
│   │   └── logic_templates.py   # Pre-defined logic patterns
│   ├── retrieval/               # RAG components
│   │   └── retriever.py         # Document retrieval with FAISS
│   ├── verification/            # Automated verification
│   │   ├── verifier.py          # NLI-based premise and inference verifiers
│   │   ├── dag_verifier.py      # Staged verification (NLI → semantic parsing → Z3/Datalog)
│   │   ├── semantic_parser.py   # Natural language to formal logic conversion
│   │   └── repair.py            # JSON repair utilities
│   ├── training/                # Training scripts
│   │   ├── finetune.py          # Fine-tuning with LoRA
│   │   ├── grpo.py              # GRPO reinforcement learning trainer
│   │   └── soundness_reward.py  # Reward function for RL training
│   ├── evaluation/              # Evaluation framework
│   │   └── evaluator.py         # Metrics and evaluation
│   ├── inference/               # Inference pipeline
│   │   └── predictor.py         # vLLM-based model prediction
│   └── utils/                   # Utilities
│       └── config_loader.py     # YAML config loading
├── scripts/                     # Executable scripts
│   ├── lora_finetune.py         # Main LoRA fine-tuning script
│   ├── generate_logic_data.py   # Generate synthetic logic data
│   ├── generate_data.py         # Generate synthetic data
│   ├── prepare_data.py          # Prepare and split data
│   ├── train_finetune.py        # Run fine-tuning (wrapper)
│   ├── train_grpo.py            # Run GRPO training
│   ├── evaluate.py              # Evaluate model
│   ├── inference_demo.py        # Demo inference with vLLM
│   ├── switch_model.py          # Model switching utility
│   ├── benchmark_logic.py       # Logic reasoning benchmarks
│   ├── inspect_tokenizer.py     # Tokenizer inspection utility
│   └── test_lora_finetune.py    # Test suite
└── data/                        # Data directory (created at runtime)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ryanlin10/LLM-Syllogistic-Formal-Verification.git
cd SyLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Edit `config.yaml` to set model paths, data paths, and hyperparameters
   - Or use `.env` to set `MODEL_NAME` (takes priority over config.yaml)

## Quick Start

### 1. Generate Training Data

Generate synthetic logic reasoning examples:
```bash
python scripts/generate_logic_data.py -n 1000 -o ./data/logic_train.jsonl
```

Or generate premise-conclusion pairs from documents:
```bash
python scripts/generate_data.py
```

### 2. Prepare Data Splits

```bash
python scripts/prepare_data.py
```

This validates, balances, and splits data into train/val/test sets.

### 3. Fine-tune the Model

```bash
python scripts/lora_finetune.py \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --train-path ./data/train.jsonl \
  --output-dir ./outputs \
  --use-wandb
```

The script supports automatic LoRA target module detection, mixed precision training (FP16/BF16), gradient checkpointing, and checkpoint resumption.

### 4. (Optional) GRPO Reinforcement Learning

```bash
python scripts/train_grpo.py --train-path ./data/rl_train.jsonl
```

Further improves the model using GRPO with verifier-based rewards.

### 5. Run Inference

```bash
python scripts/inference_demo.py "Given premises, what is the conclusion?"
```

Uses vLLM for efficient inference with optional tensor parallelism.

### 6. Evaluate

```bash
python scripts/evaluate.py --model_outputs outputs.jsonl --ground_truth data/test.jsonl
```

## Configuration

### Model Switching

Switch between models using a `.env` file or the CLI utility:

```bash
# Create .env file from template
cp .env.example .env

# Switch models
python scripts/switch_model.py mistralai/Mistral-Small-3.2-24B-Instruct-2506

# List available models
python scripts/switch_model.py --list

# Check current model
python scripts/switch_model.py --current
```

### Supported Models

The model registry includes:
- **Mistral**: `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (currently configured)
- **DeepSeek**: `deepseek-ai/deepseek-v3`, `deepseek-ai/deepseek-v2`
- **Llama**: `meta-llama/Llama-2-7b-hf`

Each model has automatic LoRA target module detection and configurable context lengths.

### Full Configuration

Edit `config.yaml` to customize:

- **Model**: Base model path (or set via `MODEL_NAME` in `.env`), LoRA settings
- **Training**: Batch sizes, learning rates, epochs
- **Reinforcement Learning**: GRPO parameters and reward settings
- **Verifier**: Confidence thresholds, model paths
- **Retrieval**: Embedding model, top-k retrieval
- **Data**: Paths for train/val/test splits

## Data Format

Training data uses JSONL format:

```json
{
  "id": "uuid",
  "context": "Question and retrieved context",
  "premises": [
    {
      "id": "p1",
      "text": "Factual premise statement",
      "evidence_spans": [
        {"doc_id": "D1", "start": 120, "end": 220, "text": "..."}
      ]
    }
  ],
  "conclusion": {
    "text": "Conclusion that follows from premises",
    "type": "entailment|contradiction|unsupported"
  },
  "confidence": 0.9,
  "timestamp": "YYYY-MM-DD"
}
```

## Verification System

The verifier uses a staged pipeline:

1. **Premise Verifier (NLI)**: Checks if each premise is supported by evidence
   - Labels: `supported`, `contradicted`, `unverifiable`
   - Uses NLI-style classifier with rule-based fallback

2. **Inference Verifier (NLI)**: Checks if the conclusion follows from premises
   - Labels: `entailed`, `non-entailed`, `weakly_supported`
   - Uses multi-premise entailment model

3. **Semantic Parsing**: Converts natural language to formal logic representations
   - Supports propositional and first-order logic formulas

4. **Symbolic Verification**: Formal verification using Z3 SMT solver and Datalog
   - Provides provably correct entailment checking when symbolic translation succeeds

Final verdict: `accept`, `review`, or `reject`

## Evaluation Metrics

- **Premise Precision/Recall**: Accuracy of generated premises
- **Evidence Recall**: Fraction of premises with linked evidence
- **Entailment Accuracy**: Correctness of conclusions
- **Verifier Calibration**: Expected Calibration Error (ECE)
- **Parse Success Rate**: Percentage of valid JSON outputs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~20GB disk space for models and data

## License

MIT License. Ensure compliance with individual model licenses for commercial use.

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure code passes linting

## Roadmap

- [ ] Human-in-the-loop review UI
- [ ] Active learning pipeline
- [ ] Production deployment templates
- [ ] Multi-domain adaptation
- [ ] Enhanced verifier training data
