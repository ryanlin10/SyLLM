#!/usr/bin/env python3
"""
Comprehensive LoRA Fine-tuning Script with Model Switching and Data Loading.

Features:
- Easy model switching via CLI or environment variables
- Train/validation/test data loading with automatic splitting
- Multiple dataset format support (JSONL, HuggingFace datasets, logic benchmarks)
- Comprehensive logging and checkpointing
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Weights & Biases integration
- Resume from checkpoint support
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mistral chat template support
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from bitsandbytes import BitsAndBytesConfig
    BNBS_AVAILABLE = True
except ImportError:
    BNBS_AVAILABLE = False


# =============================================================================
# Model Registry - Easy Model Switching
# =============================================================================

MODEL_REGISTRY = {
    # DeepSeek models
    "deepseek-v3": {
        "name": "deepseek-ai/deepseek-v3",
        "description": "DeepSeek V3 - Recommended for structured reasoning",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    "deepseek-v2": {
        "name": "deepseek-ai/deepseek-v2",
        "description": "DeepSeek V2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    # Llama models
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "Llama 2 7B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    "llama2-7b-chat": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Llama 2 7B Chat",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    "llama2-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "description": "Llama 2 13B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    "llama3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "description": "Llama 3 8B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "llama3-8b-instruct": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Llama 3 8B Instruct",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    # Mistral models
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "mistral-7b-instruct": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B Instruct v0.2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "mixtral-8x7b": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Mixtral 8x7B Instruct (MoE)",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "mistral-small-24b-instruct": {
        "name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "description": "Mistral Small 3.2 24B Instruct (supports 128k context)",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,  # Training default; model supports up to 128k
        "model_class": "Mistral3ForConditionalGeneration",
    },
    # Qwen models
    "qwen2-7b": {
        "name": "Qwen/Qwen2-7B",
        "description": "Qwen 2 7B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "qwen2-7b-instruct": {
        "name": "Qwen/Qwen2-7B-Instruct",
        "description": "Qwen 2 7B Instruct",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    # Phi models
    "phi-3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Phi-3 Mini 4K Instruct",
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "max_length": 4096,
    },
}


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model info from registry or return custom model config."""
    # Check registry by key
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    # Check if model_key matches a registry entry by HF name
    for key, info in MODEL_REGISTRY.items():
        if info.get("name") == model_key:
            return info
    # Assume it's a HuggingFace model path
    return {
        "name": model_key,
        "description": f"Custom model: {model_key}",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "max_length": 2048,
    }


def list_available_models():
    """Print available models."""
    print("\n=== Available Models ===\n")
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:25s} - {info['description']}")
    print("\n  You can also specify any HuggingFace model path directly.")
    print()


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    text_column: str = "text"
    max_length: int = 2048
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    preprocessing_num_workers: int = 4
    # Inference-focused training: only compute loss on conclusions
    conclusion_only_loss: bool = True
    # System prompt for logical reasoning
    system_prompt: str = "You are a logical reasoning assistant. Given the following premises, derive their valid conclusion."


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    """Training configuration."""
    output_dir: str = "/tmp/lora_finetuned"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    resume_from_checkpoint: Optional[str] = None
    init_adapter_path: Optional[str] = None  # Load prior LoRA adapter as init for stage continuation
    use_wandb: bool = True
    wandb_project: str = "syllm-lora-finetune"
    wandb_run_name: Optional[str] = None
    seed: int = 42
    use_8bit: bool = False
    use_4bit: bool = False
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None


# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """Data loader with Mistral chat template support for logical reasoning training."""

    def __init__(self, config: DataConfig, tokenizer, model_name: str, cache_dir: Optional[str] = None, hf_token: Optional[str] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.model_name = model_name
        # Initialize Mistral tokenizer for chat template formatting (same model as being finetuned)
        print(f"Initializing Mistral tokenizer from {model_name}...")
        tokenizer_path = download_tokenizer_from_hf_hub(model_name, token=hf_token, cache_dir=cache_dir)
        self.mistral_tokenizer = MistralTokenizer.from_file(tokenizer_path)

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_premises_conclusion(self, example: Dict[str, Any]) -> Tuple[List[str], str]:
        """
        Extract premises and content/conclusion from the Annotation format.

        Expected format from generate_logic_data.py:
        {
            "id": "uuid",
            "premises": [{"id": "p1", "text": "..."}, {"id": "p2", "text": "..."}],
            "content": "conclusion text",
            ...
        }

        Returns:
            Tuple of (list of premise texts, content/conclusion text)
        """
        premises = []
        conclusion = ""

        # Extract premises - list of {"id": "p1", "text": "premise text"}
        if "premises" in example:
            for p in example["premises"]:
                if isinstance(p, dict) and "text" in p:
                    premises.append(p["text"])
                elif isinstance(p, dict):
                    # Fallback: use string representation
                    premises.append(str(p))
                else:
                    premises.append(str(p))

        # Extract content (or legacy "conclusion") - either string or dict with "text" field
        conc = example.get("content") or example.get("conclusion", "")
        if isinstance(conc, dict) and "text" in conc:
            conclusion = conc["text"]
        else:
            conclusion = str(conc) if conc else ""

        return premises, conclusion

    def _format_user_message(self, premises: List[str]) -> str:
        """Format premises as user message with <PREMISE> tags."""
        parts = []
        for premise in premises:
            if premise and premise.strip():
                parts.append(f"<PREMISE> {premise.strip()} </PREMISE>")
        return " ".join(parts)

    def _format_assistant_message(self, conclusion: str) -> str:
        """Format conclusion as assistant message.

        If the conclusion already contains proof-trace tags (e.g. from a
        Stage 1 chain data proof trace), use it as-is.  Otherwise wrap in
        ``<CONCLUSION>`` tags.
        """
        stripped = conclusion.strip()
        if "<CONCLUSION>" in stripped or "<PREMISE>" in stripped:
            return stripped
        return f"<CONCLUSION> {stripped} </CONCLUSION>"

    def format_example(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format a single example for training using Mistral chat template.

        Handles two data formats:

        **Stage 0/1 (Annotation format):**
            User message:  ``<PREMISE> p1 </PREMISE> <PREMISE> p2 </PREMISE> ...``
            Assistant message: conclusion / proof trace from ``content``.

        **Stage 2 (benchmark CoT format):**
            User message:  full problem context from ``question`` field.
            Assistant message: semi-formal reasoning from ``content``.

        The Mistral Small 3.x chat format (V7-Tekken) is:
            ``<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user_message}[/INST]{assistant_response}</s>``

        Returns:
            Tuple of (full_text, assistant_response) where assistant_response is
            the portion we want to compute loss on.
        """
        # --- Stage 2 format: has "question" field, no structured premises ---
        if "question" in example and "premises" not in example:
            user_content = example["question"]
            assistant_content = str(example.get("content", ""))
            if not user_content or not assistant_content:
                return "", ""

            messages = [
                SystemMessage(content=self.config.system_prompt),
                UserMessage(content=user_content),
            ]
            request = ChatCompletionRequest(messages=messages)
            encoded = self.mistral_tokenizer.encode_chat_completion(request)
            full_text = encoded.text + assistant_content + "</s>"
            return full_text, assistant_content

        # --- Stage 0/1 format: structured premises + content/conclusion ---
        premises, conclusion = self._extract_premises_conclusion(example)

        if not premises or not conclusion:
            return "", ""

        user_content = self._format_user_message(premises)
        assistant_content = self._format_assistant_message(conclusion)

        messages = [
            SystemMessage(content=self.config.system_prompt),
            UserMessage(content=user_content),
        ]

        request = ChatCompletionRequest(messages=messages)
        encoded = self.mistral_tokenizer.encode_chat_completion(request)

        full_text = encoded.text + assistant_content + "</s>"

        return full_text, assistant_content

    def load_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load train, validation, and test data."""
        train_data, val_data, test_data = [], [], []

        # Load from explicit paths if provided
        if self.config.train_path and Path(self.config.train_path).exists():
            print(f"Loading training data from {self.config.train_path}")
            if self.config.train_path.endswith(".jsonl"):
                train_data = self.load_jsonl(self.config.train_path)
            else:
                train_data = self.load_json(self.config.train_path)

        if self.config.val_path and Path(self.config.val_path).exists():
            print(f"Loading validation data from {self.config.val_path}")
            if self.config.val_path.endswith(".jsonl"):
                val_data = self.load_jsonl(self.config.val_path)
            else:
                val_data = self.load_json(self.config.val_path)

        if self.config.test_path and Path(self.config.test_path).exists():
            print(f"Loading test data from {self.config.test_path}")
            if self.config.test_path.endswith(".jsonl"):
                test_data = self.load_jsonl(self.config.test_path)
            else:
                test_data = self.load_json(self.config.test_path)

        # Load from HuggingFace dataset if specified
        if self.config.dataset_name:
            print(f"Loading dataset from HuggingFace: {self.config.dataset_name}")
            hf_data = self._load_hf_dataset()
            train_data.extend(hf_data.get("train", []))
            val_data.extend(hf_data.get("val", []))
            test_data.extend(hf_data.get("test", []))

        # If we have data but no splits, create them
        if train_data and not val_data and not test_data:
            print("Creating train/val/test splits...")
            train_data, val_data, test_data = self._split_data(train_data)

        # If only train and no val, split train
        if train_data and not val_data:
            split_idx = int(len(train_data) * 0.9)
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]

        print(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data

    def _load_hf_dataset(self) -> Dict[str, List[Dict]]:
        """Load data from HuggingFace datasets."""
        result = {"train": [], "val": [], "test": []}

        try:
            dataset = load_dataset(self.config.dataset_name)

            for split in ["train", "validation", "test"]:
                if split in dataset:
                    target_key = "val" if split == "validation" else split
                    for example in dataset[split]:
                        result[target_key].append(dict(example))
        except Exception as e:
            print(f"Warning: Could not load HuggingFace dataset: {e}")

        return result

    def _split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test."""
        np.random.seed(self.config.seed)
        indices = np.random.permutation(len(data))

        train_end = int(len(data) * self.config.train_split)
        val_end = train_end + int(len(data) * self.config.val_split)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        return train_data, val_data, test_data

    def create_datasets(self) -> DatasetDict:
        """Create HuggingFace datasets from loaded data."""
        train_data, val_data, test_data = self.load_data()

        def process_examples(examples: List[Dict]) -> Dict[str, List]:
            texts = []
            assistant_responses = []
            skipped = 0

            for ex in examples:
                full_text, assistant_response = self.format_example(ex)
                # Filter out malformed examples
                if full_text and assistant_response:
                    texts.append(full_text)
                    assistant_responses.append(assistant_response)
                else:
                    skipped += 1

            if skipped > 0:
                print(f"  Skipped {skipped} malformed examples")

            return {"text": texts, "assistant_response": assistant_responses}

        datasets = {}

        if train_data:
            print(f"Processing {len(train_data)} training examples...")
            train_processed = process_examples(train_data)
            datasets["train"] = Dataset.from_dict(train_processed)
            print(f"  Valid training examples: {len(datasets['train'])}")

        if val_data:
            print(f"Processing {len(val_data)} validation examples...")
            val_processed = process_examples(val_data)
            datasets["validation"] = Dataset.from_dict(val_processed)
            print(f"  Valid validation examples: {len(datasets['validation'])}")

        if test_data:
            print(f"Processing {len(test_data)} test examples...")
            test_processed = process_examples(test_data)
            datasets["test"] = Dataset.from_dict(test_processed)
            print(f"  Valid test examples: {len(datasets['test'])}")

        return DatasetDict(datasets)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize a dataset with conclusion-only loss masking for Mistral chat format.

        The Mistral Small 3.x chat format (V7-Tekken) is:
            <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user_message}[/INST]{assistant_response}</s>

        When conclusion_only_loss is True, we mask all tokens EXCEPT the assistant
        response (everything after [/INST]), so the model only learns to generate
        the conclusion given the premises.
        """
        use_conclusion_only = self.config.conclusion_only_loss
        # The marker that indicates end of prompt and start of assistant response
        assistant_start_marker = "[/INST]"

        def tokenize_function(examples):
            # Tokenize full text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors=None,  # Return lists for easier manipulation
            )

            labels = []
            for i, text in enumerate(examples["text"]):
                input_ids = tokenized["input_ids"][i]

                if use_conclusion_only and assistant_start_marker in text:
                    # Find where [/INST] ends - everything after is the assistant response
                    marker_end = text.find(assistant_start_marker) + len(assistant_start_marker)

                    # Tokenize the prefix (everything up to and including [/INST])
                    prefix_text = text[:marker_end]
                    prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                    prefix_len = min(len(prefix_tokens), len(input_ids))

                    # Create labels: -100 for prompt tokens, actual ids for assistant response
                    label = list(input_ids)

                    # Mask the entire prompt (everything before assistant response)
                    for j in range(prefix_len):
                        label[j] = -100

                    # Mask padding tokens
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is not None:
                        for j in range(len(label)):
                            if input_ids[j] == pad_token_id:
                                label[j] = -100

                    labels.append(label)
                else:
                    # No marker found or conclusion_only disabled
                    # Use standard causal LM training (predict all tokens)
                    label = list(input_ids)
                    # Still mask padding
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is not None:
                        for j in range(len(label)):
                            if input_ids[j] == pad_token_id:
                                label[j] = -100
                    labels.append(label)

            tokenized["labels"] = labels
            return tokenized

        return dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizing (assistant-only loss)" if use_conclusion_only else "Tokenizing",
        )


# =============================================================================
# Training
# =============================================================================

class LoggingCallback(TrainerCallback):
    """Custom callback for additional logging to console and wandb."""

    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb and WANDB_AVAILABLE

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            step = state.global_step
            wandb_logs = {"step": step}

            if "loss" in logs:
                print(f"Step {step}: train_loss={logs['loss']:.4f}")
                wandb_logs["train/loss"] = logs["loss"]

            if "eval_loss" in logs:
                print(f"Step {step}: eval_loss={logs['eval_loss']:.4f}")
                wandb_logs["eval/loss"] = logs["eval_loss"]

            if "learning_rate" in logs:
                wandb_logs["train/learning_rate"] = logs["learning_rate"]

            if "epoch" in logs:
                wandb_logs["train/epoch"] = logs["epoch"]

            # Log to wandb if enabled
            if self.use_wandb and len(wandb_logs) > 1:
                wandb.log(wandb_logs, step=step)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Log evaluation metrics to wandb."""
        if metrics and self.use_wandb:
            step = state.global_step
            eval_metrics = {f"eval/{k.replace('eval_', '')}": v for k, v in metrics.items()}
            eval_metrics["step"] = step
            wandb.log(eval_metrics, step=step)
            print(f"Step {step}: Evaluation metrics logged to wandb")


def setup_model(
    model_key: str,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
) -> Tuple[Any, Any]:
    """Setup model and tokenizer with LoRA."""
    model_info = get_model_info(model_key)
    model_name = model_info["name"]

    print(f"\n=== Loading Model: {model_name} ===")
    print(f"Description: {model_info['description']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=train_config.cache_dir,
        token=train_config.hf_token,
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine quantization config
    quantization_config = None
    torch_dtype = torch.float16 if train_config.fp16 else (torch.bfloat16 if train_config.bf16 else torch.float32)

    if train_config.use_4bit and BNBS_AVAILABLE:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Using 4-bit quantization")
    elif train_config.use_8bit and BNBS_AVAILABLE:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization")

    # Load model - use special class if specified in registry
    model_class_name = model_info.get("model_class")
    if model_class_name == "Mistral3ForConditionalGeneration":
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
        print(f"Using special model class: {model_class_name}")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=train_config.cache_dir,
            token=train_config.hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=train_config.cache_dir,
            token=train_config.hf_token,
        )

    # Prepare for k-bit training if using quantization
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Get target modules
    target_modules = lora_config.target_modules or model_info.get("target_modules", ["q_proj", "v_proj"])

    # Setup LoRA — either from a prior adapter (stage continuation) or fresh
    print(f"\n=== Setting up LoRA ===")
    if train_config.init_adapter_path:
        adapter_path = train_config.init_adapter_path
        print(f"Loading prior LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        model.print_trainable_parameters()
    else:
        print(f"Rank: {lora_config.rank}, Alpha: {lora_config.alpha}")
        print(f"Target modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=target_modules,
            bias=lora_config.bias,
            modules_to_save=None,
        )
        print(f"Dropout: {lora_config.dropout}, Bias: {lora_config.bias}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return model, tokenizer


def train(
    model_key: str,
    data_config: DataConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
):
    """Main training function."""
    # Set seed
    set_seed(train_config.seed)

    # Setup model and tokenizer
    model, tokenizer = setup_model(model_key, lora_config, train_config)

    # Update data config with model's max length
    model_info = get_model_info(model_key)
    data_config.max_length = min(data_config.max_length, model_info.get("max_length", 2048))

    # Load and prepare data
    print("\n=== Loading Data ===")
    if data_config.conclusion_only_loss:
        print("Training mode: ASSISTANT-ONLY (loss computed only on assistant response)")
        print(f"  Using Mistral chat template from: {model_info['name']}")
        print(f"  Format: <PREMISE>...</PREMISE> -> <CONCLUSION>...</CONCLUSION>")
    else:
        print("Training mode: FULL TEXT (loss computed on all tokens)")
    data_loader = DataLoader(data_config, tokenizer, model_info["name"],
                             cache_dir=train_config.cache_dir,
                             hf_token=train_config.hf_token)
    datasets = data_loader.create_datasets()

    # Tokenize datasets
    print("\n=== Tokenizing Datasets ===")
    tokenized_datasets = DatasetDict({
        split: data_loader.tokenize_dataset(ds)
        for split, ds in datasets.items()
    })

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_key.replace("/", "_").replace("-", "_")
    output_dir = Path(train_config.output_dir) / f"{model_short_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if train_config.use_wandb and WANDB_AVAILABLE:
        run_name = train_config.wandb_run_name or f"{model_short_name}_{timestamp}"
        wandb.init(
            project=train_config.wandb_project,
            name=run_name,
            config={
                "model": model_key,
                "lora_rank": lora_config.rank,
                "lora_alpha": lora_config.alpha,
                "learning_rate": train_config.learning_rate,
                "batch_size": train_config.batch_size,
                "epochs": train_config.num_epochs,
            }
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio if train_config.warmup_steps == 0 else 0,
        warmup_steps=train_config.warmup_steps,
        max_grad_norm=train_config.max_grad_norm,
        lr_scheduler_type=train_config.lr_scheduler_type,
        fp16=train_config.fp16 and not train_config.bf16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        eval_strategy="steps" if "validation" in tokenized_datasets else "no",
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=train_config.load_best_model_at_end and "validation" in tokenized_datasets,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        report_to="wandb" if train_config.use_wandb and WANDB_AVAILABLE else "none",
        seed=train_config.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Callbacks
    callbacks = [LoggingCallback(use_wandb=train_config.use_wandb)]
    if train_config.early_stopping_patience > 0 and "validation" in tokenized_datasets:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_config.early_stopping_patience
            )
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train
    print("\n=== Starting Training ===")
    print(f"Output directory: {output_dir}")

    resume_from = train_config.resume_from_checkpoint
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)

    print(f"\n=== Saving Final Model to {final_dir} ===")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training config
    config_save = {
        "model_key": model_key,
        "model_name": get_model_info(model_key)["name"],
        "lora_config": {
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "dropout": lora_config.dropout,
            "target_modules": lora_config.target_modules or get_model_info(model_key).get("target_modules"),
        },
        "train_config": {
            "learning_rate": train_config.learning_rate,
            "batch_size": train_config.batch_size,
            "epochs": train_config.num_epochs,
            "fp16": train_config.fp16,
            "bf16": train_config.bf16,
        },
        "data_config": {
            "train_path": data_config.train_path,
            "val_path": data_config.val_path,
            "max_length": data_config.max_length,
        },
        "timestamp": timestamp,
    }

    with open(final_dir / "training_config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    # Evaluate on test set if available
    if "test" in tokenized_datasets:
        print("\n=== Evaluating on Test Set ===")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        print(f"Test results: {test_results}")

        with open(final_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    # Close wandb
    if train_config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_dir}")
    return str(final_dir)


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning Script with Model Switching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python lora_finetune.py --list-models

  # Fine-tune Mistral for logical inference (default: loss only on assistant response)
  python lora_finetune.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --train-path ./data/logic_train.jsonl

  # Fine-tune with custom LoRA settings
  python lora_finetune.py --model mistral-7b -r 32 -a 64 --lora-dropout 0.05

  # Train on all tokens (not just conclusions)
  python lora_finetune.py --train-path ./data/train.jsonl --train-on-all

  # Use 4-bit quantization for memory efficiency
  python lora_finetune.py --use-4bit --batch-size 2

  # Use config file
  python lora_finetune.py --config ./configs/finetune_config.yaml

Training Mode:
  Uses Mistral chat template for proper instruction format.

  Input format (from generate_logic_data.py):
    {"premises": [{"id": "p1", "text": "..."}, ...], "content": "..."}

  Converted to Mistral chat format:
    User: <PREMISE> premise1 </PREMISE> <PREMISE> premise2 </PREMISE> ...
    Assistant: <CONCLUSION> conclusion </CONCLUSION>

  By default, loss is computed only on the assistant response (conclusion).
  Use --train-on-all to train on the full text instead.
        """
    )

    # Model selection
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                       help="Model key from registry or HuggingFace model path")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")

    # Data paths
    parser.add_argument("--train-path", type=str, help="Path to training data (JSONL or JSON)")
    parser.add_argument("--val-path", type=str, help="Path to validation data")
    parser.add_argument("--test-path", type=str, help="Path to test data")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")

    # LoRA configuration
    parser.add_argument("--lora-rank", "-r", type=int, default=16,
                       help="LoRA rank (higher = more capacity, default: 16)")
    parser.add_argument("--lora-alpha", "-a", type=int, default=32,
                       help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout probability (default: 0.1)")
    parser.add_argument("--target-modules", type=str, nargs="+",
                       help="Target modules for LoRA (space-separated). If not specified, uses model defaults.")
    parser.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"],
                       help="LoRA bias training mode (default: none)")

    # Inference training configuration
    parser.add_argument("--conclusion-only-loss", action="store_true", default=True,
                       help="Only compute loss on assistant response (default: True)")
    parser.add_argument("--train-on-all", action="store_true",
                       help="Train on all tokens, not just assistant response (disables conclusion-only-loss)")
    parser.add_argument("--system-prompt", type=str,
                       default="You are a logical reasoning assistant. Given the following premises, derive their valid conclusion.",
                       help="System prompt for logical reasoning training")

    # Training configuration
    parser.add_argument("--output-dir", type=str, default="/tmp/lora_finetuned",
                       help="Output directory for saved models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # Precision and memory
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training (overrides FP16)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                       help="Disable gradient checkpointing")

    # Logging and saving
    parser.add_argument("--logging-steps", type=int, default=1, help="Log train loss every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--eval-steps", type=int, default=10, help="Evaluate val loss every N steps (must divide save-steps)")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="syllm-lora-finetune",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")

    # Other
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--init-adapter", type=str,
        help="Path to a prior LoRA adapter directory to use as initialization (stage 0->1 continuation)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--cache-dir", type=str,
                       help="Directory for caching downloaded models (overrides HF_HOME/TRANSFORMERS_CACHE)")
    parser.add_argument("--hf-token", type=str,
                       help="HuggingFace API token for gated models (or set HF_TOKEN env var)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List models and exit
    if args.list_models:
        list_available_models()
        return

    # Resolve HF token: CLI arg > HF_TOKEN env var > HUGGING_FACE_HUB_TOKEN env var
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Redirect model cache early so all HuggingFace downloads go to the right place
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Load config file if provided
    file_config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            file_config = yaml.safe_load(f)

    # Build configurations (CLI args override file config)
    data_config = DataConfig(
        train_path=args.train_path or file_config.get("data", {}).get("train_path"),
        val_path=args.val_path or file_config.get("data", {}).get("val_path"),
        test_path=args.test_path or file_config.get("data", {}).get("test_path"),
        dataset_name=args.dataset or file_config.get("data", {}).get("dataset"),
        max_length=args.max_length,
        seed=args.seed,
        # Inference-focused training settings
        conclusion_only_loss=not args.train_on_all,
        system_prompt=args.system_prompt,
    )

    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias=args.lora_bias,
    )

    train_config = TrainConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume,
        init_adapter_path=args.init_adapter,
        seed=args.seed,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
    )

    # Validate we have data
    if not data_config.train_path and not data_config.dataset_name:
        print("Error: Must provide --train-path or --dataset")
        print("Run with --help for usage information")
        sys.exit(1)

    # Get model key
    model_key = args.model or file_config.get("model", {}).get("name", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

    # Run training
    output_path = train(
        model_key=model_key,
        data_config=data_config,
        lora_config=lora_config,
        train_config=train_config,
    )

    print(f"\nTraining complete! Model saved to: {output_path}")


if __name__ == "__main__":
    main()
