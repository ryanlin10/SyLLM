"""Fine-tuning script for DeepSeek V3 with structured output enforcement."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm

from ..data.schema import Annotation, safe_parse_model_output, format_prompt
from ..data.curation import DataCurator


def format_target(premises: List[Dict[str, Any]], conclusion: str) -> str:
    """Format target JSON output."""
    premise_texts = [p.get("text", p) if isinstance(p, dict) else p for p in premises]
    target = {
        "premises": premise_texts,
        "content": conclusion
    }
    return json.dumps(target, ensure_ascii=False)


class StructuredOutputDataset:
    """Dataset for structured output fine-tuning."""
    
    def __init__(
        self,
        tokenizer,
        annotations: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        
        # Process annotations
        self.examples = []
        for ann in annotations:
            context = ann.get("context", "")
            premises = ann.get("premises", [])
            
            conclusion_data = ann.get("content") or ann.get("conclusion", {})
            if isinstance(conclusion_data, dict):
                conclusion = conclusion_data.get("text", "")
            else:
                conclusion = conclusion_data
            
            prompt = format_prompt(context, system_prompt=self.system_prompt)
            target = format_target(premises, conclusion)
            
            full_text = prompt + target
            self.examples.append({
                "text": full_text,
                "prompt": prompt,
                "target": target
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def tokenize(self, examples):
        """Tokenize examples."""
        texts = examples["text"]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask prompt tokens)
        labels = tokenized["input_ids"].clone()
        prompt_lengths = []
        
        for i, prompt in enumerate(examples["prompt"]):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            prompt_lengths.append(prompt_len)
            # Mask prompt tokens in labels
            labels[i, :prompt_len] = -100
        
        tokenized["labels"] = labels
        return tokenized


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable support."""
    from ..utils.config_loader import load_config as load_config_with_env
    return load_config_with_env(config_path)


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model and tokenizer with LoRA."""
    model_config = config["model"]
    base_model = model_config["base_model"]
    
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    if model_config.get("use_lora", True):
        print("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config.get("lora_rank", 16),
            lora_alpha=model_config.get("lora_alpha", 32),
            lora_dropout=model_config.get("lora_dropout", 0.1),
            target_modules=model_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def train(config_path: str = "./config.yaml"):
    """Main training function."""
    config = load_config(config_path)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load data
    print("Loading training data...")
    curator = DataCurator()
    train_data = curator.load_jsonl(config["data"]["train_path"])
    val_data = curator.load_jsonl(config["data"]["val_path"])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = StructuredOutputDataset(
        tokenizer,
        train_data,
        system_prompt=config.get("prompts", {}).get("system_prompt")
    )
    val_dataset = StructuredOutputDataset(
        tokenizer,
        val_data,
        system_prompt=config.get("prompts", {}).get("system_prompt")
    )
    
    # Tokenize
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.tokenize({
        "text": [ex["text"] for ex in train_dataset.examples],
        "prompt": [ex["prompt"] for ex in train_dataset.examples]
    })
    val_tokenized = val_dataset.tokenize({
        "text": [ex["text"] for ex in val_dataset.examples],
        "prompt": [ex["prompt"] for ex in val_dataset.examples]
    })
    
    # Create HuggingFace Dataset
    train_hf = Dataset.from_dict(train_tokenized)
    val_hf = Dataset.from_dict(val_tokenized)
    
    # Training arguments
    train_config = config["training"]
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["num_train_epochs"],
        warmup_steps=train_config["warmup_steps"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        evaluation_strategy=train_config["evaluation_strategy"],
        eval_steps=train_config["eval_steps"],
        save_total_limit=train_config["save_total_limit"],
        fp16=train_config.get("fp16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        report_to=train_config.get("report_to", "none"),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_output_dir = Path(train_config["output_dir"]) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved final model to {final_output_dir}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.yaml"
    train(config_path)

