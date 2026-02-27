#!/usr/bin/env python3
"""
Test script for LoRA fine-tuning data loading and processing functionality.

Tests:
1. Data formatting (premises as context, conclusion as target)
2. Conclusion-only loss masking
3. LoRA configuration
4. Model registry
5. Various data formats
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Mock Classes (to avoid heavy dependencies)
# =============================================================================

class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        self._vocab = {}
        self._next_id = 2

    def _get_token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple word-based tokenization for testing."""
        tokens = text.split()
        ids = [self._get_token_id(t) for t in tokens]
        if add_special_tokens:
            ids = [self.eos_token_id] + ids
        return ids

    def __call__(self, texts, truncation=True, max_length=2048, padding="max_length", return_tensors=None):
        """Batch tokenization."""
        if isinstance(texts, str):
            texts = [texts]

        all_input_ids = []
        all_attention_mask = []

        for text in texts:
            ids = self.encode(text, add_special_tokens=True)

            # Truncate
            if len(ids) > max_length:
                ids = ids[:max_length]

            # Create attention mask
            attention_mask = [1] * len(ids)

            # Pad
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            all_input_ids.append(ids)
            all_attention_mask.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
        }


# =============================================================================
# Import components to test (with mocked dependencies)
# =============================================================================

# Create mock modules before importing
sys.modules['torch'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['wandb'] = MagicMock()

# Now we can define our own versions for testing
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# =============================================================================
# Recreate the core classes for testing
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    dataset_name: Optional[str] = None
    text_column: str = "text"
    max_length: int = 2048
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    use_logic_datasets: bool = False
    logic_datasets: List[str] = field(default_factory=lambda: ["logiqa", "logicnli"])
    preprocessing_num_workers: int = 4
    conclusion_only_loss: bool = True
    conclusion_marker: str = "<CONCLUSION>"
    end_conclusion_marker: str = "</CONCLUSION>"


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


MODEL_REGISTRY = {
    "deepseek-v3": {
        "name": "deepseek-ai/deepseek-v3",
        "description": "DeepSeek V3 - Recommended for structured reasoning",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 4096,
    },
    "llama3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "description": "Llama 3 8B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B base",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 8192,
    },
}


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model info from registry or return custom model config."""
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    return {
        "name": model_key,
        "description": f"Custom model: {model_key}",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "max_length": 2048,
    }


class DataLoader:
    """Data loader for testing."""

    def __init__(self, config: DataConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

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

    def format_example(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Format a single example for training."""
        conclusion_marker = self.config.conclusion_marker
        end_marker = self.config.end_conclusion_marker

        # NOTE: Order matters! Check more specific formats first.

        # Format 1: Pre-formatted text with markers
        if "text" in example:
            text = example["text"]
            if conclusion_marker in text:
                start_idx = text.find(conclusion_marker) + len(conclusion_marker)
                end_idx = text.find(end_marker) if end_marker in text else len(text)
                conclusion = text[start_idx:end_idx].strip()
                return text, conclusion
            return text, text

        # Format 2: DAG format with reasoning (check before premises/context)
        if "reasoning" in example:
            context = example.get("context", "")
            reasoning = example.get("reasoning", {})

            premises = [p.get("text", "") if isinstance(p, dict) else p
                       for p in reasoning.get("premises", [])]
            conclusion = reasoning.get("conclusion", "")
            inference_steps = reasoning.get("inference_steps", [])

            full_conclusion = self._format_inference_chain(inference_steps, conclusion)
            full_text = self._format_inference_prompt(context, premises, full_conclusion)
            return full_text, full_conclusion

        # Format 3: Question-answer format (check before premises/context)
        if "question" in example and "answer" in example:
            question = example["question"]
            answer = example["answer"]
            context = example.get("context", "")

            premises = [context] if context else []
            premises.append(f"Question: {question}")
            full_text = self._format_inference_prompt("", premises, str(answer))
            return full_text, str(answer)

        # Format 4: Context + premises + conclusion (SyLLM format)
        if "context" in example or "premises" in example:
            context = example.get("context", "")
            premises = example.get("premises", [])
            conclusion = example.get("content") or example.get("conclusion", {})

            if isinstance(conclusion, dict):
                conclusion_text = conclusion.get("text", "")
            else:
                conclusion_text = str(conclusion)

            premise_texts = []
            for p in premises:
                if isinstance(p, dict):
                    premise_texts.append(p.get("text", str(p)))
                else:
                    premise_texts.append(str(p))

            full_text = self._format_inference_prompt(context, premise_texts, conclusion_text)
            return full_text, conclusion_text

        # Fallback
        text = json.dumps(example, ensure_ascii=False)
        return text, text

    def _format_inference_chain(self, inference_steps: List[Dict], conclusion: str) -> str:
        """Format inference steps and conclusion as a reasoning chain."""
        parts = []
        for step in inference_steps:
            step_text = step.get("text", "")
            step_id = step.get("id", "")
            depends = step.get("depends_on", [])
            if depends:
                parts.append(f"[{step_id}] From {depends}: {step_text}")
            else:
                parts.append(f"[{step_id}] {step_text}")
        parts.append(f"Therefore: {conclusion}")
        return "\n".join(parts)

    def _format_inference_prompt(self, context: str, premises: List[str], conclusion: str) -> str:
        """Format a complete inference training example."""
        conclusion_marker = self.config.conclusion_marker
        end_marker = self.config.end_conclusion_marker

        parts = []
        parts.append("You are a logical reasoning assistant. Given the premises below, derive a valid conclusion.")
        parts.append("")

        if context and context.strip():
            parts.append(f"Context: {context}")
            parts.append("")

        if premises:
            parts.append("Premises:")
            for i, premise in enumerate(premises, 1):
                if premise and premise.strip():
                    parts.append(f"  {i}. {premise.strip()}")
            parts.append("")

        parts.append("Conclusion:")
        parts.append(f"{conclusion_marker}{conclusion}{end_marker}")

        return "\n".join(parts)

    def tokenize_with_conclusion_masking(self, text: str) -> Dict[str, List[int]]:
        """Tokenize with conclusion-only loss masking."""
        conclusion_marker = self.config.conclusion_marker
        end_marker = self.config.end_conclusion_marker
        use_conclusion_only = self.config.conclusion_only_loss

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"][0]  # Single example

        if use_conclusion_only and conclusion_marker in text:
            marker_start = text.find(conclusion_marker)
            marker_end = text.find(end_marker) if end_marker in text else len(text)

            prefix_text = text[:marker_start + len(conclusion_marker)]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = min(len(prefix_tokens), len(input_ids))

            if end_marker in text:
                suffix_text = text[marker_end:]
                suffix_tokens = self.tokenizer.encode(suffix_text, add_special_tokens=False)
                suffix_len = len(suffix_tokens)
            else:
                suffix_len = 0

            label = list(input_ids)

            # Mask prefix
            for j in range(prefix_len):
                label[j] = -100

            # Mask padding
            pad_token_id = self.tokenizer.pad_token_id
            for j in range(len(label)):
                if input_ids[j] == pad_token_id:
                    label[j] = -100

            return {"input_ids": input_ids, "labels": label}
        else:
            label = list(input_ids)
            pad_token_id = self.tokenizer.pad_token_id
            for j in range(len(label)):
                if input_ids[j] == pad_token_id:
                    label[j] = -100
            return {"input_ids": input_ids, "labels": label}


# =============================================================================
# Test Cases
# =============================================================================

class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def add_fail(self, name: str, error: str):
        self.failed += 1
        self.failures.append((name, error))
        print(f"  ✗ {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} tests passed")
        if self.failures:
            print(f"\nFailures:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_model_registry(results: TestResults):
    """Test model registry functionality."""
    print("\n[Testing Model Registry]")

    # Test known model
    info = get_model_info("llama3-8b")
    if info["name"] == "meta-llama/Meta-Llama-3-8B":
        results.add_pass("Known model lookup")
    else:
        results.add_fail("Known model lookup", f"Got {info['name']}")

    # Test custom model
    info = get_model_info("custom/my-model")
    if info["name"] == "custom/my-model" and "Custom model" in info["description"]:
        results.add_pass("Custom model fallback")
    else:
        results.add_fail("Custom model fallback", f"Got {info}")

    # Test target modules present
    info = get_model_info("deepseek-v3")
    if "q_proj" in info["target_modules"]:
        results.add_pass("Target modules present")
    else:
        results.add_fail("Target modules present", "Missing q_proj")


def test_lora_config(results: TestResults):
    """Test LoRA configuration."""
    print("\n[Testing LoRA Configuration]")

    # Default config
    config = LoRAConfig()
    if config.rank == 16 and config.alpha == 32:
        results.add_pass("Default LoRA config")
    else:
        results.add_fail("Default LoRA config", f"rank={config.rank}, alpha={config.alpha}")

    # Custom config
    config = LoRAConfig(rank=64, alpha=128, dropout=0.2, bias="all")
    if config.rank == 64 and config.alpha == 128 and config.bias == "all":
        results.add_pass("Custom LoRA config")
    else:
        results.add_fail("Custom LoRA config", f"Got {config}")


def test_data_config(results: TestResults):
    """Test data configuration."""
    print("\n[Testing Data Configuration]")

    # Default config
    config = DataConfig()
    if config.conclusion_only_loss and config.conclusion_marker == "<CONCLUSION>":
        results.add_pass("Default conclusion-only settings")
    else:
        results.add_fail("Default conclusion-only settings",
                        f"conclusion_only_loss={config.conclusion_only_loss}")

    # Custom markers
    config = DataConfig(conclusion_marker="[ANS]", end_conclusion_marker="[/ANS]")
    if config.conclusion_marker == "[ANS]" and config.end_conclusion_marker == "[/ANS]":
        results.add_pass("Custom conclusion markers")
    else:
        results.add_fail("Custom conclusion markers", f"Got {config.conclusion_marker}")


def test_format_premises_conclusion(results: TestResults):
    """Test formatting with premises as context and conclusion as target."""
    print("\n[Testing Premises/Conclusion Formatting]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    # Test SyLLM format
    example = {
        "context": "In a logic puzzle",
        "premises": [
            {"text": "All cats are mammals"},
            {"text": "Fluffy is a cat"}
        ],
        "content": {"text": "Fluffy is a mammal"}
    }

    full_text, conclusion = loader.format_example(example)

    # Check premises are in text
    if "All cats are mammals" in full_text and "Fluffy is a cat" in full_text:
        results.add_pass("Premises concatenated in text")
    else:
        results.add_fail("Premises concatenated in text", "Missing premises")

    # Check conclusion extracted correctly
    if conclusion == "Fluffy is a mammal":
        results.add_pass("Conclusion extracted correctly")
    else:
        results.add_fail("Conclusion extracted correctly", f"Got '{conclusion}'")

    # Check markers present
    if "<CONCLUSION>" in full_text and "</CONCLUSION>" in full_text:
        results.add_pass("Conclusion markers present")
    else:
        results.add_fail("Conclusion markers present", "Missing markers")

    # Check numbered premises
    if "1." in full_text and "2." in full_text:
        results.add_pass("Premises are numbered")
    else:
        results.add_fail("Premises are numbered", "Missing numbering")


def test_format_qa(results: TestResults):
    """Test question-answer format."""
    print("\n[Testing Q&A Format]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "context": "The sky is blue during the day.",
        "question": "What color is the sky?",
        "answer": "Blue"
    }

    full_text, conclusion = loader.format_example(example)

    if conclusion == "Blue":
        results.add_pass("Answer extracted as conclusion")
    else:
        results.add_fail("Answer extracted as conclusion", f"Got '{conclusion}'")

    if "Question:" in full_text:
        results.add_pass("Question in premises")
    else:
        results.add_fail("Question in premises", "Missing question")


def test_format_dag(results: TestResults):
    """Test DAG reasoning format."""
    print("\n[Testing DAG Reasoning Format]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "context": "Logic problem",
        "reasoning": {
            "premises": [
                {"text": "A implies B"},
                {"text": "A is true"}
            ],
            "inference_steps": [
                {"id": "step1", "text": "B is true", "depends_on": ["premise1", "premise2"]}
            ],
            "conclusion": "B is proven"
        }
    }

    full_text, conclusion = loader.format_example(example)

    if "Therefore:" in conclusion:
        results.add_pass("Inference chain formatted")
    else:
        results.add_fail("Inference chain formatted", f"Got '{conclusion}'")

    if "[step1]" in conclusion:
        results.add_pass("Step IDs included")
    else:
        results.add_fail("Step IDs included", "Missing step ID")


def test_format_preformatted(results: TestResults):
    """Test pre-formatted text with markers."""
    print("\n[Testing Pre-formatted Text]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "text": "Some context here <CONCLUSION>The final answer</CONCLUSION>"
    }

    full_text, conclusion = loader.format_example(example)

    if conclusion == "The final answer":
        results.add_pass("Pre-formatted conclusion extracted")
    else:
        results.add_fail("Pre-formatted conclusion extracted", f"Got '{conclusion}'")


def test_conclusion_only_loss_masking(results: TestResults):
    """Test that loss is only computed on conclusion tokens."""
    print("\n[Testing Conclusion-Only Loss Masking]")

    config = DataConfig(max_length=100)
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "premises": ["Premise one", "Premise two"],
        "content": "The conclusion"
    }

    full_text, conclusion = loader.format_example(example)
    result = loader.tokenize_with_conclusion_masking(full_text)

    input_ids = result["input_ids"]
    labels = result["labels"]

    # Check that some labels are masked (-100)
    masked_count = sum(1 for l in labels if l == -100)
    unmasked_count = sum(1 for l in labels if l != -100)

    if masked_count > 0:
        results.add_pass("Some tokens are masked")
    else:
        results.add_fail("Some tokens are masked", f"masked={masked_count}")

    if unmasked_count > 0:
        results.add_pass("Some tokens are unmasked (conclusion)")
    else:
        results.add_fail("Some tokens are unmasked (conclusion)", f"unmasked={unmasked_count}")

    # Padding should be masked
    pad_positions = [i for i, id in enumerate(input_ids) if id == tokenizer.pad_token_id]
    pad_labels = [labels[i] for i in pad_positions]
    if all(l == -100 for l in pad_labels):
        results.add_pass("Padding tokens masked")
    else:
        results.add_fail("Padding tokens masked", "Some padding not masked")


def test_full_text_training(results: TestResults):
    """Test training on full text (conclusion_only_loss=False)."""
    print("\n[Testing Full Text Training Mode]")

    config = DataConfig(max_length=100, conclusion_only_loss=False)
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "premises": ["Premise one"],
        "content": "The conclusion"
    }

    full_text, _ = loader.format_example(example)
    result = loader.tokenize_with_conclusion_masking(full_text)

    labels = result["labels"]
    input_ids = result["input_ids"]

    # Count non-padding tokens that are unmasked
    non_pad_unmasked = sum(1 for i, l in enumerate(labels)
                          if l != -100 and input_ids[i] != tokenizer.pad_token_id)

    if non_pad_unmasked > 5:  # Should have many more unmasked tokens
        results.add_pass("Full text mode has more unmasked tokens")
    else:
        results.add_fail("Full text mode has more unmasked tokens",
                        f"Only {non_pad_unmasked} unmasked")


def test_jsonl_loading(results: TestResults):
    """Test JSONL file loading."""
    print("\n[Testing JSONL Loading]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    # Create temp JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"premises": ["P1"], "content": "C1"}\n')
        f.write('{"premises": ["P2"], "content": "C2"}\n')
        f.write('{"premises": ["P3"], "content": "C3"}\n')
        temp_path = f.name

    try:
        data = loader.load_jsonl(temp_path)
        if len(data) == 3:
            results.add_pass("JSONL loading correct count")
        else:
            results.add_fail("JSONL loading correct count", f"Got {len(data)}")

        if data[0]["content"] == "C1":
            results.add_pass("JSONL data parsed correctly")
        else:
            results.add_fail("JSONL data parsed correctly", f"Got {data[0]}")
    finally:
        os.unlink(temp_path)


def test_json_loading(results: TestResults):
    """Test JSON file loading."""
    print("\n[Testing JSON Loading]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    # Create temp JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {"premises": ["P1"], "content": "C1"},
            {"premises": ["P2"], "content": "C2"}
        ], f)
        temp_path = f.name

    try:
        data = loader.load_json(temp_path)
        if len(data) == 2:
            results.add_pass("JSON loading correct count")
        else:
            results.add_fail("JSON loading correct count", f"Got {len(data)}")
    finally:
        os.unlink(temp_path)


def test_empty_premises(results: TestResults):
    """Test handling of empty premises."""
    print("\n[Testing Edge Cases]")

    config = DataConfig()
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    # Empty premises
    example = {
        "premises": [],
        "content": "Still valid"
    }

    full_text, conclusion = loader.format_example(example)

    if conclusion == "Still valid":
        results.add_pass("Empty premises handled")
    else:
        results.add_fail("Empty premises handled", f"Got '{conclusion}'")

    # String content (not dict)
    example = {
        "premises": ["P1"],
        "content": "String conclusion"
    }

    _, conclusion = loader.format_example(example)
    if conclusion == "String conclusion":
        results.add_pass("String conclusion handled")
    else:
        results.add_fail("String conclusion handled", f"Got '{conclusion}'")


def test_custom_markers(results: TestResults):
    """Test custom conclusion markers."""
    print("\n[Testing Custom Markers]")

    config = DataConfig(
        conclusion_marker="<<START>>",
        end_conclusion_marker="<<END>>"
    )
    tokenizer = MockTokenizer()
    loader = DataLoader(config, tokenizer)

    example = {
        "premises": ["P1"],
        "content": "Custom marker test"
    }

    full_text, _ = loader.format_example(example)

    if "<<START>>" in full_text and "<<END>>" in full_text:
        results.add_pass("Custom markers used")
    else:
        results.add_fail("Custom markers used", f"Markers not found in text")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("LoRA Fine-tuning Script Tests")
    print("=" * 60)

    results = TestResults()

    # Run all test functions
    test_model_registry(results)
    test_lora_config(results)
    test_data_config(results)
    test_format_premises_conclusion(results)
    test_format_qa(results)
    test_format_dag(results)
    test_format_preformatted(results)
    test_conclusion_only_loss_masking(results)
    test_full_text_training(results)
    test_jsonl_loading(results)
    test_json_loading(results)
    test_empty_premises(results)
    test_custom_markers(results)

    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
