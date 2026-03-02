"""Synthetic data generation pipeline for premise-conclusion pairs."""

import json
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .schema import Annotation, Premise, EvidenceSpan, ANNOTATION_SCHEMA


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    model_name: str = "deepseek-ai/deepseek-v3"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    num_examples: int = 1000
    include_adversarial: bool = True
    adversarial_ratio: float = 0.2


class SyntheticDataGenerator:
    """Generate synthetic premise-conclusion pairs using a base LLM."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the base model for generation."""
        print(f"Loading model {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def _generate_premise_conclusion_prompt(self, document: str, question: Optional[str] = None) -> str:
        """Create prompt for generating premise-conclusion pairs."""
        base_prompt = """Generate a structured reasoning chain in JSON format based on the following document.

Document:
{document}

{question_section}

You must respond in the following JSON format:
{{
  "premises": [
    "Premise 1: a concise factual statement",
    "Premise 2: another factual statement"
  ],
  "content": "A conclusion that logically follows from the premises"
}}

Requirements:
- Premises should be factual statements that can be verified
- Conclusion must logically follow from all premises
- Include 2-5 premises
- Be specific and accurate

JSON response:"""
        
        question_section = f"Question: {question}\n\n" if question else ""
        return base_prompt.format(document=document, question_section=question_section)
    
    def _generate_adversarial_prompt(self, document: str, question: Optional[str] = None) -> str:
        """Create prompt for generating adversarial (broken) examples."""
        base_prompt = """Generate a structured reasoning chain in JSON format that contains a logical error or unsupported conclusion.

Document:
{document}

{question_section}

You must respond in the following JSON format:
{{
  "premises": [
    "Premise 1: a factual statement",
    "Premise 2: another statement"
  ],
  "conclusion": "A conclusion that does NOT logically follow or is unsupported"
}}

Requirements:
- Premises should be factual but the conclusion should have a logical flaw
- Examples of flaws: missing premise, contradiction, scope shift, or unsupported inference
- Include 2-5 premises
- The error should be subtle but detectable

JSON response:"""
        
        question_section = f"Question: {question}\n\n" if question else ""
        return base_prompt.format(document=document, question_section=question_section)
    
    def generate_from_document(
        self, 
        document: str, 
        question: Optional[str] = None,
        adversarial: bool = False
    ) -> Optional[Annotation]:
        """Generate a single annotation from a document."""
        prompt = self._generate_adversarial_prompt(document, question) if adversarial else \
                 self._generate_premise_conclusion_prompt(document, question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Parse the generated output
        from .schema import safe_parse_model_output
        parsed, error = safe_parse_model_output(generated_text)
        
        if parsed is None:
            return None
        
        # Convert to Annotation format
        premises = [
            Premise(
                id=f"p{i+1}",
                text=p,
                evidence_spans=[]  # Would need document chunking to link evidence
            ) for i, p in enumerate(parsed.get("premises", []))
        ]
        
        content_text = parsed.get("content") or parsed.get("conclusion", "")
        if isinstance(content_text, dict):
            content_text = content_text.get("text", "")
            conclusion_type = content_text.get("type", "unsupported" if adversarial else "entailment")
        else:
            conclusion_type = "unsupported" if adversarial else "entailment"

        return Annotation(
            id=str(uuid.uuid4()),
            context=document,
            premises=premises,
            content=content_text,
            conclusion_type=conclusion_type,
            confidence=0.7 if adversarial else 0.9,
            annotator_id="synthetic_generator",
            timestamp=datetime.now().isoformat()
        )
    
    def generate_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
        output_path: str = "./data/synthetic.jsonl"
    ) -> List[Annotation]:
        """Generate a batch of synthetic annotations."""
        annotations = []
        num_adversarial = int(self.config.num_examples * self.config.adversarial_ratio)
        num_normal = self.config.num_examples - num_adversarial
        
        print(f"Generating {num_normal} normal and {num_adversarial} adversarial examples...")
        
        for i in range(self.config.num_examples):
            doc_idx = i % len(documents)
            document = documents[doc_idx]
            question = questions[doc_idx] if questions and doc_idx < len(questions) else None
            
            adversarial = i < num_adversarial
            annotation = self.generate_from_document(document, question, adversarial=adversarial)
            
            if annotation:
                annotations.append(annotation)
                print(f"Generated {i+1}/{self.config.num_examples}: {len(annotations)} valid")
            else:
                print(f"Failed to generate example {i+1}")
        
        # Save to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(ann.to_jsonl() + "\n")
        
        print(f"Saved {len(annotations)} annotations to {output_path}")
        return annotations


class DataAugmenter:
    """Augment existing annotations with perturbations."""
    
    @staticmethod
    def create_negation_flip(annotation: Annotation) -> Annotation:
        """Create a version with negated conclusion."""
        negated_content = f"NOT ({annotation.content})"
        return Annotation(
            id=str(uuid.uuid4()),
            context=annotation.context,
            premises=annotation.premises,
            content=negated_content,
            conclusion_type="contradiction",
            confidence=0.8,
            annotator_id="augmenter_negation",
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_entity_swap(annotation: Annotation, entity_map: Dict[str, str]) -> Annotation:
        """Swap entities in premises and conclusion."""
        def swap_text(text: str) -> str:
            for old, new in entity_map.items():
                text = text.replace(old, new)
            return text
        
        new_premises = [
            Premise(
                id=p.id,
                text=swap_text(p.text),
                evidence_spans=p.evidence_spans
            ) for p in annotation.premises
        ]
        
        new_content = swap_text(annotation.content)

        return Annotation(
            id=str(uuid.uuid4()),
            context=annotation.context,
            premises=new_premises,
            content=new_content,
            conclusion_type="unsupported",  # Entity swap usually breaks support
            confidence=0.7,
            annotator_id="augmenter_entity_swap",
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_missing_premise(annotation: Annotation, drop_ratio: float = 0.3) -> Annotation:
        """Create version with some premises removed (creating unsupported conclusion)."""
        num_to_keep = max(1, int(len(annotation.premises) * (1 - drop_ratio)))
        kept_premises = random.sample(annotation.premises, num_to_keep)
        
        return Annotation(
            id=str(uuid.uuid4()),
            context=annotation.context,
            premises=kept_premises,
            content=annotation.content,
            conclusion_type="unsupported",
            confidence=0.6,
            annotator_id="augmenter_missing_premise",
            timestamp=datetime.now().isoformat()
        )


