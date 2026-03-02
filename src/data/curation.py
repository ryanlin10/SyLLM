"""Data curation, validation, and dataset management."""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random
from collections import Counter

from .schema import Annotation, validate_annotation, safe_parse_model_output


class DataCurator:
    """Manages data curation, validation, and splitting."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load annotations from JSONL file."""
        annotations = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    annotations.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
        return annotations
    
    def save_jsonl(self, annotations: List[Dict[str, Any]], file_path: str):
        """Save annotations to JSONL file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    
    def validate_dataset(self, annotations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate annotations and return valid ones with error messages."""
        valid = []
        errors = []
        
        for i, ann in enumerate(annotations):
            is_valid, error = validate_annotation(ann)
            if is_valid:
                valid.append(ann)
            else:
                errors.append(f"Annotation {i} (id: {ann.get('id', 'unknown')}): {error}")
        
        return valid, errors
    
    def check_data_quality(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run quality checks on dataset."""
        stats = {
            "total": len(annotations),
            "avg_premises_per_example": 0,
            "conclusion_types": Counter(),
            "with_evidence": 0,
            "confidence_scores": [],
        }
        
        premise_counts = []
        for ann in annotations:
            premises = ann.get("premises", [])
            premise_counts.append(len(premises))
            
            # Check for evidence spans
            has_evidence = any(
                len(p.get("evidence_spans", [])) > 0 
                for p in premises
            )
            if has_evidence:
                stats["with_evidence"] += 1
            
            # Content/conclusion type
            content = ann.get("content") or ann.get("conclusion", {})
            if isinstance(content, dict):
                conclusion_type = content.get("type", "entailment")
            else:
                conclusion_type = "entailment"
            stats["conclusion_types"][conclusion_type] += 1
            
            # Confidence
            conf = ann.get("confidence", 1.0)
            stats["confidence_scores"].append(conf)
        
        stats["avg_premises_per_example"] = sum(premise_counts) / len(premise_counts) if premise_counts else 0
        stats["avg_confidence"] = sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) if stats["confidence_scores"] else 0
        
        return stats
    
    def split_dataset(
        self,
        annotations: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_by_time: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/val/test."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        if split_by_time:
            # Sort by timestamp
            sorted_anns = sorted(
                annotations,
                key=lambda x: x.get("timestamp", "2000-01-01")
            )
        else:
            sorted_anns = annotations.copy()
            random.shuffle(sorted_anns)
        
        total = len(sorted_anns)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = sorted_anns[:train_end]
        val_data = sorted_anns[train_end:val_end]
        test_data = sorted_anns[val_end:]
        
        return train_data, val_data, test_data
    
    def balance_classes(
        self,
        annotations: List[Dict[str, Any]],
        target_counts: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Balance dataset by conclusion type."""
        # Group by content/conclusion type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for ann in annotations:
            content = ann.get("content") or ann.get("conclusion", {})
            if isinstance(content, dict):
                ctype = content.get("type", "entailment")
            else:
                ctype = "entailment"
            
            if ctype not in by_type:
                by_type[ctype] = []
            by_type[ctype].append(ann)
        
        # Determine target counts
        if target_counts is None:
            # Use the maximum count as target
            max_count = max(len(v) for v in by_type.values())
            target_counts = {k: max_count for k in by_type.keys()}
        
        # Sample to balance
        balanced = []
        for ctype, examples in by_type.items():
            target = target_counts.get(ctype, len(examples))
            if len(examples) >= target:
                balanced.extend(random.sample(examples, target))
            else:
                balanced.extend(examples)
                # Optionally duplicate to reach target
        
        random.shuffle(balanced)
        return balanced
    
    def create_train_split(
        self,
        input_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        validate: bool = True,
        balance: bool = False
    ):
        """Create train/val/test splits from input JSONL."""
        print(f"Loading data from {input_path}...")
        annotations = self.load_jsonl(input_path)
        print(f"Loaded {len(annotations)} annotations")
        
        # Validate
        if validate:
            valid, errors = self.validate_dataset(annotations)
            print(f"Validated: {len(valid)} valid, {len(errors)} errors")
            if errors:
                print("Sample errors:")
                for err in errors[:5]:
                    print(f"  - {err}")
            annotations = valid
        
        # Quality check
        quality_stats = self.check_data_quality(annotations)
        print("Quality statistics:")
        for key, value in quality_stats.items():
            if key != "confidence_scores":
                print(f"  {key}: {value}")
        
        # Balance if requested
        if balance:
            annotations = self.balance_classes(annotations)
            print(f"After balancing: {len(annotations)} annotations")
        
        # Split
        train, val, test = self.split_dataset(
            annotations,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.save_jsonl(train, f"{output_dir}/train.jsonl")
        self.save_jsonl(val, f"{output_dir}/val.jsonl")
        self.save_jsonl(test, f"{output_dir}/test.jsonl")
        
        print(f"Saved splits to {output_dir}")
        
        return train, val, test


