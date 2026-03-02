"""Comprehensive evaluation framework."""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..data.schema import safe_parse_model_output, Annotation
from ..verification.verifier import VerifierPipeline, VerifierConfig
from ..data.curation import DataCurator


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    verifier_config: VerifierConfig
    metrics: List[str] = None
    human_eval_sample: float = 0.1
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "premise_precision",
                "premise_recall",
                "evidence_recall",
                "entailment_accuracy",
                "verifier_calibration",
                "parse_success_rate"
            ]


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.detailed_results = []
    
    def add_metric(self, name: str, value: float):
        """Add a metric."""
        self.metrics[name] = value
    
    def add_result(self, result: Dict[str, Any]):
        """Add detailed result for an example."""
        self.detailed_results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "num_examples": len(self.detailed_results),
            "sample_results": self.detailed_results[:10]
        }
    
    def print_summary(self):
        """Print summary of metrics."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        for name, value in sorted(self.metrics.items()):
            print(f"{name}: {value:.4f}")
        print("="*50)


class ModelEvaluator:
    """Evaluate model outputs comprehensively."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.verifier = VerifierPipeline(config.verifier_config)
    
    def evaluate_batch(
        self,
        model_outputs: List[str],
        ground_truth: List[Dict[str, Any]],
        evidence_spans: Optional[List[List[List[Dict[str, Any]]]]] = None
    ) -> EvaluationMetrics:
        """Evaluate a batch of model outputs."""
        metrics = EvaluationMetrics()
        
        parsed_outputs = []
        parse_errors = 0
        
        # Parse outputs
        for i, output in enumerate(model_outputs):
            parsed, error = safe_parse_model_output(output)
            if parsed is None:
                parse_errors += 1
                parsed = {"premises": [], "content": ""}
            parsed_outputs.append(parsed)
        
        parse_success_rate = 1.0 - (parse_errors / len(model_outputs))
        metrics.add_metric("parse_success_rate", parse_success_rate)
        
        # Premise metrics
        premise_precision, premise_recall = self._compute_premise_metrics(
            parsed_outputs, ground_truth
        )
        metrics.add_metric("premise_precision", premise_precision)
        metrics.add_metric("premise_recall", premise_recall)
        
        # Evidence recall
        evidence_recall = self._compute_evidence_recall(parsed_outputs, ground_truth)
        metrics.add_metric("evidence_recall", evidence_recall)
        
        # Entailment accuracy
        entailment_accuracy = self._compute_entailment_accuracy(parsed_outputs, ground_truth)
        metrics.add_metric("entailment_accuracy", entailment_accuracy)
        
        # Verifier calibration
        verifier_calibration = self._compute_verifier_calibration(
            parsed_outputs, ground_truth, evidence_spans
        )
        metrics.add_metric("verifier_calibration", verifier_calibration)
        
        # Per-example detailed results
        for i, (parsed, gt) in enumerate(zip(parsed_outputs, ground_truth)):
            verdict = self.verifier.verify_output(
                parsed,
                evidence_spans[i] if evidence_spans and i < len(evidence_spans) else None
            )
            
            metrics.add_result({
                "example_id": gt.get("id", i),
                "parse_success": parsed is not None,
                "verdict": verdict["verdict"],
                "verifier_confidence": verdict["confidence"],
                "premises_count": len(parsed.get("premises", [])),
                "ground_truth_premises_count": len(gt.get("premises", []))
            })
        
        return metrics
    
    def _compute_premise_metrics(
        self,
        parsed_outputs: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Compute premise precision and recall."""
        all_premises_pred = []
        all_premises_true = []
        
        for parsed, gt in zip(parsed_outputs, ground_truth):
            pred_premises = parsed.get("premises", [])
            true_premises = gt.get("premises", [])
            
            # Extract premise texts
            pred_texts = [p if isinstance(p, str) else p.get("text", "") for p in pred_premises]
            true_texts = [p if isinstance(p, str) else p.get("text", "") for p in true_premises]
            
            all_premises_pred.extend(pred_texts)
            all_premises_true.extend(true_texts)
        
        # Simple word overlap-based matching
        matches = 0
        for pred in all_premises_pred:
            for true_prem in all_premises_true:
                # Simple overlap check
                pred_words = set(pred.lower().split())
                true_words = set(true_prem.lower().split())
                overlap = len(pred_words & true_words) / max(len(pred_words), 1)
                if overlap > 0.5:
                    matches += 1
                    break
        
        precision = matches / len(all_premises_pred) if all_premises_pred else 0.0
        recall = matches / len(all_premises_true) if all_premises_true else 0.0
        
        return precision, recall
    
    def _compute_evidence_recall(
        self,
        parsed_outputs: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """Compute evidence recall."""
        total_evidence = 0
        recalled_evidence = 0
        
        for parsed, gt in zip(parsed_outputs, ground_truth):
            gt_premises = gt.get("premises", [])
            for premise in gt_premises:
                if isinstance(premise, dict):
                    evidence_spans = premise.get("evidence_spans", [])
                    total_evidence += len(evidence_spans)
        
        # This is a simplified version - would need actual retrieval matching
        # For now, return a placeholder
        return 0.5 if total_evidence > 0 else 1.0
    
    def _compute_entailment_accuracy(
        self,
        parsed_outputs: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """Compute entailment accuracy."""
        correct = 0
        total = 0
        
        for parsed, gt in zip(parsed_outputs, ground_truth):
            pred_conclusion = parsed.get("content") or parsed.get("conclusion", "")
            gt_conclusion_data = gt.get("content") or gt.get("conclusion", {})
            
            if isinstance(gt_conclusion_data, dict):
                gt_conclusion = gt_conclusion_data.get("text", "")
                gt_type = gt_conclusion_data.get("type", "entailment")
            else:
                gt_conclusion = gt_conclusion_data
                gt_type = "entailment"
            
            # Simple similarity check
            pred_words = set(pred_conclusion.lower().split())
            gt_words = set(gt_conclusion.lower().split())
            overlap = len(pred_words & gt_words) / max(len(gt_words), 1)
            
            if overlap > 0.6:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_verifier_calibration(
        self,
        parsed_outputs: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        evidence_spans: Optional[List[List[List[Dict[str, Any]]]]]
    ) -> float:
        """Compute verifier calibration (Expected Calibration Error approximation)."""
        confidence_bins = []
        correctness_bins = []
        
        for i, (parsed, gt) in enumerate(zip(parsed_outputs, ground_truth)):
            verdict = self.verifier.verify_output(
                parsed,
                evidence_spans[i] if evidence_spans and i < len(evidence_spans) else None
            )
            
            confidence = verdict["confidence"]
            # Determine correctness (simplified)
            gt_conclusion = gt.get("content") or gt.get("conclusion", {})
            if isinstance(gt_conclusion, dict):
                gt_type = gt_conclusion.get("type", "entailment")
            else:
                gt_type = "entailment"
            
            is_correct = (verdict["verdict"] == "accept" and gt_type == "entailment") or \
                        (verdict["verdict"] != "accept" and gt_type != "entailment")
            
            confidence_bins.append(confidence)
            correctness_bins.append(1.0 if is_correct else 0.0)
        
        # Simple ECE approximation
        if len(confidence_bins) == 0:
            return 0.0
        
        # Bin confidences
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        
        for i in range(len(bins) - 1):
            bin_mask = (np.array(confidence_bins) >= bins[i]) & (np.array(confidence_bins) < bins[i + 1])
            if bin_mask.sum() > 0:
                bin_conf = np.mean(np.array(confidence_bins)[bin_mask])
                bin_acc = np.mean(np.array(correctness_bins)[bin_mask])
                ece += abs(bin_conf - bin_acc) * bin_mask.sum()
        
        ece /= len(confidence_bins)
        return ece


def evaluate_model(
    model_outputs: List[str],
    ground_truth_path: str,
    verifier_config_path: Optional[str] = None,
    config_path: str = "./config.yaml"
) -> EvaluationMetrics:
    """Evaluate model outputs against ground truth."""
    import yaml
    
    from ..data.curation import DataCurator
    
    # Load configs with environment variable support
    from ..utils.config_loader import load_config
    config = load_config(config_path)
    
    verifier_config = VerifierConfig(**config["verifier"])
    eval_config = EvaluationConfig(
        verifier_config=verifier_config,
        metrics=config["evaluation"]["metrics"]
    )
    
    # Load ground truth
    curator = DataCurator()
    ground_truth = curator.load_jsonl(ground_truth_path)
    
    # Evaluate
    evaluator = ModelEvaluator(eval_config)
    metrics = evaluator.evaluate_batch(model_outputs, ground_truth)
    
    return metrics

