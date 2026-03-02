"""Evaluation framework for model outputs."""

from .evaluator import ModelEvaluator, EvaluationMetrics, EvaluationConfig
from .benchmark_registry import (
    BenchmarkConfig,
    BenchmarkType,
    BenchmarkCategory,
    BENCHMARK_REGISTRY,
    get_benchmark_config,
    list_benchmarks,
)
from .benchmark_evaluator import BenchmarkEvaluator, ComparisonResult, BenchmarkResult
from .benchmark_loaders import get_loader, BenchmarkItem
from .report_generator import ReportGenerator

__all__ = [
    # Existing
    "ModelEvaluator",
    "EvaluationMetrics",
    "EvaluationConfig",
    # Benchmark evaluation
    "BenchmarkConfig",
    "BenchmarkType",
    "BenchmarkCategory",
    "BENCHMARK_REGISTRY",
    "get_benchmark_config",
    "list_benchmarks",
    "BenchmarkEvaluator",
    "ComparisonResult",
    "BenchmarkResult",
    "ReportGenerator",
    # Loaders
    "get_loader",
    "BenchmarkItem",
]
