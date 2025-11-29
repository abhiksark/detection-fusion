"""
DetectionFusion Evaluation Module

This module provides comprehensive evaluation capabilities for object detection
models and their ensemble combinations, including ground truth comparison,
standard metrics calculation, and detailed error analysis.

Key Components:
- EvaluationMetrics: Standard object detection metrics (AP, mAP, precision, recall)
- ErrorAnalyzer: Detailed error classification and analysis
- Evaluator: Main orchestrator for evaluation workflows
"""

from .error_analysis import ErrorAnalyzer, ErrorClassifier
from .evaluator import Evaluator
from .metrics import APCalculator, EvaluationMetrics

__all__ = [
    "EvaluationMetrics",
    "APCalculator",
    "ErrorAnalyzer",
    "ErrorClassifier",
    "Evaluator",
]
