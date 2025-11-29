"""
DetectionFusion Package

A comprehensive toolkit for fusing multiple object detection results with
ground truth validation and error analysis.
"""

from typing import Dict, List, Optional

from ._compat import has_matplotlib, has_rich, has_torch, require_torch
from ._version import __version__
from .exceptions import ConfigurationError, DetectionFusionError, FormatError

# ============================================================
# CONVENIENCE FUNCTIONS (primary user-facing API)
# ============================================================


def merge_detections(
    path: str,
    strategy: str = "weighted_vote",
    format: str = "auto",
    **kwargs,
) -> List["Detection"]:
    """Merge detections from multiple models using specified strategy.

    Args:
        path: Path to labels directory containing model subdirectories
        strategy: Strategy name (default: weighted_vote)
        format: Annotation format (auto-detected by default)
        **kwargs: Strategy-specific parameters (e.g., iou_threshold)

    Returns:
        List of merged detections
    """
    from .core.detection_set import DetectionSet
    from .data import FileDetectionLoader
    from .strategies import StrategyRegistry

    loader = FileDetectionLoader(path, format=format)
    detection_set = DetectionSet(loader.load_all())
    strategy_obj = StrategyRegistry.create(strategy, **kwargs)
    return strategy_obj.merge(detection_set.raw_data)


def evaluate_detections(
    predictions: List["Detection"],
    gt_path: str,
    iou_threshold: float = 0.5,
) -> "EvaluationResult":
    """Evaluate predictions against ground truth.

    Args:
        predictions: List of predicted detections
        gt_path: Path to ground truth directory
        iou_threshold: IoU threshold for matching

    Returns:
        EvaluationResult with precision, recall, F1, etc.
    """
    from .data import GroundTruthRepository
    from .pipeline.context import EvaluationResult
    from .pipeline.stages import EvaluationStage

    stage = EvaluationStage(gt_path, iou_threshold=iou_threshold)
    gt_repo = GroundTruthRepository(gt_path)
    ground_truth = gt_repo.load()

    return stage._evaluate(predictions, ground_truth, iou_threshold)


def convert_annotations(
    input_path: str,
    output_path: str,
    input_format: str = "auto",
    output_format: str = "yolo",
    image_size: Optional[tuple] = None,
) -> None:
    """Convert annotations between formats.

    Args:
        input_path: Path to input file or directory
        output_path: Path to output file or directory
        input_format: Input format (auto, yolo, voc_xml, coco)
        output_format: Output format (yolo, voc_xml, coco)
        image_size: Optional (width, height) for formats requiring size
    """
    from pathlib import Path

    from .data.formats import FormatRegistry

    input_p = Path(input_path)
    output_p = Path(output_path)

    if input_format == "auto":
        reader = FormatRegistry.auto_detect_reader(input_p)
    else:
        reader = FormatRegistry.get_reader(input_format)

    writer = FormatRegistry.get_writer(output_format)

    if input_p.is_file():
        detections = reader.read_file(input_p, image_size)
        writer.write_file(detections, output_p, image_size)
    else:
        result = reader.read_directory(input_p)
        output_p.mkdir(parents=True, exist_ok=True)
        writer.write_directory(result, output_p)


# ============================================================
# CORE CLASSES
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================
from .config import ConfigLoader, StrategyConfig
from .core.detection import Detection
from .core.detection_set import DetectionSet

# ============================================================
# DATA ACCESS
# ============================================================
from .data import FileDetectionLoader, GroundTruthRepository
from .data.formats import FormatRegistry
from .evaluation.error_analysis import ErrorAnalyzer

# ============================================================
# EVALUATION
# ============================================================
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import EvaluationMetrics

# ============================================================
# PIPELINE
# ============================================================
from .pipeline import DetectionPipeline, PipelineContext
from .pipeline.context import EvaluationResult

# ============================================================
# STRATEGIES
# ============================================================
from .strategies import BaseStrategy, StrategyRegistry

__all__ = [
    # Version
    "__version__",
    # Convenience functions
    "merge_detections",
    "evaluate_detections",
    "convert_annotations",
    # Compatibility
    "has_torch",
    "has_matplotlib",
    "has_rich",
    "require_torch",
    # Exceptions
    "DetectionFusionError",
    "ConfigurationError",
    "FormatError",
    # Core
    "Detection",
    "DetectionSet",
    # Data
    "FileDetectionLoader",
    "GroundTruthRepository",
    "FormatRegistry",
    # Config
    "StrategyConfig",
    "ConfigLoader",
    # Strategies
    "StrategyRegistry",
    "BaseStrategy",
    # Pipeline
    "DetectionPipeline",
    "PipelineContext",
    "EvaluationResult",
    # Evaluation
    "Evaluator",
    "EvaluationMetrics",
    "ErrorAnalyzer",
]
