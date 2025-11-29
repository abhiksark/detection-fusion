"""
Detailed Error Analysis for Object Detection

This module provides comprehensive error analysis capabilities including
error type classification, spatial analysis, and confidence-based error patterns.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class ErrorType(Enum):
    """Types of detection errors."""

    FALSE_POSITIVE = "false_positive"  # Detected but not in GT
    FALSE_NEGATIVE = "false_negative"  # In GT but not detected
    LOCALIZATION_ERROR = "localization_error"  # Correct class, poor localization
    CLASSIFICATION_ERROR = "classification_error"  # Good localization, wrong class
    DUPLICATE_DETECTION = "duplicate_detection"  # Multiple detections for same GT
    TRUE_POSITIVE = "true_positive"  # Correct detection


@dataclass
class ErrorInstance:
    """Represents a single error instance."""

    error_type: ErrorType
    prediction: Optional[Detection]
    ground_truth: Optional[Detection]
    iou: float = 0.0
    confidence: float = 0.0
    spatial_location: Tuple[float, float] = (0.0, 0.0)  # Center coordinates


@dataclass
class ErrorSummary:
    """Summary of errors by type."""

    false_positives: int = 0
    false_negatives: int = 0
    localization_errors: int = 0
    classification_errors: int = 0
    duplicate_detections: int = 0
    true_positives: int = 0

    @property
    def total_errors(self) -> int:
        """Total number of errors (excluding true positives)."""
        return (
            self.false_positives
            + self.false_negatives
            + self.localization_errors
            + self.classification_errors
            + self.duplicate_detections
        )

    @property
    def total_detections(self) -> int:
        """Total number of detections analyzed."""
        return self.total_errors + self.true_positives


class ErrorAnalyzer:
    """
    Comprehensive error analysis for object detection results.

    Analyzes prediction errors in detail, classifying them by type,
    spatial distribution, confidence patterns, and object characteristics.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        localization_iou_threshold: float = 0.1,
        classification_iou_threshold: float = 0.5,
    ):
        """
        Initialize error analyzer.

        Args:
            iou_threshold: IoU threshold for matching predictions to GT
            localization_iou_threshold: Min IoU for localization errors (vs false positive)
            classification_iou_threshold: Min IoU for classification errors
        """
        self.iou_threshold = iou_threshold
        self.localization_iou_threshold = localization_iou_threshold
        self.classification_iou_threshold = classification_iou_threshold

    def analyze_errors(
        self, predictions: List[Detection], ground_truth: List[Detection]
    ) -> Tuple[List[ErrorInstance], ErrorSummary]:
        """
        Analyze all errors in predictions vs ground truth.

        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections

        Returns:
            Tuple of (error_instances, error_summary)
        """
        error_instances = []
        used_gt_indices = set()

        # Sort predictions by confidence (descending)
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        # Track GT matches for duplicate detection
        gt_match_count = defaultdict(int)

        # Analyze each prediction
        for pred in sorted_predictions:
            best_iou = 0.0
            best_gt_idx = -1
            best_gt = None

            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truth):
                iou = calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt = gt

            # Classify the prediction
            error_type, error_instance = self._classify_prediction(
                pred, best_gt, best_iou, best_gt_idx, used_gt_indices, gt_match_count
            )

            error_instances.append(error_instance)

            # Update tracking
            if best_gt_idx != -1 and error_type == ErrorType.TRUE_POSITIVE:
                used_gt_indices.add(best_gt_idx)
            elif best_gt_idx != -1:
                gt_match_count[best_gt_idx] += 1

        # Add false negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in used_gt_indices and gt_match_count[gt_idx] == 0:
                error_instance = ErrorInstance(
                    error_type=ErrorType.FALSE_NEGATIVE,
                    prediction=None,
                    ground_truth=gt,
                    iou=0.0,
                    confidence=0.0,
                    spatial_location=(gt.x, gt.y),
                )
                error_instances.append(error_instance)

        # Generate summary
        summary = self._generate_error_summary(error_instances)

        return error_instances, summary

    def analyze_by_confidence(
        self, error_instances: List[ErrorInstance], confidence_bins: int = 10
    ) -> Dict[str, Dict[float, int]]:
        """
        Analyze error patterns by confidence ranges.

        Args:
            error_instances: List of error instances
            confidence_bins: Number of confidence bins

        Returns:
            Dictionary mapping error types to confidence distribution
        """
        # Create confidence bins
        confidences = [err.confidence for err in error_instances if err.prediction is not None]
        if not confidences:
            return {}

        min_conf, max_conf = min(confidences), max(confidences)
        bin_edges = np.linspace(min_conf, max_conf, confidence_bins + 1)

        # Count errors by type and confidence bin
        error_by_conf = defaultdict(lambda: defaultdict(int))

        for error in error_instances:
            if error.prediction is None:
                continue  # Skip false negatives (no confidence)

            # Find confidence bin
            conf = error.confidence
            bin_idx = np.digitize(conf, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, confidence_bins - 1))  # Clamp to valid range
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2

            error_by_conf[error.error_type.value][bin_center] += 1

        return dict(error_by_conf)

    def analyze_spatial_distribution(
        self, error_instances: List[ErrorInstance], grid_size: Tuple[int, int] = (10, 10)
    ) -> Dict[str, np.ndarray]:
        """
        Analyze spatial distribution of errors.

        Args:
            error_instances: List of error instances
            grid_size: Size of spatial grid (rows, cols)

        Returns:
            Dictionary mapping error types to spatial grids
        """
        rows, cols = grid_size
        spatial_grids = {}

        # Initialize grids for each error type
        for error_type in ErrorType:
            spatial_grids[error_type.value] = np.zeros((rows, cols))

        # Count errors in each grid cell
        for error in error_instances:
            x, y = error.spatial_location

            # Convert to grid coordinates (assuming normalized coordinates 0-1)
            grid_x = int(min(x * cols, cols - 1))
            grid_y = int(min(y * rows, rows - 1))

            spatial_grids[error.error_type.value][grid_y, grid_x] += 1

        return spatial_grids

    def analyze_by_object_size(
        self,
        error_instances: List[ErrorInstance],
        size_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze error patterns by object size.

        Args:
            error_instances: List of error instances
            size_ranges: Dictionary defining size categories and their ranges

        Returns:
            Dictionary mapping size categories to error type counts
        """
        if size_ranges is None:
            size_ranges = {"small": (0.0, 0.05), "medium": (0.05, 0.15), "large": (0.15, 1.0)}

        error_by_size = defaultdict(lambda: defaultdict(int))

        for error in error_instances:
            # Get object area
            if error.prediction is not None:
                area = error.prediction.area
            elif error.ground_truth is not None:
                area = error.ground_truth.area
            else:
                continue

            # Classify by size
            size_category = "unknown"
            for category, (min_size, max_size) in size_ranges.items():
                if min_size <= area < max_size:
                    size_category = category
                    break

            error_by_size[size_category][error.error_type.value] += 1

        return dict(error_by_size)

    def _classify_prediction(
        self,
        prediction: Detection,
        best_gt: Optional[Detection],
        best_iou: float,
        best_gt_idx: int,
        used_gt_indices: set,
        gt_match_count: Dict[int, int],
    ) -> Tuple[ErrorType, ErrorInstance]:
        """
        Classify a single prediction into error type.

        Args:
            prediction: The prediction to classify
            best_gt: Best matching ground truth (if any)
            best_iou: IoU with best matching ground truth
            best_gt_idx: Index of best matching ground truth
            used_gt_indices: Set of already used ground truth indices
            gt_match_count: Count of matches per ground truth

        Returns:
            Tuple of (error_type, error_instance)
        """
        # No matching ground truth or IoU too low
        if best_gt is None or best_iou < self.localization_iou_threshold:
            return ErrorType.FALSE_POSITIVE, ErrorInstance(
                error_type=ErrorType.FALSE_POSITIVE,
                prediction=prediction,
                ground_truth=None,
                iou=best_iou,
                confidence=prediction.confidence,
                spatial_location=(prediction.x, prediction.y),
            )

        # Check for duplicate detection
        if best_gt_idx in used_gt_indices or gt_match_count[best_gt_idx] > 0:
            return ErrorType.DUPLICATE_DETECTION, ErrorInstance(
                error_type=ErrorType.DUPLICATE_DETECTION,
                prediction=prediction,
                ground_truth=best_gt,
                iou=best_iou,
                confidence=prediction.confidence,
                spatial_location=(prediction.x, prediction.y),
            )

        # Good localization but wrong class
        if (
            best_iou >= self.classification_iou_threshold
            and prediction.class_id != best_gt.class_id
        ):
            return ErrorType.CLASSIFICATION_ERROR, ErrorInstance(
                error_type=ErrorType.CLASSIFICATION_ERROR,
                prediction=prediction,
                ground_truth=best_gt,
                iou=best_iou,
                confidence=prediction.confidence,
                spatial_location=(prediction.x, prediction.y),
            )

        # Correct class but poor localization
        if (
            self.localization_iou_threshold <= best_iou < self.iou_threshold
            and prediction.class_id == best_gt.class_id
        ):
            return ErrorType.LOCALIZATION_ERROR, ErrorInstance(
                error_type=ErrorType.LOCALIZATION_ERROR,
                prediction=prediction,
                ground_truth=best_gt,
                iou=best_iou,
                confidence=prediction.confidence,
                spatial_location=(prediction.x, prediction.y),
            )

        # True positive (good IoU and correct class)
        if best_iou >= self.iou_threshold and prediction.class_id == best_gt.class_id:
            return ErrorType.TRUE_POSITIVE, ErrorInstance(
                error_type=ErrorType.TRUE_POSITIVE,
                prediction=prediction,
                ground_truth=best_gt,
                iou=best_iou,
                confidence=prediction.confidence,
                spatial_location=(prediction.x, prediction.y),
            )

        # Default to false positive if none of the above
        return ErrorType.FALSE_POSITIVE, ErrorInstance(
            error_type=ErrorType.FALSE_POSITIVE,
            prediction=prediction,
            ground_truth=best_gt,
            iou=best_iou,
            confidence=prediction.confidence,
            spatial_location=(prediction.x, prediction.y),
        )

    def _generate_error_summary(self, error_instances: List[ErrorInstance]) -> ErrorSummary:
        """Generate summary statistics from error instances."""
        summary = ErrorSummary()

        for error in error_instances:
            if error.error_type == ErrorType.FALSE_POSITIVE:
                summary.false_positives += 1
            elif error.error_type == ErrorType.FALSE_NEGATIVE:
                summary.false_negatives += 1
            elif error.error_type == ErrorType.LOCALIZATION_ERROR:
                summary.localization_errors += 1
            elif error.error_type == ErrorType.CLASSIFICATION_ERROR:
                summary.classification_errors += 1
            elif error.error_type == ErrorType.DUPLICATE_DETECTION:
                summary.duplicate_detections += 1
            elif error.error_type == ErrorType.TRUE_POSITIVE:
                summary.true_positives += 1

        return summary


class ErrorClassifier:
    """
    Advanced error classification with customizable rules and thresholds.
    """

    def __init__(self, custom_rules: Optional[Dict] = None):
        """
        Initialize with custom classification rules.

        Args:
            custom_rules: Dictionary of custom classification rules
        """
        self.rules = custom_rules or {}

    def classify_by_context(
        self, error_instances: List[ErrorInstance], image_metadata: Optional[Dict] = None
    ) -> Dict[str, List[ErrorInstance]]:
        """
        Classify errors by contextual information.

        Args:
            error_instances: List of error instances
            image_metadata: Optional metadata about the image context

        Returns:
            Dictionary grouping errors by context
        """
        # This would be extended based on specific needs
        # For now, group by basic categories

        context_groups = {
            "high_confidence_errors": [],
            "low_confidence_errors": [],
            "edge_errors": [],
            "center_errors": [],
            "small_object_errors": [],
            "large_object_errors": [],
        }

        for error in error_instances:
            # High/low confidence
            if error.confidence > 0.7:
                context_groups["high_confidence_errors"].append(error)
            elif error.confidence < 0.3:
                context_groups["low_confidence_errors"].append(error)

            # Spatial location
            x, y = error.spatial_location
            if x < 0.2 or x > 0.8 or y < 0.2 or y > 0.8:  # Near edges
                context_groups["edge_errors"].append(error)
            else:
                context_groups["center_errors"].append(error)

            # Object size (if available)
            if error.prediction and error.prediction.area < 0.05:
                context_groups["small_object_errors"].append(error)
            elif error.prediction and error.prediction.area > 0.15:
                context_groups["large_object_errors"].append(error)

        return context_groups
