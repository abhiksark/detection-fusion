"""
Detection statistics utilities.

This module provides common statistical functions for detection analysis,
eliminating duplicate calculations across CLI tools.
"""

from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np

if TYPE_CHECKING:
    from .detection import Detection


def average_confidence(detections: "List[Detection]") -> float:
    """
    Calculate average confidence of detections.

    This replaces the duplicated pattern:
        sum(d.confidence for d in results) / len(results) if results else 0

    Args:
        detections: List of Detection objects

    Returns:
        Average confidence (0.0 if empty)
    """
    if not detections:
        return 0.0
    return sum(d.confidence for d in detections) / len(detections)


def confidence_stats(detections: "List[Detection]") -> Dict[str, float]:
    """
    Calculate confidence statistics for detections.

    Args:
        detections: List of Detection objects

    Returns:
        Dict with mean, std, min, max, median confidence values
    """
    if not detections:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    confidences = [d.confidence for d in detections]
    return {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "median": float(np.median(confidences)),
    }


def detection_summary(detections: "List[Detection]") -> Dict:
    """
    Get comprehensive summary statistics for a list of detections.

    Args:
        detections: List of Detection objects

    Returns:
        Dict containing count, avg_confidence, unique_classes, class_distribution
    """
    if not detections:
        return {"count": 0, "avg_confidence": 0.0, "unique_classes": 0, "class_distribution": {}}

    class_counts = Counter(d.class_id for d in detections)

    return {
        "count": len(detections),
        "avg_confidence": average_confidence(detections),
        "unique_classes": len(class_counts),
        "class_distribution": dict(class_counts),
    }


def class_statistics(detections: "List[Detection]") -> Dict[int, Dict]:
    """
    Calculate per-class statistics.

    Args:
        detections: List of Detection objects

    Returns:
        Dict mapping class_id to stats dict (count, avg_confidence, etc.)
    """
    if not detections:
        return {}

    # Group by class
    by_class: Dict[int, List["Detection"]] = {}
    for det in detections:
        if det.class_id not in by_class:
            by_class[det.class_id] = []
        by_class[det.class_id].append(det)

    # Calculate stats per class
    stats = {}
    for class_id, class_dets in by_class.items():
        confidences = [d.confidence for d in class_dets]
        stats[class_id] = {
            "count": len(class_dets),
            "avg_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
        }

    return stats


def model_statistics(detections_by_model: "Dict[str, List[Detection]]") -> Dict[str, Dict]:
    """
    Calculate per-model statistics.

    Args:
        detections_by_model: Dict mapping model names to detection lists

    Returns:
        Dict mapping model name to stats dict
    """
    stats = {}
    for model, dets in detections_by_model.items():
        stats[model] = detection_summary(dets)
    return stats


def unique_classes(detections: "List[Detection]") -> Set[int]:
    """
    Get set of unique class IDs in detections.

    Args:
        detections: List of Detection objects

    Returns:
        Set of unique class IDs
    """
    return set(d.class_id for d in detections)


def unique_images(detections: "List[Detection]") -> Set[str]:
    """
    Get set of unique image names in detections.

    Args:
        detections: List of Detection objects

    Returns:
        Set of unique image names
    """
    return set(d.image_name for d in detections if d.image_name)


def filter_by_confidence(
    detections: "List[Detection]", min_confidence: float = 0.0, max_confidence: float = 1.0
) -> "List[Detection]":
    """
    Filter detections by confidence range.

    Args:
        detections: List of Detection objects
        min_confidence: Minimum confidence (inclusive)
        max_confidence: Maximum confidence (inclusive)

    Returns:
        Filtered list of detections
    """
    return [d for d in detections if min_confidence <= d.confidence <= max_confidence]


def filter_by_class(detections: "List[Detection]", class_ids: Set[int]) -> "List[Detection]":
    """
    Filter detections by class ID.

    Args:
        detections: List of Detection objects
        class_ids: Set of class IDs to keep

    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.class_id in class_ids]


__all__ = [
    "average_confidence",
    "confidence_stats",
    "detection_summary",
    "class_statistics",
    "model_statistics",
    "unique_classes",
    "unique_images",
    "filter_by_confidence",
    "filter_by_class",
]
