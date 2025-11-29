"""
Detection aggregation utilities for merging detection clusters.

This module provides canonical implementations for cluster merging and
model weight calculation, replacing duplicated code across strategy files.
"""

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..core.detection import Detection


def calculate_model_weights(
    detections: "Dict[str, List[Detection]]", method: str = "confidence"
) -> Dict[str, float]:
    """
    Calculate normalized weights for each model.

    This is the canonical implementation replacing 3 duplicated versions
    in voting.py, clustering.py, and probabilistic.py.

    Args:
        detections: Dict mapping model names to their detections
        method: Weighting method:
            - "confidence": Weight by average confidence (default)
            - "count": Weight by detection count
            - "uniform": Equal weights for all models

    Returns:
        Dict mapping model names to normalized weights (sum to 1.0)
    """
    weights = {}

    for model, model_dets in detections.items():
        if not model_dets:
            weights[model] = 0.0
            continue

        if method == "confidence":
            weights[model] = np.mean([d.confidence for d in model_dets])
        elif method == "count":
            weights[model] = float(len(model_dets))
        else:  # uniform
            weights[model] = 1.0

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {m: w / total for m, w in weights.items()}

    return weights


def merge_cluster_weighted(
    cluster: "List[Detection]",
    weights: Optional[List[float]] = None,
    class_id: Optional[int] = None,
    source: str = "merged",
) -> "Detection":
    """
    Merge a cluster of detections using weighted averaging.

    This is the canonical cluster merging function that replaces duplicated
    weighted averaging code across strategy files.

    Args:
        cluster: List of Detection objects to merge
        weights: Optional weights for each detection (uses confidence if None)
        class_id: Class ID for result (uses majority vote if None)
        source: Model source string for the merged detection

    Returns:
        Single merged Detection object
    """
    from ..core.detection import Detection

    if not cluster:
        raise ValueError("Cannot merge empty cluster")

    if len(cluster) == 1:
        det = cluster[0]
        return Detection(
            class_id=det.class_id,
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            confidence=det.confidence,
            model_source=source,
            image_name=det.image_name,
        )

    # Use confidence as weights if not provided
    if weights is None:
        weights = [d.confidence for d in cluster]

    weights_arr = np.array(weights)
    if weights_arr.sum() > 0:
        weights_arr = weights_arr / weights_arr.sum()
    else:
        weights_arr = np.ones(len(cluster)) / len(cluster)

    # Determine class ID
    if class_id is None:
        class_id = vote_class_majority(cluster)

    # Weighted average of boxes
    avg_x = np.average([d.x for d in cluster], weights=weights_arr)
    avg_y = np.average([d.y for d in cluster], weights=weights_arr)
    avg_w = np.average([d.w for d in cluster], weights=weights_arr)
    avg_h = np.average([d.h for d in cluster], weights=weights_arr)
    avg_conf = np.average([d.confidence for d in cluster], weights=weights_arr)

    # Use image_name from first detection (assuming same image)
    image_name = cluster[0].image_name if cluster else ""

    return Detection(
        class_id=class_id,
        x=float(avg_x),
        y=float(avg_y),
        w=float(avg_w),
        h=float(avg_h),
        confidence=float(avg_conf),
        model_source=source,
        image_name=image_name,
    )


def merge_cluster_simple(cluster: "List[Detection]", source: str = "merged") -> "Detection":
    """
    Merge a cluster using simple averaging (no weights).

    Args:
        cluster: List of Detection objects to merge
        source: Model source string for the merged detection

    Returns:
        Single merged Detection object
    """
    weights = [1.0] * len(cluster)
    return merge_cluster_weighted(cluster, weights=weights, source=source)


def vote_class_majority(cluster: "List[Detection]") -> int:
    """
    Simple majority voting for class selection.

    Args:
        cluster: List of Detection objects

    Returns:
        Most common class ID in the cluster
    """
    if not cluster:
        raise ValueError("Cannot vote on empty cluster")

    votes = Counter(d.class_id for d in cluster)
    return votes.most_common(1)[0][0]


def vote_class_weighted(cluster: "List[Detection]", weights: List[float]) -> int:
    """
    Weighted voting for class selection.

    Args:
        cluster: List of Detection objects
        weights: Weight for each detection

    Returns:
        Class ID with highest weighted vote
    """
    if not cluster:
        raise ValueError("Cannot vote on empty cluster")

    if len(cluster) != len(weights):
        raise ValueError("Cluster and weights must have same length")

    scores: Dict[int, float] = defaultdict(float)
    for det, weight in zip(cluster, weights):
        scores[det.class_id] += weight

    return max(scores, key=scores.get)


def get_detection_weight(
    detection: "Detection",
    model_weights: Optional[Dict[str, float]] = None,
    use_confidence: bool = True,
) -> float:
    """
    Calculate weight for a single detection.

    Args:
        detection: Detection object
        model_weights: Optional dict of model name -> weight
        use_confidence: If True, multiply by detection confidence

    Returns:
        Weight for this detection
    """
    weight = 1.0

    if model_weights and detection.model_source in model_weights:
        weight = model_weights[detection.model_source]

    if use_confidence:
        weight *= detection.confidence

    return weight


__all__ = [
    "calculate_model_weights",
    "merge_cluster_weighted",
    "merge_cluster_simple",
    "vote_class_majority",
    "vote_class_weighted",
    "get_detection_weight",
]
