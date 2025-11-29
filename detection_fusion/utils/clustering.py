"""
Clustering utilities for grouping overlapping detections.

This module provides the canonical clustering implementations that replace
the duplicated _cluster_detections() methods across strategy files.
"""

from typing import TYPE_CHECKING, List

from .metrics import calculate_iou

if TYPE_CHECKING:
    from ..core.detection import Detection


def cluster_by_iou(
    detections: "List[Detection]", iou_threshold: float = 0.5, same_class_only: bool = True
) -> "List[List[Detection]]":
    """
    Cluster detections by IoU overlap using greedy algorithm.

    Groups overlapping detections into clusters. A detection is added to a
    cluster if it has IoU >= threshold with any detection already in the cluster.

    This is the canonical implementation that replaces the 6 duplicated versions
    across voting.py, probabilistic.py, distance_based.py, and adaptive.py.

    Args:
        detections: List of Detection objects to cluster
        iou_threshold: Minimum IoU for two detections to be in same cluster
        same_class_only: If True, only cluster detections of the same class

    Returns:
        List of clusters, where each cluster is a list of Detection objects
    """
    if not detections:
        return []

    clusters: List[List["Detection"]] = []
    used: set = set()

    for i, det1 in enumerate(detections):
        if i in used:
            continue

        cluster = [det1]
        used.add(i)

        for j, det2 in enumerate(detections):
            if j <= i or j in used:
                continue

            # Skip if class mismatch and same_class_only is True
            if same_class_only and det1.class_id != det2.class_id:
                continue

            # Check if det2 overlaps with any detection in cluster
            for cluster_det in cluster:
                if same_class_only and cluster_det.class_id != det2.class_id:
                    continue

                iou = calculate_iou(cluster_det.bbox, det2.bbox)
                if iou >= iou_threshold:
                    cluster.append(det2)
                    used.add(j)
                    break

        clusters.append(cluster)

    return clusters


def cluster_by_center_distance(
    detections: "List[Detection]", distance_threshold: float = 0.1, same_class_only: bool = True
) -> "List[List[Detection]]":
    """
    Cluster detections by center point distance.

    Groups detections whose center points are within a distance threshold.
    Useful for cases where IoU may not be appropriate.

    Args:
        detections: List of Detection objects to cluster
        distance_threshold: Maximum distance between centers (normalized coords)
        same_class_only: If True, only cluster detections of the same class

    Returns:
        List of clusters, where each cluster is a list of Detection objects
    """
    import numpy as np

    if not detections:
        return []

    clusters: List[List["Detection"]] = []
    used: set = set()

    for i, det1 in enumerate(detections):
        if i in used:
            continue

        cluster = [det1]
        used.add(i)

        for j, det2 in enumerate(detections):
            if j <= i or j in used:
                continue

            if same_class_only and det1.class_id != det2.class_id:
                continue

            # Check distance with any detection in cluster
            for cluster_det in cluster:
                if same_class_only and cluster_det.class_id != det2.class_id:
                    continue

                dist = np.sqrt((cluster_det.x - det2.x) ** 2 + (cluster_det.y - det2.y) ** 2)
                if dist <= distance_threshold:
                    cluster.append(det2)
                    used.add(j)
                    break

        clusters.append(cluster)

    return clusters


def flatten_detections(detections_by_model: "dict") -> "List[Detection]":
    """
    Flatten model-keyed detections dict into a single list.

    Args:
        detections_by_model: Dict mapping model names to lists of detections

    Returns:
        Single flat list of all detections
    """
    all_detections = []
    for model_dets in detections_by_model.values():
        all_detections.extend(model_dets)
    return all_detections


__all__ = [
    "cluster_by_iou",
    "cluster_by_center_distance",
    "flatten_detections",
]
