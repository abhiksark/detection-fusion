"""
Strategy mixins providing shared functionality.

These mixins replace the duplicated methods across strategy files:
- ClusteringMixin: Replaces 6 copies of _cluster_detections()
- ModelWeightsMixin: Replaces 3 copies of _calculate_model_weights()
- BoxMergingMixin: Replaces duplicated box averaging code
- ClassVotingMixin: Replaces duplicated class voting code
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

from ..utils.aggregation import (
    calculate_model_weights,
    merge_cluster_weighted,
    vote_class_majority,
    vote_class_weighted,
)
from ..utils.clustering import cluster_by_center_distance, cluster_by_iou

if TYPE_CHECKING:
    from ..core.detection import Detection


class ClusteringMixin:
    """Mixin providing detection clustering functionality.

    Requires the using class to have an `iou_threshold` attribute
    (typically from BaseStrategy or config).

    This replaces the 6 duplicated _cluster_detections() implementations
    across voting.py, probabilistic.py, distance_based.py, and adaptive.py.
    """

    # These should be defined by the class using this mixin
    iou_threshold: float

    def cluster_detections(
        self, detections: "List[Detection]", same_class_only: bool = True
    ) -> "List[List[Detection]]":
        """Cluster overlapping detections using IoU.

        Args:
            detections: List of Detection objects to cluster
            same_class_only: If True, only cluster same-class detections

        Returns:
            List of clusters (each cluster is a list of Detection objects)
        """
        threshold = getattr(self, "iou_threshold", 0.5)
        return cluster_by_iou(detections, threshold, same_class_only)

    def cluster_by_distance(
        self,
        detections: "List[Detection]",
        distance_threshold: float = 0.1,
        same_class_only: bool = True,
    ) -> "List[List[Detection]]":
        """Cluster detections by center point distance.

        Args:
            detections: List of Detection objects to cluster
            distance_threshold: Maximum distance between centers
            same_class_only: If True, only cluster same-class detections

        Returns:
            List of clusters
        """
        return cluster_by_center_distance(detections, distance_threshold, same_class_only)


class ModelWeightsMixin:
    """Mixin for computing model reliability weights.

    Replaces the 3 duplicated _calculate_model_weights() implementations
    in voting.py, clustering.py, and probabilistic.py.
    """

    # Cache for model weights
    _model_weights: Dict[str, float]

    def compute_model_weights(
        self, detections: "Dict[str, List[Detection]]", method: str = "confidence"
    ) -> Dict[str, float]:
        """Compute normalized weights for each model.

        Args:
            detections: Dict mapping model name to detections
            method: "confidence" (avg conf), "count", or "uniform"

        Returns:
            Dict mapping model name to normalized weight
        """
        weights = calculate_model_weights(detections, method)
        self._model_weights = weights
        return weights

    def get_model_weight(self, model_name: str) -> float:
        """Get cached weight for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Weight value (defaults to 1.0 if not found)
        """
        if not hasattr(self, "_model_weights"):
            return 1.0
        return self._model_weights.get(model_name, 1.0)


class BoxMergingMixin:
    """Mixin for merging bounding boxes from a cluster.

    Provides weighted and simple box merging functionality.
    """

    def merge_cluster(
        self,
        cluster: "List[Detection]",
        weights: Optional[List[float]] = None,
        class_id: Optional[int] = None,
        source: str = "merged",
    ) -> "Detection":
        """Merge a cluster of detections into a single detection.

        Args:
            cluster: List of Detection objects to merge
            weights: Optional weights for each detection
            class_id: Class ID for result (uses majority vote if None)
            source: Model source string for merged detection

        Returns:
            Single merged Detection object
        """
        return merge_cluster_weighted(cluster, weights, class_id, source)

    def merge_cluster_uniform(
        self, cluster: "List[Detection]", source: str = "merged"
    ) -> "Detection":
        """Merge cluster with uniform weights (simple average).

        Args:
            cluster: List of Detection objects to merge
            source: Model source string for merged detection

        Returns:
            Single merged Detection object
        """
        weights = [1.0] * len(cluster)
        return merge_cluster_weighted(cluster, weights, source=source)


class ClassVotingMixin:
    """Mixin for class voting within clusters."""

    def vote_majority(self, cluster: "List[Detection]") -> int:
        """Simple majority voting for class selection.

        Args:
            cluster: List of Detection objects

        Returns:
            Most common class ID in the cluster
        """
        return vote_class_majority(cluster)

    def vote_weighted(self, cluster: "List[Detection]", weights: List[float]) -> int:
        """Weighted voting for class selection.

        Args:
            cluster: List of Detection objects
            weights: Weight for each detection

        Returns:
            Class ID with highest weighted vote
        """
        return vote_class_weighted(cluster, weights)

    def vote_by_confidence(self, cluster: "List[Detection]") -> int:
        """Vote on class weighted by detection confidence.

        Args:
            cluster: List of Detection objects

        Returns:
            Class ID with highest confidence-weighted vote
        """
        weights = [d.confidence for d in cluster]
        return vote_class_weighted(cluster, weights)


class DetectionUtilsMixin:
    """Mixin providing common detection utilities."""

    def flatten_detections(self, detections: "Dict[str, List[Detection]]") -> "List[Detection]":
        """Flatten model-keyed detections into a single list.

        Args:
            detections: Dict mapping model names to detection lists

        Returns:
            Single flat list of all detections
        """
        all_dets = []
        for model_dets in detections.values():
            all_dets.extend(model_dets)
        return all_dets

    def filter_by_confidence(
        self, detections: "List[Detection]", min_confidence: float = 0.0
    ) -> "List[Detection]":
        """Filter detections by minimum confidence.

        Args:
            detections: List of Detection objects
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d.confidence >= min_confidence]

    def group_by_class(self, detections: "List[Detection]") -> "Dict[int, List[Detection]]":
        """Group detections by class ID.

        Args:
            detections: List of Detection objects

        Returns:
            Dict mapping class_id to list of detections
        """
        groups: Dict[int, List["Detection"]] = defaultdict(list)
        for det in detections:
            groups[det.class_id].append(det)
        return dict(groups)

    def group_by_model(self, detections: "List[Detection]") -> "Dict[str, List[Detection]]":
        """Group detections by source model.

        Args:
            detections: List of Detection objects

        Returns:
            Dict mapping model_source to list of detections
        """
        groups: Dict[str, List["Detection"]] = defaultdict(list)
        for det in detections:
            groups[det.model_source].append(det)
        return dict(groups)


# Combined mixin for strategies that need all functionality
class FullStrategyMixin(
    ClusteringMixin, ModelWeightsMixin, BoxMergingMixin, ClassVotingMixin, DetectionUtilsMixin
):
    """Combined mixin providing all strategy utilities.

    Use this when a strategy needs access to all shared functionality.
    """

    pass


__all__ = [
    "ClusteringMixin",
    "ModelWeightsMixin",
    "BoxMergingMixin",
    "ClassVotingMixin",
    "DetectionUtilsMixin",
    "FullStrategyMixin",
]
