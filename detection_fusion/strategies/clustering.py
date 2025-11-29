"""
Clustering-based ensemble strategies.

Strategies that use spatial clustering to group and merge detections.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

from ..core.detection import Detection
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, ClassVotingMixin, DetectionUtilsMixin, ModelWeightsMixin


class DBSCANClustering(
    BaseStrategy, ModelWeightsMixin, BoxMergingMixin, ClassVotingMixin, DetectionUtilsMixin
):
    """DBSCAN clustering-based ensemble strategy.

    Uses density-based spatial clustering to group nearby detections
    by their center points, then merges each cluster into a single detection.
    """

    metadata = StrategyMetadata(
        name="dbscan",
        category="clustering",
        description="Density-based spatial clustering of detections",
    )

    def __init__(self, eps: float = 0.1, min_samples: int = 2, **kwargs):
        super().__init__(iou_threshold=0.5, **kwargs)  # IoU not used in DBSCAN
        self.eps = eps
        self.min_samples = min_samples

    @property
    def name(self) -> str:
        return "dbscan"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using DBSCAN clustering.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override eps/min_samples with same-named keys

        Returns:
            List of merged detections, one per cluster
        """
        eps = kwargs.get("eps", self.eps)
        min_samples = kwargs.get("min_samples", self.min_samples)

        # Flatten all detections
        all_detections = self.flatten_detections(detections)
        if not all_detections:
            return []

        # Calculate model weights for weighted merging
        self.compute_model_weights(detections)

        # Extract centers for clustering
        centers = np.array([[det.x, det.y] for det in all_detections])

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)

        # Group detections by cluster
        cluster_groups = defaultdict(list)
        for det, label in zip(all_detections, clustering.labels_):
            if label != -1:  # Ignore noise points
                cluster_groups[label].append(det)

        # Merge each cluster using mixins
        merged_detections = []
        for cluster in cluster_groups.values():
            # Calculate weights: model weight * confidence
            weights = [self.get_model_weight(det.model_source) * det.confidence for det in cluster]

            # Use mixins for voting and merging
            voted_class = self.vote_majority(cluster)
            merged_det = self.merge_cluster(
                cluster, weights=weights, class_id=voted_class, source=f"dbscan_{len(cluster)}"
            )
            merged_detections.append(merged_det)

        return merged_detections
