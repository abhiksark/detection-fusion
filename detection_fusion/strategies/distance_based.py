"""
Distance-based ensemble strategies.

Strategies that use spatial distance for weighting and clustering.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from ..core.detection import Detection
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, ClassVotingMixin, ClusteringMixin


class DistanceWeightedVoting(BaseStrategy, ClusteringMixin, BoxMergingMixin, ClassVotingMixin):
    """Voting strategy that weights by spatial distance between detections.

    Detections closer to the cluster centroid receive higher weights,
    combined with confidence weights for voting and box averaging.
    """

    metadata = StrategyMetadata(
        name="distance_weighted",
        category="distance_based",
        description="Weights by spatial distance to cluster centroid",
    )

    def __init__(self, iou_threshold: float = 0.5, distance_weight: float = 1.0, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.distance_weight = distance_weight

    @property
    def name(self) -> str:
        return "distance_weighted"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using distance-weighted voting.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override distance_weight with 'distance_weight' key

        Returns:
            List of merged detections with distance-weighted boxes
        """
        distance_weight = kwargs.get("distance_weight", self.distance_weight)

        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Cluster overlapping detections using mixin
        clusters = self.cluster_detections(all_detections)

        # Apply distance-weighted voting
        merged_detections = []
        for cluster in clusters:
            merged_det = self._distance_weighted_vote(cluster, distance_weight)
            merged_detections.append(merged_det)

        return merged_detections

    def _distance_weighted_vote(
        self, cluster: List[Detection], distance_weight: float
    ) -> Detection:
        """Apply distance-weighted voting to a cluster."""
        if len(cluster) == 1:
            return cluster[0]

        # Calculate centroid
        centers = np.array([[det.x, det.y] for det in cluster])
        centroid = np.mean(centers, axis=0)

        # Calculate distance weights (closer to centroid = higher weight)
        distances = euclidean_distances(centers, centroid.reshape(1, -1)).flatten()
        max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
        distance_weights = 1.0 - (distances / max_distance)

        # Combine with confidence weights
        confidence_weights = np.array([det.confidence for det in cluster])
        final_weights = (distance_weights * distance_weight + confidence_weights) / 2
        final_weights = final_weights / final_weights.sum()

        # Vote on class using mixin
        voted_class = self.vote_weighted(cluster, final_weights.tolist())

        # Use mixin for box merging
        return self.merge_cluster(
            cluster,
            weights=final_weights.tolist(),
            class_id=voted_class,
            source=f"distance_weighted_{len(cluster)}",
        )


class CentroidClustering(BaseStrategy, BoxMergingMixin, ClassVotingMixin):
    """Clustering strategy based on detection centroids.

    Uses agglomerative clustering on detection center points
    to group nearby detections before merging.
    """

    metadata = StrategyMetadata(
        name="centroid_clustering",
        category="distance_based",
        description="Agglomerative clustering based on detection centers",
    )

    def __init__(self, distance_threshold: float = 0.1, min_cluster_size: int = 2, **kwargs):
        super().__init__(iou_threshold=0.5, **kwargs)  # IoU not used
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    @property
    def name(self) -> str:
        return "centroid_clustering"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using centroid-based clustering.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override distance_threshold/min_cluster_size

        Returns:
            List of merged detections, one per cluster
        """
        distance_threshold = kwargs.get("distance_threshold", self.distance_threshold)
        min_cluster_size = kwargs.get("min_cluster_size", self.min_cluster_size)

        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Agglomerative clustering based on distance
        clusters = self._agglomerative_clustering(all_detections, distance_threshold)

        # Merge each cluster using mixins
        merged_detections = []
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                voted_class = self.vote_majority(cluster)
                merged_det = self.merge_cluster(
                    cluster, class_id=voted_class, source=f"centroid_cluster_{len(cluster)}"
                )
                merged_detections.append(merged_det)

        return merged_detections

    def _agglomerative_clustering(
        self, detections: List[Detection], distance_threshold: float
    ) -> List[List[Detection]]:
        """Simple agglomerative clustering based on centroid distance."""
        clusters = [[det] for det in detections]

        while True:
            # Find closest pair of clusters
            min_distance = float("inf")
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate centroid distance between clusters
                    centroid_i = np.mean([[det.x, det.y] for det in clusters[i]], axis=0)
                    centroid_j = np.mean([[det.x, det.y] for det in clusters[j]], axis=0)

                    distance = np.linalg.norm(centroid_i - centroid_j)

                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j

            # Stop if minimum distance exceeds threshold
            if min_distance > distance_threshold:
                break

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

        return clusters
