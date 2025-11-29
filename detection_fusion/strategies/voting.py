"""
Voting-based ensemble strategies.

Strategies that use voting mechanisms to combine detections from multiple models.
"""

from typing import Dict, List

import numpy as np

from ..core.detection import Detection
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, ClassVotingMixin, ClusteringMixin, ModelWeightsMixin
from .params import ParamSchema, ParamSpec

# Parameter schema for voting strategies
MAJORITY_VOTING_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for detection overlap",
        ),
        ParamSpec(
            name="min_votes",
            param_type="int",
            default=2,
            min_value=1,
            description="Minimum models that must agree",
        ),
    ]
)

WEIGHTED_VOTING_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for detection overlap",
        ),
        ParamSpec(
            name="use_model_weights",
            param_type="bool",
            default=True,
            description="Whether to use model weights in voting",
        ),
    ]
)


class MajorityVoting(BaseStrategy, ClusteringMixin, BoxMergingMixin, ClassVotingMixin):
    """Majority voting strategy for ensemble.

    Clusters overlapping detections and keeps those where at least
    `min_votes` models agree. Final detection uses averaged boxes
    and majority-voted class.
    """

    metadata = StrategyMetadata(
        name="majority_vote",
        category="voting",
        description="Keep detections where multiple models agree",
        params_schema=MAJORITY_VOTING_SCHEMA,
    )

    def __init__(self, iou_threshold: float = 0.5, min_votes: int = 2, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.min_votes = min_votes

    @property
    def name(self) -> str:
        return f"majority_vote_{self.min_votes}"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using majority voting.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override min_votes with 'min_votes' key

        Returns:
            List of merged detections meeting vote threshold
        """
        min_votes = kwargs.get("min_votes", self.min_votes)

        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Cluster overlapping detections (uses ClusteringMixin)
        clusters = self.cluster_detections(all_detections)

        # Vote on each cluster that meets threshold
        merged_detections = []
        for cluster in clusters:
            if len(cluster) >= min_votes:
                # Use mixins for voting and merging
                voted_class = self.vote_majority(cluster)
                merged_det = self.merge_cluster(
                    cluster, class_id=voted_class, source=f"ensemble_vote_{len(cluster)}"
                )
                merged_detections.append(merged_det)

        return merged_detections


class WeightedVoting(
    BaseStrategy, ClusteringMixin, ModelWeightsMixin, BoxMergingMixin, ClassVotingMixin
):
    """Weighted voting strategy using confidence scores.

    Weights detections by model performance (average confidence)
    and individual detection confidence for voting and averaging.
    """

    metadata = StrategyMetadata(
        name="weighted_vote",
        category="voting",
        description="Weight detections by model and detection confidence",
        params_schema=WEIGHTED_VOTING_SCHEMA,
    )

    def __init__(self, iou_threshold: float = 0.5, use_model_weights: bool = True, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.use_model_weights = use_model_weights
        self._model_weights: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "weighted_vote"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using weighted voting.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override use_model_weights with 'use_model_weights' key

        Returns:
            List of merged detections with weighted boxes
        """
        use_model_weights = kwargs.get("use_model_weights", self.use_model_weights)

        # Calculate model weights if needed (uses ModelWeightsMixin)
        if use_model_weights:
            self.compute_model_weights(detections)

        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Cluster overlapping detections (uses ClusteringMixin)
        clusters = self.cluster_detections(all_detections)

        # Apply weighted voting to each cluster
        merged_detections = []
        for cluster in clusters:
            merged_det = self._weighted_vote_on_cluster(cluster, use_model_weights)
            merged_detections.append(merged_det)

        return merged_detections

    def _weighted_vote_on_cluster(
        self, cluster: List[Detection], use_model_weights: bool
    ) -> Detection:
        """Apply weighted voting to a cluster.

        Args:
            cluster: List of detections to merge
            use_model_weights: Whether to include model weights

        Returns:
            Single merged detection
        """
        # Calculate per-detection weights
        if use_model_weights:
            weights = [self.get_model_weight(det.model_source) * det.confidence for det in cluster]
        else:
            weights = [det.confidence for det in cluster]

        # Normalize weights
        weights_arr = np.array(weights)
        if weights_arr.sum() > 0:
            weights_arr = weights_arr / weights_arr.sum()
        else:
            weights_arr = np.ones(len(cluster)) / len(cluster)

        # Weighted vote on class (uses ClassVotingMixin)
        voted_class = self.vote_weighted(cluster, weights_arr.tolist())

        # Weighted merge (uses BoxMergingMixin)
        return self.merge_cluster(
            cluster,
            weights=weights_arr.tolist(),
            class_id=voted_class,
            source=f"ensemble_weighted_{len(cluster)}",
        )
