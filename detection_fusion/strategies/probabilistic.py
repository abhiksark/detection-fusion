"""
Probabilistic ensemble strategies.

Strategies that use probabilistic methods for combining detections.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..core.detection import Detection
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, ClusteringMixin, ModelWeightsMixin


class SoftVoting(BaseStrategy, ClusteringMixin, ModelWeightsMixin, BoxMergingMixin):
    """Soft voting with temperature scaling.

    Uses temperature-scaled probabilities and model weights
    to compute weighted class probabilities for each cluster.
    """

    metadata = StrategyMetadata(
        name="soft_voting",
        category="probabilistic",
        description="Probabilistic voting with temperature scaling",
    )

    def __init__(self, iou_threshold: float = 0.5, temperature: float = 1.0, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "soft_voting"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using soft voting.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override temperature with 'temperature' key

        Returns:
            List of merged detections with probabilistic class selection
        """
        temperature = kwargs.get("temperature", self.temperature)

        # Compute model weights using mixin
        self.compute_model_weights(detections)

        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Cluster overlapping detections using mixin
        clusters = self.cluster_detections(all_detections)

        # Apply soft voting
        merged_detections = []
        for cluster in clusters:
            merged_det = self._soft_vote_cluster(cluster, temperature)
            merged_detections.append(merged_det)

        return merged_detections

    def _soft_vote_cluster(self, cluster: List[Detection], temperature: float) -> Detection:
        """Apply soft voting to a cluster."""
        # Calculate class probabilities with temperature scaling
        class_probs = defaultdict(float)
        total_weight = 0

        for det in cluster:
            model_weight = self.get_model_weight(det.model_source)
            prob = np.exp(det.confidence / temperature)
            weight = model_weight * prob
            class_probs[det.class_id] += weight
            total_weight += weight

        # Normalize probabilities
        if total_weight > 0:
            for class_id in class_probs:
                class_probs[class_id] /= total_weight

        # Select class with highest probability
        voted_class = max(class_probs, key=class_probs.get)
        class_confidence = class_probs[voted_class]

        # Calculate weights for box merging
        weights = [self.get_model_weight(det.model_source) * det.confidence for det in cluster]

        # Use mixin for box merging, but override confidence
        merged_det = self.merge_cluster(
            cluster, weights=weights, class_id=voted_class, source=f"soft_vote_{len(cluster)}"
        )
        # Override confidence with class probability
        return Detection(
            class_id=merged_det.class_id,
            x=merged_det.x,
            y=merged_det.y,
            w=merged_det.w,
            h=merged_det.h,
            confidence=class_confidence,
            model_source=merged_det.model_source,
        )


class BayesianFusion(BaseStrategy, ClusteringMixin, BoxMergingMixin):
    """Bayesian fusion with class priors.

    Uses Bayesian inference to compute posterior class probabilities
    based on detection confidences and learned class priors.
    """

    metadata = StrategyMetadata(
        name="bayesian",
        category="probabilistic",
        description="Bayesian inference with learned class priors",
    )

    def __init__(
        self, iou_threshold: float = 0.5, class_priors: Optional[Dict[int, float]] = None, **kwargs
    ):
        super().__init__(iou_threshold, **kwargs)
        self.class_priors = class_priors

    @property
    def name(self) -> str:
        return "bayesian"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using Bayesian fusion.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override class_priors with 'class_priors' key

        Returns:
            List of merged detections with Bayesian posteriors
        """
        # Flatten all detections
        all_detections = self._flatten(detections)
        if not all_detections:
            return []

        # Calculate class priors if not provided
        class_priors = kwargs.get("class_priors", self.class_priors)
        if class_priors is None:
            class_priors = self._calculate_class_priors(all_detections)

        # Cluster overlapping detections using mixin
        clusters = self.cluster_detections(all_detections)

        # Apply Bayesian fusion
        merged_detections = []
        for cluster in clusters:
            merged_det = self._bayesian_merge_cluster(cluster, class_priors)
            merged_detections.append(merged_det)

        return merged_detections

    def _calculate_class_priors(self, detections: List[Detection]) -> Dict[int, float]:
        """Calculate class priors from detections."""
        class_counts = Counter([det.class_id for det in detections])
        total = len(detections)
        return {cls: count / total for cls, count in class_counts.items()}

    def _bayesian_merge_cluster(
        self, cluster: List[Detection], class_priors: Dict[int, float]
    ) -> Detection:
        """Apply Bayesian fusion to merge a cluster."""
        # Get all possible classes in cluster
        possible_classes = set(det.class_id for det in cluster)

        # Calculate posterior probability for each class
        class_posteriors = {}

        for class_id in possible_classes:
            # Prior probability
            prior = class_priors.get(class_id, 0.01)

            # Calculate likelihood
            likelihood = 1.0
            for det in cluster:
                if det.class_id == class_id:
                    likelihood *= det.confidence
                else:
                    likelihood *= (1 - det.confidence) * 0.1

            # Posterior (unnormalized)
            class_posteriors[class_id] = prior * likelihood

        # Normalize posteriors
        total_posterior = sum(class_posteriors.values())
        if total_posterior > 0:
            for class_id in class_posteriors:
                class_posteriors[class_id] /= total_posterior

        # Select class with highest posterior
        voted_class = max(class_posteriors, key=class_posteriors.get)
        confidence = class_posteriors[voted_class]

        # Use mixin for box merging
        merged_det = self.merge_cluster(
            cluster, class_id=voted_class, source=f"bayesian_{len(cluster)}"
        )

        # Override confidence with posterior
        return Detection(
            class_id=merged_det.class_id,
            x=merged_det.x,
            y=merged_det.y,
            w=merged_det.w,
            h=merged_det.h,
            confidence=confidence,
            model_source=merged_det.model_source,
        )
