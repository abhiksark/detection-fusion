"""
Adaptive ensemble strategies.

Strategies that adapt their behavior based on context (object size, density, scale).
"""

from typing import Dict, List

import numpy as np

from ..core.detection import Detection
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, ClassVotingMixin, ClusteringMixin, DetectionUtilsMixin
from .voting import MajorityVoting


class AdaptiveThresholdStrategy(BaseStrategy):
    """Strategy that adapts IoU threshold based on object size.

    Uses a lower IoU threshold for small objects and a higher
    threshold for large objects to account for localization variance.
    """

    metadata = StrategyMetadata(
        name="adaptive_threshold",
        category="adaptive",
        description="Different IoU thresholds for object sizes",
    )

    def __init__(
        self,
        small_threshold: float = 0.3,
        large_threshold: float = 0.7,
        size_cutoff: float = 0.05,
        **kwargs,
    ):
        super().__init__(iou_threshold=0.5, **kwargs)
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        self.size_cutoff = size_cutoff

    @property
    def name(self) -> str:
        return "adaptive_threshold"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections with size-adaptive thresholds.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override thresholds with same-named keys

        Returns:
            List of merged detections with size-specific processing
        """
        small_thresh = kwargs.get("small_threshold", self.small_threshold)
        large_thresh = kwargs.get("large_threshold", self.large_threshold)
        size_cutoff = kwargs.get("size_cutoff", self.size_cutoff)

        # Separate detections by size
        small_objects = {model: [] for model in detections}
        large_objects = {model: [] for model in detections}

        for model, dets in detections.items():
            for det in dets:
                if det.w * det.h < size_cutoff:
                    small_objects[model].append(det)
                else:
                    large_objects[model].append(det)

        # Apply different strategies based on size
        small_voter = MajorityVoting(small_thresh, min_votes=2)
        large_voter = MajorityVoting(large_thresh, min_votes=2)

        small_results = small_voter.merge(small_objects)
        large_results = large_voter.merge(large_objects)

        return small_results + large_results


class DensityAdaptiveStrategy(BaseStrategy, DetectionUtilsMixin):
    """Strategy that adapts based on detection density in spatial regions.

    Uses aggressive NMS in high-density regions and conservative
    majority voting in low-density regions.
    """

    metadata = StrategyMetadata(
        name="density_adaptive",
        category="adaptive",
        description="Context-aware processing for detection density",
    )

    def __init__(
        self,
        iou_threshold: float = 0.5,
        grid_size: int = 5,
        high_density_threshold: int = 10,
        **kwargs,
    ):
        super().__init__(iou_threshold, **kwargs)
        self.grid_size = grid_size
        self.high_density_threshold = high_density_threshold

    @property
    def name(self) -> str:
        return "density_adaptive"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections with density-adaptive processing.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override thresholds with same-named keys

        Returns:
            List of merged detections with density-specific processing
        """
        grid_size = kwargs.get("grid_size", self.grid_size)
        density_thresh = kwargs.get("high_density_threshold", self.high_density_threshold)

        # Flatten all detections
        all_detections = self.flatten_detections(detections)
        if not all_detections:
            return []

        # Calculate density map
        density_map = self._calculate_density_map(all_detections, grid_size)

        # Separate detections by density regions
        high_density_dets = {model: [] for model in detections}
        low_density_dets = {model: [] for model in detections}

        for model, dets in detections.items():
            for det in dets:
                grid_x = min(int(det.x * grid_size), grid_size - 1)
                grid_y = min(int(det.y * grid_size), grid_size - 1)

                if density_map[grid_y, grid_x] > density_thresh:
                    high_density_dets[model].append(det)
                else:
                    low_density_dets[model].append(det)

        # Apply different strategies - import here to avoid circular
        from .nms import NMSStrategy

        # High density: aggressive NMS
        nms_strategy = NMSStrategy(iou_threshold=0.3, score_threshold=0.2)
        high_results = nms_strategy.merge(high_density_dets)

        # Low density: conservative majority voting
        majority_strategy = MajorityVoting(iou_threshold=0.7, min_votes=2)
        low_results = majority_strategy.merge(low_density_dets)

        return high_results + low_results

    def _calculate_density_map(self, detections: List[Detection], grid_size: int) -> np.ndarray:
        """Calculate detection density in a spatial grid."""
        density_map = np.zeros((grid_size, grid_size))

        for det in detections:
            grid_x = min(int(det.x * grid_size), grid_size - 1)
            grid_y = min(int(det.y * grid_size), grid_size - 1)
            density_map[grid_y, grid_x] += 1

        return density_map


class MultiScaleStrategy(BaseStrategy):
    """Strategy that handles multiple scales of objects differently.

    Applies scale-specific voting strategies with different IoU thresholds
    and vote requirements for tiny, small, medium, and large objects.
    """

    metadata = StrategyMetadata(
        name="multi_scale", category="adaptive", description="Scale-specific processing"
    )

    def __init__(self, iou_threshold: float = 0.5, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.scale_thresholds = [0.01, 0.05, 0.15]  # Tiny, small, medium cutoffs

    @property
    def name(self) -> str:
        return "multi_scale"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections with scale-specific processing.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override scale_thresholds with 'scale_thresholds' key

        Returns:
            List of merged detections with scale-specific processing
        """
        scale_thresholds = kwargs.get("scale_thresholds", self.scale_thresholds)

        # Group detections by scale
        scale_groups = self._group_by_scale(detections, scale_thresholds)

        # Scale-specific strategy configurations
        scale_configs = {
            "tiny": (0.2, 1),  # Very permissive
            "small": (0.3, 2),  # Moderate
            "medium": (0.5, 2),  # Standard
            "large": (0.7, 3),  # Strict
        }

        merged_results = []
        for scale, scale_dets in scale_groups.items():
            if not any(scale_dets.values()):
                continue

            iou_thresh, min_votes = scale_configs[scale]
            strategy = MajorityVoting(iou_threshold=iou_thresh, min_votes=min_votes)
            merged_results.extend(strategy.merge(scale_dets))

        return merged_results

    def _group_by_scale(
        self, detections: Dict[str, List[Detection]], thresholds: List[float]
    ) -> Dict[str, Dict[str, List[Detection]]]:
        """Group detections by object scale."""
        scale_groups = {
            scale: {model: [] for model in detections}
            for scale in ["tiny", "small", "medium", "large"]
        }

        for model, dets in detections.items():
            for det in dets:
                area = det.w * det.h

                if area < thresholds[0]:
                    scale_groups["tiny"][model].append(det)
                elif area < thresholds[1]:
                    scale_groups["small"][model].append(det)
                elif area < thresholds[2]:
                    scale_groups["medium"][model].append(det)
                else:
                    scale_groups["large"][model].append(det)

        return scale_groups


class ConsensusRankingStrategy(BaseStrategy, ClusteringMixin, BoxMergingMixin, ClassVotingMixin):
    """Strategy based on ranking consensus across models.

    Combines model ranking (detection order by confidence within model)
    with confidence scores to determine weights for voting and merging.
    """

    metadata = StrategyMetadata(
        name="consensus_ranking",
        category="adaptive",
        description="Combines model ranking with confidence",
    )

    def __init__(self, iou_threshold: float = 0.5, rank_weight: float = 0.5, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.rank_weight = rank_weight

    @property
    def name(self) -> str:
        return "consensus_ranking"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using rank-based consensus.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override rank_weight with 'rank_weight' key

        Returns:
            List of merged detections with rank-weighted consensus
        """
        rank_weight = kwargs.get("rank_weight", self.rank_weight)

        # Compute per-model rankings (rank 1 = highest confidence)
        model_rankings = {}
        for model, dets in detections.items():
            sorted_dets = sorted(dets, key=lambda x: x.confidence, reverse=True)
            model_rankings[model] = {det: rank + 1 for rank, det in enumerate(sorted_dets)}

        # Cluster overlapping detections using mixin
        all_detections = self._flatten(detections)
        clusters = self.cluster_detections(all_detections)

        # Rank-based consensus for each cluster
        merged_detections = []
        for cluster in clusters:
            merged_det = self._rank_based_consensus(cluster, model_rankings, rank_weight)
            merged_detections.append(merged_det)

        return merged_detections

    def _rank_based_consensus(
        self,
        cluster: List[Detection],
        model_rankings: Dict[str, Dict[Detection, int]],
        rank_weight: float,
    ) -> Detection:
        """Apply rank-based consensus to a cluster."""
        if len(cluster) == 1:
            return cluster[0]

        # Calculate consensus score for each detection
        consensus_scores = []
        for det in cluster:
            rank = model_rankings.get(det.model_source, {}).get(det, float("inf"))
            rank_score = 1.0 / rank if rank != float("inf") else 0.0
            score = rank_weight * rank_score + (1 - rank_weight) * det.confidence
            consensus_scores.append(score)

        # Normalize weights
        weights = np.array(consensus_scores)
        weights = weights / weights.sum()

        # Use mixins for voting and merging
        voted_class = self.vote_weighted(cluster, weights.tolist())

        return self.merge_cluster(
            cluster,
            weights=weights.tolist(),
            class_id=voted_class,
            source=f"consensus_rank_{len(cluster)}",
        )
