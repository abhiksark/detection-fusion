"""
Confidence-based ensemble strategies.

Strategies that use confidence scores for filtering, weighting, and prioritization.
"""

from typing import Dict, List

import numpy as np

from ..core.detection import Detection
from ..utils.metrics import calculate_iou
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, DetectionUtilsMixin


class ConfidenceThresholdVoting(BaseStrategy):
    """Voting strategy with dynamic confidence thresholds.

    Calculates adaptive confidence thresholds per model based on
    the median confidence, then applies majority voting.
    """

    metadata = StrategyMetadata(
        name="confidence_threshold",
        category="confidence_based",
        description="Adaptive confidence thresholds per model",
    )

    def __init__(
        self,
        iou_threshold: float = 0.5,
        base_confidence: float = 0.5,
        adaptive_threshold: bool = True,
        **kwargs,
    ):
        super().__init__(iou_threshold, **kwargs)
        self.base_confidence = base_confidence
        self.adaptive_threshold = adaptive_threshold

    @property
    def name(self) -> str:
        return "confidence_threshold"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using adaptive confidence thresholds.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override adaptive_threshold with 'adaptive_threshold' key

        Returns:
            List of merged detections after confidence filtering
        """
        adaptive = kwargs.get("adaptive_threshold", self.adaptive_threshold)

        # Calculate thresholds
        if adaptive:
            thresholds = self._calculate_adaptive_thresholds(detections)
        else:
            thresholds = {model: self.base_confidence for model in detections.keys()}

        # Filter detections by confidence
        filtered_detections = {}
        for model, dets in detections.items():
            threshold = thresholds[model]
            filtered_detections[model] = [det for det in dets if det.confidence >= threshold]

        # Apply majority voting to filtered detections
        from .voting import MajorityVoting

        voter = MajorityVoting(self.iou_threshold, min_votes=2)
        return voter.merge(filtered_detections)

    def _calculate_adaptive_thresholds(
        self, detections: Dict[str, List[Detection]]
    ) -> Dict[str, float]:
        """Calculate adaptive confidence thresholds per model."""
        thresholds = {}

        for model, dets in detections.items():
            if not dets:
                thresholds[model] = self.base_confidence
                continue

            confidences = [det.confidence for det in dets]
            median_conf = np.median(confidences)
            # Clamp threshold to reasonable range
            thresholds[model] = np.clip(median_conf, 0.2, 0.8)

        return thresholds


class ConfidenceWeightedNMS(BaseStrategy, DetectionUtilsMixin, BoxMergingMixin):
    """NMS with confidence-based box regression.

    Applies NMS and merges overlapping detections using
    confidence-weighted box averaging with configurable power scaling.
    """

    metadata = StrategyMetadata(
        name="confidence_weighted_nms",
        category="confidence_based",
        description="NMS with confidence-weighted box regression",
    )

    def __init__(self, iou_threshold: float = 0.5, confidence_power: float = 2.0, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.confidence_power = confidence_power

    @property
    def name(self) -> str:
        return "confidence_weighted_nms"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using confidence-weighted NMS.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override confidence_power with 'confidence_power' key

        Returns:
            List of merged detections after confidence-weighted NMS
        """
        confidence_power = kwargs.get("confidence_power", self.confidence_power)

        # Flatten and group by class
        all_detections = self.flatten_detections(detections)
        if not all_detections:
            return []

        class_groups = self.group_by_class(all_detections)

        # Apply confidence-weighted NMS per class
        merged_detections = []
        for class_id, class_dets in class_groups.items():
            if class_dets:
                nms_results = self._confidence_weighted_nms(class_dets, confidence_power)
                merged_detections.extend(nms_results)

        return merged_detections

    def _confidence_weighted_nms(
        self, detections: List[Detection], confidence_power: float
    ) -> List[Detection]:
        """Apply confidence-weighted NMS to a list of detections."""
        if not detections:
            return []

        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        kept_detections = []

        while sorted_dets:
            current = sorted_dets.pop(0)

            # Find overlapping detections
            overlapping = []
            remaining = []
            for det in sorted_dets:
                iou = calculate_iou(current.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)

            # Merge overlapping detections
            if overlapping:
                all_overlapping = [current] + overlapping
                weights = [det.confidence**confidence_power for det in all_overlapping]
                merged = self.merge_cluster(
                    all_overlapping,
                    weights=weights,
                    class_id=current.class_id,
                    source=f"conf_weighted_nms_{len(all_overlapping)}",
                )
                # Override with max confidence
                merged = Detection(
                    class_id=merged.class_id,
                    x=merged.x,
                    y=merged.y,
                    w=merged.w,
                    h=merged.h,
                    confidence=max(det.confidence for det in all_overlapping),
                    model_source=merged.model_source,
                )
                kept_detections.append(merged)
            else:
                kept_detections.append(current)

            sorted_dets = remaining

        return kept_detections


class HighConfidenceFirst(BaseStrategy, DetectionUtilsMixin):
    """Strategy that prioritizes high-confidence detections.

    First accepts all high-confidence detections, then adds
    medium-confidence detections that don't overlap spatially.
    """

    metadata = StrategyMetadata(
        name="high_confidence_first",
        category="confidence_based",
        description="Prioritizes high-confidence detections",
    )

    def __init__(
        self,
        iou_threshold: float = 0.5,
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(iou_threshold, **kwargs)
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

    @property
    def name(self) -> str:
        return "high_confidence_first"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections prioritizing high confidence.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override thresholds with 'high_conf_threshold'/'low_conf_threshold'

        Returns:
            List of merged detections with priority given to high confidence
        """
        high_thresh = kwargs.get("high_conf_threshold", self.high_conf_threshold)
        low_thresh = kwargs.get("low_conf_threshold", self.low_conf_threshold)

        # Flatten all detections
        all_detections = self.flatten_detections(detections)
        if not all_detections:
            return []

        # Separate by confidence levels
        high_conf = self.filter_by_confidence(all_detections, min_confidence=high_thresh)
        medium_conf = [d for d in all_detections if low_thresh <= d.confidence < high_thresh]

        merged_detections = []
        used_positions = []

        # First, add all high-confidence detections
        for det in high_conf:
            merged_detections.append(det)
            used_positions.append((det.x, det.y))

        # Then, add medium-confidence detections that don't overlap
        for det in medium_conf:
            overlaps = False
            for pos in used_positions:
                distance = np.sqrt((det.x - pos[0]) ** 2 + (det.y - pos[1]) ** 2)
                if distance < 0.1:  # Spatial overlap threshold
                    overlaps = True
                    break

            if not overlaps:
                merged_detections.append(det)
                used_positions.append((det.x, det.y))

        return merged_detections
