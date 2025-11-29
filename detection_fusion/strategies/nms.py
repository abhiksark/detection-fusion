"""
NMS-based ensemble strategies.

Strategies that use Non-Maximum Suppression to combine detections.
"""

from typing import Dict, List

import torch
import torchvision.ops as ops

from ..core.detection import Detection
from ..utils.metrics import calculate_iou
from .base import BaseStrategy, StrategyMetadata
from .mixins import BoxMergingMixin, DetectionUtilsMixin


class NMSStrategy(BaseStrategy, DetectionUtilsMixin):
    """Standard Non-Maximum Suppression strategy.

    Applies torchvision NMS per class to suppress overlapping detections,
    keeping only the highest confidence detection in each overlap group.
    """

    metadata = StrategyMetadata(
        name="nms", category="nms", description="Standard Non-Maximum Suppression per class"
    )

    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.1, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.score_threshold = score_threshold

    @property
    def name(self) -> str:
        return "nms"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using NMS.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override score_threshold with 'score_threshold' key

        Returns:
            List of merged detections after NMS
        """
        score_threshold = kwargs.get("score_threshold", self.score_threshold)

        # Flatten and filter by confidence
        all_detections = self.filter_by_confidence(
            self.flatten_detections(detections), min_confidence=score_threshold
        )

        if not all_detections:
            return []

        # Group by class and apply NMS per class
        class_groups = self.group_by_class(all_detections)

        merged_detections = []
        for class_id, class_dets in class_groups.items():
            if not class_dets:
                continue

            # Convert to tensors
            boxes = torch.tensor([det.xyxy for det in class_dets])
            scores = torch.tensor([det.confidence for det in class_dets])

            # Apply NMS
            keep_indices = ops.nms(boxes, scores, self.iou_threshold)

            # Keep selected detections
            for idx in keep_indices:
                merged_detections.append(class_dets[idx])

        return merged_detections


class AffirmativeNMS(BaseStrategy, DetectionUtilsMixin, BoxMergingMixin):
    """NMS that requires agreement from multiple models.

    Applies NMS and then filters to keep only detections where
    at least `min_models` different models agree on the detection.
    """

    metadata = StrategyMetadata(
        name="affirmative_nms",
        category="nms",
        description="NMS requiring agreement from multiple models",
    )

    def __init__(self, iou_threshold: float = 0.5, min_models: int = 2, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.min_models = min_models

    @property
    def name(self) -> str:
        return f"affirmative_nms_{self.min_models}"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections using affirmative NMS.

        Args:
            detections: Dict mapping model names to detection lists
            **kwargs: Override min_models with 'min_models' key

        Returns:
            List of merged detections where multiple models agree
        """
        min_models = kwargs.get("min_models", self.min_models)
        merged_detections = []

        # Get all unique classes
        all_dets = self.flatten_detections(detections)
        all_classes = set(det.class_id for det in all_dets)

        # Process each class
        for class_id in all_classes:
            # Group detections by model for this class
            class_by_model = {}
            for model, model_dets in detections.items():
                class_dets = [d for d in model_dets if d.class_id == class_id]
                if class_dets:
                    class_by_model[model] = class_dets

            # Skip if not enough models detected this class
            if len(class_by_model) < min_models:
                continue

            # Collect all detections for this class
            all_class_dets = [d for dets in class_by_model.values() for d in dets]
            if not all_class_dets:
                continue

            # Apply NMS
            boxes = torch.tensor([det.xyxy for det in all_class_dets])
            scores = torch.tensor([det.confidence for det in all_class_dets])
            keep_indices = ops.nms(boxes, scores, self.iou_threshold)

            # For each kept detection, check if enough models agree
            for idx in keep_indices:
                base_det = all_class_dets[idx]

                # Find all detections from different models that overlap
                matching_models = {base_det.model_source}
                matching_dets = [base_det]

                for det in all_class_dets:
                    if det != base_det and det.model_source not in matching_models:
                        iou = calculate_iou(base_det.bbox, det.bbox)
                        if iou >= self.iou_threshold:
                            matching_models.add(det.model_source)
                            matching_dets.append(det)

                # Keep if enough models agree - use mixin for merging
                if len(matching_models) >= min_models:
                    merged_det = self.merge_cluster(
                        matching_dets,
                        class_id=class_id,
                        source=f"affirmative_{len(matching_models)}",
                    )
                    merged_detections.append(merged_det)

        return merged_detections
