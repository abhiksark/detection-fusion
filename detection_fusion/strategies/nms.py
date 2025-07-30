import torch
import torchvision.ops as ops
from typing import List, Dict
from collections import defaultdict

from .base import BaseStrategy
from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class NMSStrategy(BaseStrategy):
    """Standard Non-Maximum Suppression strategy."""
    
    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.1):
        super().__init__(iou_threshold)
        self.score_threshold = score_threshold
    
    @property
    def name(self) -> str:
        return "nms"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for det in all_detections:
            if det.confidence >= self.score_threshold:
                class_groups[det.class_id].append(det)
        
        # Apply NMS per class
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


class AffirmativeNMS(BaseStrategy):
    """NMS that requires agreement from multiple models."""
    
    def __init__(self, iou_threshold: float = 0.5, min_models: int = 2):
        super().__init__(iou_threshold)
        self.min_models = min_models
    
    @property
    def name(self) -> str:
        return f"affirmative_nms_{self.min_models}"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        merged_detections = []
        
        # Get all unique classes
        all_classes = set()
        for model_dets in detections.values():
            for det in model_dets:
                all_classes.add(det.class_id)
        
        # Process each class
        for class_id in all_classes:
            # Collect detections for this class from each model
            class_detections_by_model = defaultdict(list)
            
            for model, model_detections in detections.items():
                for det in model_detections:
                    if det.class_id == class_id:
                        class_detections_by_model[model].append(det)
            
            # Skip if not enough models detected this class
            if len(class_detections_by_model) < self.min_models:
                continue
            
            # Collect all detections for this class
            all_class_dets = []
            for dets in class_detections_by_model.values():
                all_class_dets.extend(dets)
            
            if not all_class_dets:
                continue
            
            # Apply NMS
            boxes = torch.tensor([det.xyxy for det in all_class_dets])
            scores = torch.tensor([det.confidence for det in all_class_dets])
            
            keep_indices = ops.nms(boxes, scores, self.iou_threshold)
            
            # For each kept detection, check if enough models agree
            for idx in keep_indices:
                base_det = all_class_dets[idx]
                
                # Count how many models have a similar detection
                matching_models = set([base_det.model_source])
                matching_dets = [base_det]
                
                for det in all_class_dets:
                    if det != base_det and det.model_source not in matching_models:
                        iou = calculate_iou(base_det.bbox, det.bbox)
                        if iou >= self.iou_threshold:
                            matching_models.add(det.model_source)
                            matching_dets.append(det)
                
                # Keep if enough models agree
                if len(matching_models) >= self.min_models:
                    # Average the matching detections
                    import numpy as np
                    
                    weights = [det.confidence for det in matching_dets]
                    weights = np.array(weights) / np.sum(weights)
                    
                    avg_x = np.average([det.x for det in matching_dets], weights=weights)
                    avg_y = np.average([det.y for det in matching_dets], weights=weights)
                    avg_w = np.average([det.w for det in matching_dets], weights=weights)
                    avg_h = np.average([det.h for det in matching_dets], weights=weights)
                    avg_conf = np.average([det.confidence for det in matching_dets], weights=weights)
                    
                    merged_det = Detection(
                        class_id=class_id,
                        x=avg_x,
                        y=avg_y,
                        w=avg_w,
                        h=avg_h,
                        confidence=avg_conf,
                        model_source=f"affirmative_{len(matching_models)}"
                    )
                    merged_detections.append(merged_det)
        
        return merged_detections