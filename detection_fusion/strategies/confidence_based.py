import numpy as np
from typing import List, Dict
from collections import defaultdict

from .base import BaseStrategy
from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class ConfidenceThresholdVoting(BaseStrategy):
    """Voting strategy with dynamic confidence thresholds."""
    
    def __init__(self, iou_threshold: float = 0.5, base_confidence: float = 0.5,
                 adaptive_threshold: bool = True):
        super().__init__(iou_threshold)
        self.base_confidence = base_confidence
        self.adaptive_threshold = adaptive_threshold
    
    @property
    def name(self) -> str:
        return "confidence_threshold"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Calculate adaptive thresholds if enabled
        if self.adaptive_threshold:
            confidence_thresholds = self._calculate_adaptive_thresholds(detections)
        else:
            confidence_thresholds = {model: self.base_confidence for model in detections.keys()}
        
        # Filter detections by confidence
        filtered_detections = {}
        for model, dets in detections.items():
            threshold = confidence_thresholds[model]
            filtered_detections[model] = [
                det for det in dets if det.confidence >= threshold
            ]
        
        # Apply majority voting to filtered detections
        from .voting import MajorityVoting
        voter = MajorityVoting(self.iou_threshold, min_votes=2)
        return voter.merge(filtered_detections)
    
    def _calculate_adaptive_thresholds(self, detections: Dict[str, List[Detection]]) -> Dict[str, float]:
        """Calculate adaptive confidence thresholds per model."""
        thresholds = {}
        
        for model, dets in detections.items():
            if not dets:
                thresholds[model] = self.base_confidence
                continue
            
            confidences = [det.confidence for det in dets]
            
            # Use median as adaptive threshold
            median_conf = np.median(confidences)
            # Ensure threshold is not too low or too high
            adaptive_threshold = np.clip(median_conf, 0.2, 0.8)
            thresholds[model] = adaptive_threshold
        
        return thresholds


class ConfidenceWeightedNMS(BaseStrategy):
    """NMS with confidence-based box regression."""
    
    def __init__(self, iou_threshold: float = 0.5, confidence_power: float = 2.0):
        super().__init__(iou_threshold)
        self.confidence_power = confidence_power
    
    @property
    def name(self) -> str:
        return "confidence_weighted_nms"
    
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
            class_groups[det.class_id].append(det)
        
        # Apply confidence-weighted NMS per class
        merged_detections = []
        for class_id, class_dets in class_groups.items():
            if not class_dets:
                continue
            
            nms_results = self._confidence_weighted_nms(class_dets)
            merged_detections.extend(nms_results)
        
        return merged_detections
    
    def _confidence_weighted_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply confidence-weighted NMS to a list of detections."""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        kept_detections = []
        
        while sorted_dets:
            # Take highest confidence detection
            current = sorted_dets.pop(0)
            kept_detections.append(current)
            
            # Find overlapping detections
            overlapping = []
            remaining = []
            
            for det in sorted_dets:
                iou = calculate_iou(current.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)
            
            # If there are overlapping detections, merge them
            if overlapping:
                all_overlapping = [current] + overlapping
                merged = self._merge_overlapping_detections(all_overlapping)
                
                # Replace current detection with merged one
                kept_detections[-1] = merged
            
            sorted_dets = remaining
        
        return kept_detections
    
    def _merge_overlapping_detections(self, detections: List[Detection]) -> Detection:
        """Merge overlapping detections using confidence weighting."""
        # Weight by confidence raised to a power
        weights = np.array([det.confidence ** self.confidence_power for det in detections])
        weights = weights / weights.sum()
        
        # Weighted average of all properties
        avg_x = np.average([det.x for det in detections], weights=weights)
        avg_y = np.average([det.y for det in detections], weights=weights)
        avg_w = np.average([det.w for det in detections], weights=weights)
        avg_h = np.average([det.h for det in detections], weights=weights)
        
        # Take highest confidence
        max_conf = max(det.confidence for det in detections)
        
        # Keep the class of the highest confidence detection
        best_det = max(detections, key=lambda x: x.confidence)
        
        return Detection(
            class_id=best_det.class_id,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=max_conf,
            model_source=f"conf_weighted_nms_{len(detections)}"
        )


class HighConfidenceFirst(BaseStrategy):
    """Strategy that prioritizes high-confidence detections."""
    
    def __init__(self, iou_threshold: float = 0.5, high_conf_threshold: float = 0.8,
                 low_conf_threshold: float = 0.3):
        super().__init__(iou_threshold)
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
    
    @property
    def name(self) -> str:
        return "high_confidence_first"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Separate by confidence levels
        high_conf = [d for d in all_detections if d.confidence >= self.high_conf_threshold]
        medium_conf = [d for d in all_detections 
                      if self.low_conf_threshold <= d.confidence < self.high_conf_threshold]
        
        merged_detections = []
        used_positions = []
        
        # First, add all high-confidence detections
        for det in high_conf:
            merged_detections.append(det)
            used_positions.append((det.x, det.y))
        
        # Then, add medium-confidence detections that don't overlap with high-confidence ones
        for det in medium_conf:
            overlaps_with_high_conf = False
            
            for used_pos in used_positions:
                # Simple distance check (could use IoU for more accuracy)
                distance = np.sqrt((det.x - used_pos[0])**2 + (det.y - used_pos[1])**2)
                if distance < 0.1:  # Threshold for spatial overlap
                    overlaps_with_high_conf = True
                    break
            
            if not overlaps_with_high_conf:
                merged_detections.append(det)
                used_positions.append((det.x, det.y))
        
        return merged_detections