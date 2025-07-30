import numpy as np
from typing import List, Dict
from collections import Counter, defaultdict

from .base import BaseStrategy
from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class MajorityVoting(BaseStrategy):
    """Majority voting strategy for ensemble."""
    
    def __init__(self, iou_threshold: float = 0.5, min_votes: int = 2):
        super().__init__(iou_threshold)
        self.min_votes = min_votes
    
    @property
    def name(self) -> str:
        return f"majority_vote_{self.min_votes}"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Cluster overlapping detections
        clusters = self._cluster_detections(all_detections)
        
        # Vote on each cluster
        merged_detections = []
        for cluster in clusters:
            if len(cluster) >= self.min_votes:
                merged_det = self._vote_on_cluster(cluster)
                merged_detections.append(merged_det)
        
        return merged_detections
    
    def _cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """Group overlapping detections into clusters."""
        clusters = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det1]
            used.add(i)
            
            # Find all detections that overlap with current cluster
            for j, det2 in enumerate(detections):
                if j <= i or j in used:
                    continue
                
                # Check if det2 overlaps with any detection in cluster
                for cluster_det in cluster:
                    iou = calculate_iou(cluster_det.bbox, det2.bbox)
                    if iou >= self.iou_threshold:
                        cluster.append(det2)
                        used.add(j)
                        break
            
            clusters.append(cluster)
        
        return clusters
    
    def _vote_on_cluster(self, cluster: List[Detection]) -> Detection:
        """Apply majority voting to a cluster of detections."""
        # Vote on class
        class_votes = Counter([det.class_id for det in cluster])
        voted_class = class_votes.most_common(1)[0][0]
        
        # Average the bounding boxes
        avg_x = np.mean([det.x for det in cluster])
        avg_y = np.mean([det.y for det in cluster])
        avg_w = np.mean([det.w for det in cluster])
        avg_h = np.mean([det.h for det in cluster])
        avg_conf = np.mean([det.confidence for det in cluster])
        
        return Detection(
            class_id=voted_class,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=avg_conf,
            model_source=f"ensemble_vote_{len(cluster)}"
        )


class WeightedVoting(BaseStrategy):
    """Weighted voting strategy using confidence scores."""
    
    def __init__(self, iou_threshold: float = 0.5, use_model_weights: bool = True):
        super().__init__(iou_threshold)
        self.use_model_weights = use_model_weights
        self.model_weights = {}
    
    @property
    def name(self) -> str:
        return "weighted_vote"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Calculate model weights if needed
        if self.use_model_weights:
            self._calculate_model_weights(detections)
        
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Cluster overlapping detections
        clusters = self._cluster_detections(all_detections)
        
        # Apply weighted voting
        merged_detections = []
        for cluster in clusters:
            merged_det = self._weighted_vote_on_cluster(cluster)
            merged_detections.append(merged_det)
        
        return merged_detections
    
    def _calculate_model_weights(self, detections: Dict[str, List[Detection]]):
        """Calculate weights for each model based on average confidence."""
        for model, model_detections in detections.items():
            if model_detections:
                avg_conf = np.mean([det.confidence for det in model_detections])
                self.model_weights[model] = avg_conf
            else:
                self.model_weights[model] = 0.0
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model in self.model_weights:
                self.model_weights[model] /= total_weight
    
    def _cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """Group overlapping detections into clusters."""
        clusters = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j <= i or j in used:
                    continue
                    
                iou = calculate_iou(det1.bbox, det2.bbox)
                if iou >= self.iou_threshold:
                    cluster.append(det2)
                    used.add(j)
        
        return clusters
    
    def _weighted_vote_on_cluster(self, cluster: List[Detection]) -> Detection:
        """Apply weighted voting to a cluster."""
        # Calculate weights
        if self.use_model_weights:
            weights = np.array([
                self.model_weights.get(det.model_source, 1.0) * det.confidence 
                for det in cluster
            ])
        else:
            weights = np.array([det.confidence for det in cluster])
        
        weights = weights / weights.sum()
        
        # Weighted vote on class
        class_scores = defaultdict(float)
        for det, weight in zip(cluster, weights):
            class_scores[det.class_id] += weight
        
        voted_class = max(class_scores, key=class_scores.get)
        
        # Weighted average of bounding boxes
        avg_x = np.average([det.x for det in cluster], weights=weights)
        avg_y = np.average([det.y for det in cluster], weights=weights)
        avg_w = np.average([det.w for det in cluster], weights=weights)
        avg_h = np.average([det.h for det in cluster], weights=weights)
        avg_conf = np.average([det.confidence for det in cluster], weights=weights)
        
        return Detection(
            class_id=voted_class,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=avg_conf,
            model_source=f"ensemble_weighted_{len(cluster)}"
        )