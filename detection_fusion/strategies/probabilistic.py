import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict, Counter

from .base import BaseStrategy
from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class SoftVoting(BaseStrategy):
    """Soft voting with temperature scaling."""
    
    def __init__(self, iou_threshold: float = 0.5, temperature: float = 1.0):
        super().__init__(iou_threshold)
        self.temperature = temperature
    
    @property
    def name(self) -> str:
        return "soft_voting"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Get model weights
        model_weights = self._calculate_model_weights(detections)
        
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Cluster overlapping detections
        clusters = self._cluster_detections(all_detections)
        
        # Apply soft voting
        merged_detections = []
        for cluster in clusters:
            merged_det = self._soft_vote_cluster(cluster, model_weights)
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
            
            for j, det2 in enumerate(detections):
                if j <= i or j in used:
                    continue
                    
                iou = calculate_iou(det1.bbox, det2.bbox)
                if iou >= self.iou_threshold:
                    cluster.append(det2)
                    used.add(j)
        
        return clusters
    
    def _soft_vote_cluster(self, cluster: List[Detection], 
                          model_weights: Dict[str, float]) -> Detection:
        """Apply soft voting to a cluster."""
        # Calculate class probabilities
        class_probs = defaultdict(float)
        total_weight = 0
        
        for det in cluster:
            # Get model weight
            model_weight = model_weights.get(det.model_source, 1.0)
            
            # Apply temperature scaling to confidence
            prob = np.exp(det.confidence / self.temperature)
            
            # Accumulate weighted probability
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
        
        # Calculate weighted average of bounding boxes
        weights = []
        for det in cluster:
            model_weight = model_weights.get(det.model_source, 1.0)
            weights.append(model_weight * det.confidence)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        avg_x = np.average([det.x for det in cluster], weights=weights)
        avg_y = np.average([det.y for det in cluster], weights=weights)
        avg_w = np.average([det.w for det in cluster], weights=weights)
        avg_h = np.average([det.h for det in cluster], weights=weights)
        
        return Detection(
            class_id=voted_class,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=class_confidence,
            model_source=f"soft_vote_{len(cluster)}"
        )
    
    def _calculate_model_weights(self, detections: Dict[str, List[Detection]]) -> Dict[str, float]:
        """Calculate weights for each model."""
        model_weights = {}
        
        for model, model_detections in detections.items():
            if model_detections:
                avg_conf = np.mean([det.confidence for det in model_detections])
                model_weights[model] = avg_conf
            else:
                model_weights[model] = 0.0
        
        # Normalize
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            for model in model_weights:
                model_weights[model] /= total_weight
        
        return model_weights


class BayesianFusion(BaseStrategy):
    """Bayesian fusion with class priors."""
    
    def __init__(self, iou_threshold: float = 0.5, 
                 class_priors: Optional[Dict[int, float]] = None):
        super().__init__(iou_threshold)
        self.class_priors = class_priors
    
    @property
    def name(self) -> str:
        return "bayesian"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Calculate class priors if not provided
        if self.class_priors is None:
            self.class_priors = self._calculate_class_priors(all_detections)
        
        # Cluster overlapping detections
        clusters = self._cluster_detections(all_detections)
        
        # Apply Bayesian fusion
        merged_detections = []
        for cluster in clusters:
            merged_det = self._bayesian_merge_cluster(cluster)
            merged_detections.append(merged_det)
        
        return merged_detections
    
    def _calculate_class_priors(self, detections: List[Detection]) -> Dict[int, float]:
        """Calculate class priors from detections."""
        class_counts = Counter([det.class_id for det in detections])
        total = len(detections)
        return {cls: count/total for cls, count in class_counts.items()}
    
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
    
    def _bayesian_merge_cluster(self, cluster: List[Detection]) -> Detection:
        """Apply Bayesian fusion to merge a cluster."""
        # Get all possible classes in cluster
        possible_classes = set(det.class_id for det in cluster)
        
        # Calculate posterior probability for each class
        class_posteriors = {}
        
        for class_id in possible_classes:
            # Prior probability
            prior = self.class_priors.get(class_id, 0.01)
            
            # Calculate likelihood
            likelihood = 1.0
            for det in cluster:
                if det.class_id == class_id:
                    # Probability of observing this detection given the class
                    likelihood *= det.confidence
                else:
                    # Probability of not observing this class
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
        
        # Weighted average of bounding boxes
        weights = [det.confidence for det in cluster]
        weights = np.array(weights) / np.sum(weights)
        
        avg_x = np.average([det.x for det in cluster], weights=weights)
        avg_y = np.average([det.y for det in cluster], weights=weights)
        avg_w = np.average([det.w for det in cluster], weights=weights)
        avg_h = np.average([det.h for det in cluster], weights=weights)
        
        return Detection(
            class_id=voted_class,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=confidence,
            model_source=f"bayesian_{len(cluster)}"
        )