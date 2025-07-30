import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict
from collections import Counter, defaultdict

from .base import BaseStrategy
from ..core.detection import Detection


class DBSCANClustering(BaseStrategy):
    """DBSCAN clustering-based ensemble strategy."""
    
    def __init__(self, eps: float = 0.1, min_samples: int = 2):
        super().__init__(iou_threshold=0.5)  # Not used in DBSCAN
        self.eps = eps
        self.min_samples = min_samples
    
    @property
    def name(self) -> str:
        return "dbscan"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Extract centers for clustering
        centers = np.array([[det.x, det.y] for det in all_detections])
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(centers)
        
        # Group detections by cluster
        clusters = defaultdict(list)
        for det, label in zip(all_detections, clustering.labels_):
            if label != -1:  # Ignore noise points
                clusters[label].append(det)
        
        # Merge each cluster
        merged_detections = []
        for cluster_detections in clusters.values():
            merged_det = self._merge_cluster(cluster_detections, detections)
            merged_detections.append(merged_det)
        
        return merged_detections
    
    def _merge_cluster(self, cluster: List[Detection], 
                      all_detections: Dict[str, List[Detection]]) -> Detection:
        """Merge detections in a cluster."""
        # Vote on class
        class_votes = Counter([det.class_id for det in cluster])
        voted_class = class_votes.most_common(1)[0][0]
        
        # Calculate model weights if available
        model_weights = self._calculate_model_weights(all_detections)
        
        # Calculate weights for averaging
        weights = []
        for det in cluster:
            model_weight = model_weights.get(det.model_source, 1.0)
            weights.append(model_weight * det.confidence)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of properties
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
            model_source=f"dbscan_{len(cluster)}"
        )
    
    def _calculate_model_weights(self, detections: Dict[str, List[Detection]]) -> Dict[str, float]:
        """Calculate weights for each model based on average confidence."""
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