import numpy as np
from typing import List, Dict
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import euclidean_distances

from .base import BaseStrategy
from ..core.detection import Detection


class DistanceWeightedVoting(BaseStrategy):
    """Voting strategy that weights by spatial distance between detections."""
    
    def __init__(self, iou_threshold: float = 0.5, distance_weight: float = 1.0):
        super().__init__(iou_threshold)
        self.distance_weight = distance_weight
    
    @property
    def name(self) -> str:
        return "distance_weighted"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Cluster overlapping detections
        clusters = self._cluster_detections(all_detections)
        
        # Apply distance-weighted voting
        merged_detections = []
        for cluster in clusters:
            merged_det = self._distance_weighted_vote(cluster)
            merged_detections.append(merged_det)
        
        return merged_detections
    
    def _cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """Group overlapping detections into clusters."""
        from ..utils.metrics import calculate_iou
        
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
            
            clusters.append(cluster)
        
        return clusters
    
    def _distance_weighted_vote(self, cluster: List[Detection]) -> Detection:
        """Apply distance-weighted voting to a cluster."""
        if len(cluster) == 1:
            return cluster[0]
        
        # Calculate centroid
        centers = np.array([[det.x, det.y] for det in cluster])
        centroid = np.mean(centers, axis=0)
        
        # Calculate distance weights (closer to centroid = higher weight)
        distances = euclidean_distances(centers, centroid.reshape(1, -1)).flatten()
        max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
        distance_weights = 1.0 - (distances / max_distance)
        
        # Combine with confidence weights
        confidence_weights = np.array([det.confidence for det in cluster])
        final_weights = (distance_weights * self.distance_weight + confidence_weights) / 2
        final_weights = final_weights / final_weights.sum()
        
        # Vote on class
        class_scores = defaultdict(float)
        for det, weight in zip(cluster, final_weights):
            class_scores[det.class_id] += weight
        
        voted_class = max(class_scores, key=class_scores.get)
        
        # Weighted average of properties
        avg_x = np.average([det.x for det in cluster], weights=final_weights)
        avg_y = np.average([det.y for det in cluster], weights=final_weights)
        avg_w = np.average([det.w for det in cluster], weights=final_weights)
        avg_h = np.average([det.h for det in cluster], weights=final_weights)
        avg_conf = np.average([det.confidence for det in cluster], weights=final_weights)
        
        return Detection(
            class_id=voted_class,
            x=avg_x,
            y=avg_y,
            w=avg_w,
            h=avg_h,
            confidence=avg_conf,
            model_source=f"distance_weighted_{len(cluster)}"
        )


class CentroidClustering(BaseStrategy):
    """Clustering strategy based on detection centroids."""
    
    def __init__(self, distance_threshold: float = 0.1, min_cluster_size: int = 2):
        super().__init__(iou_threshold=0.5)  # Not used in this strategy
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
    
    @property
    def name(self) -> str:
        return "centroid_clustering"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Extract centers for clustering
        centers = np.array([[det.x, det.y] for det in all_detections])
        
        # Simple agglomerative clustering based on distance
        clusters = self._agglomerative_clustering(centers, all_detections)
        
        # Merge each cluster
        merged_detections = []
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                merged_det = self._merge_cluster(cluster)
                merged_detections.append(merged_det)
        
        return merged_detections
    
    def _agglomerative_clustering(self, centers: np.ndarray, 
                                detections: List[Detection]) -> List[List[Detection]]:
        """Simple agglomerative clustering based on centroid distance."""
        clusters = [[det] for det in detections]
        
        while True:
            # Find closest pair of clusters
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate centroid distance between clusters
                    cluster_i_centers = np.array([[det.x, det.y] for det in clusters[i]])
                    cluster_j_centers = np.array([[det.x, det.y] for det in clusters[j]])
                    
                    centroid_i = np.mean(cluster_i_centers, axis=0)
                    centroid_j = np.mean(cluster_j_centers, axis=0)
                    
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Stop if minimum distance exceeds threshold
            if min_distance > self.distance_threshold:
                break
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        return clusters
    
    def _merge_cluster(self, cluster: List[Detection]) -> Detection:
        """Merge detections in a cluster."""
        # Vote on class
        class_votes = Counter([det.class_id for det in cluster])
        voted_class = class_votes.most_common(1)[0][0]
        
        # Average properties weighted by confidence
        weights = np.array([det.confidence for det in cluster])
        weights = weights / weights.sum()
        
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
            model_source=f"centroid_cluster_{len(cluster)}"
        )