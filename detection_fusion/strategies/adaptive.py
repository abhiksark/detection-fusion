import numpy as np
from typing import List, Dict
from collections import defaultdict

from .base import BaseStrategy
from ..core.detection import Detection
from ..utils.metrics import calculate_iou


class AdaptiveThresholdStrategy(BaseStrategy):
    """Strategy that adapts IoU threshold based on object size."""
    
    def __init__(self, small_threshold: float = 0.3, large_threshold: float = 0.7, 
                 size_cutoff: float = 0.05):
        super().__init__(iou_threshold=0.5)  # Default, will be overridden
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        self.size_cutoff = size_cutoff
    
    @property
    def name(self) -> str:
        return "adaptive_threshold"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Separate detections by size
        small_objects = {}
        large_objects = {}
        
        for model, dets in detections.items():
            small_objects[model] = []
            large_objects[model] = []
            
            for det in dets:
                object_size = det.w * det.h
                if object_size < self.size_cutoff:
                    small_objects[model].append(det)
                else:
                    large_objects[model].append(det)
        
        # Apply different strategies based on size
        from .voting import MajorityVoting
        
        small_voter = MajorityVoting(self.small_threshold, min_votes=2)
        large_voter = MajorityVoting(self.large_threshold, min_votes=2)
        
        small_results = small_voter.merge(small_objects)
        large_results = large_voter.merge(large_objects)
        
        return small_results + large_results


class DensityAdaptiveStrategy(BaseStrategy):
    """Strategy that adapts based on detection density in spatial regions."""
    
    def __init__(self, iou_threshold: float = 0.5, grid_size: int = 5,
                 high_density_threshold: int = 10):
        super().__init__(iou_threshold)
        self.grid_size = grid_size
        self.high_density_threshold = high_density_threshold
    
    @property
    def name(self) -> str:
        return "density_adaptive"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten all detections
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        if not all_detections:
            return []
        
        # Calculate density map
        density_map = self._calculate_density_map(all_detections)
        
        # Separate detections by density regions
        high_density_dets = {}
        low_density_dets = {}
        
        for model, dets in detections.items():
            high_density_dets[model] = []
            low_density_dets[model] = []
            
            for det in dets:
                grid_x = min(int(det.x * self.grid_size), self.grid_size - 1)
                grid_y = min(int(det.y * self.grid_size), self.grid_size - 1)
                
                if density_map[grid_y, grid_x] > self.high_density_threshold:
                    high_density_dets[model].append(det)
                else:
                    low_density_dets[model].append(det)
        
        # Apply different strategies
        from .voting import MajorityVoting
        from .nms import NMSStrategy
        
        # High density regions: more aggressive NMS
        nms_strategy = NMSStrategy(iou_threshold=0.3, score_threshold=0.2)
        high_density_results = nms_strategy.merge(high_density_dets)
        
        # Low density regions: conservative majority voting
        majority_strategy = MajorityVoting(iou_threshold=0.7, min_votes=2)
        low_density_results = majority_strategy.merge(low_density_dets)
        
        return high_density_results + low_density_results
    
    def _calculate_density_map(self, detections: List[Detection]) -> np.ndarray:
        """Calculate detection density in a spatial grid."""
        density_map = np.zeros((self.grid_size, self.grid_size))
        
        for det in detections:
            grid_x = min(int(det.x * self.grid_size), self.grid_size - 1)
            grid_y = min(int(det.y * self.grid_size), self.grid_size - 1)
            density_map[grid_y, grid_x] += 1
        
        return density_map


class MultiScaleStrategy(BaseStrategy):
    """Strategy that handles multiple scales of objects differently."""
    
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(iou_threshold)
        self.scale_thresholds = [0.01, 0.05, 0.15]  # Small, medium, large
    
    @property
    def name(self) -> str:
        return "multi_scale"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Separate detections by scale
        scale_groups = self._group_by_scale(detections)
        
        merged_results = []
        
        # Apply scale-specific strategies
        for scale, scale_detections in scale_groups.items():
            if not any(scale_detections.values()):
                continue
            
            if scale == "tiny":
                # Tiny objects: very permissive
                from .voting import MajorityVoting
                strategy = MajorityVoting(iou_threshold=0.2, min_votes=1)
            elif scale == "small":
                # Small objects: moderate
                from .voting import MajorityVoting
                strategy = MajorityVoting(iou_threshold=0.3, min_votes=2)
            elif scale == "medium":
                # Medium objects: standard
                from .voting import MajorityVoting
                strategy = MajorityVoting(iou_threshold=0.5, min_votes=2)
            else:  # large
                # Large objects: strict
                from .voting import MajorityVoting
                strategy = MajorityVoting(iou_threshold=0.7, min_votes=3)
            
            scale_results = strategy.merge(scale_detections)
            merged_results.extend(scale_results)
        
        return merged_results
    
    def _group_by_scale(self, detections: Dict[str, List[Detection]]) -> Dict[str, Dict[str, List[Detection]]]:
        """Group detections by object scale."""
        scale_groups = {
            "tiny": {model: [] for model in detections.keys()},
            "small": {model: [] for model in detections.keys()},
            "medium": {model: [] for model in detections.keys()},
            "large": {model: [] for model in detections.keys()}
        }
        
        for model, dets in detections.items():
            for det in dets:
                object_area = det.w * det.h
                
                if object_area < self.scale_thresholds[0]:
                    scale_groups["tiny"][model].append(det)
                elif object_area < self.scale_thresholds[1]:
                    scale_groups["small"][model].append(det)
                elif object_area < self.scale_thresholds[2]:
                    scale_groups["medium"][model].append(det)
                else:
                    scale_groups["large"][model].append(det)
        
        return scale_groups


class ConsensusRankingStrategy(BaseStrategy):
    """Strategy based on ranking consensus across models."""
    
    def __init__(self, iou_threshold: float = 0.5, rank_weight: float = 0.5):
        super().__init__(iou_threshold)
        self.rank_weight = rank_weight
    
    @property
    def name(self) -> str:
        return "consensus_ranking"
    
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        # Flatten and rank detections per model
        model_rankings = {}
        
        for model, dets in detections.items():
            # Sort by confidence (rank 1 = highest confidence)
            sorted_dets = sorted(dets, key=lambda x: x.confidence, reverse=True)
            model_rankings[model] = {
                det: rank + 1 for rank, det in enumerate(sorted_dets)
            }
        
        # Find overlapping detections across models
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)
        
        clusters = self._cluster_detections(all_detections)
        
        # Rank-based consensus for each cluster
        merged_detections = []
        for cluster in clusters:
            merged_det = self._rank_based_consensus(cluster, model_rankings)
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
            
            clusters.append(cluster)
        
        return clusters
    
    def _rank_based_consensus(self, cluster: List[Detection], 
                            model_rankings: Dict[str, Dict[Detection, int]]) -> Detection:
        """Apply rank-based consensus to a cluster."""
        if len(cluster) == 1:
            return cluster[0]
        
        # Calculate consensus score for each detection
        consensus_scores = []
        
        for det in cluster:
            rank = model_rankings[det.model_source].get(det, float('inf'))
            # Lower rank (higher confidence) = higher score
            rank_score = 1.0 / rank if rank != float('inf') else 0.0
            
            # Combine rank score with confidence
            consensus_score = (self.rank_weight * rank_score + 
                             (1 - self.rank_weight) * det.confidence)
            consensus_scores.append(consensus_score)
        
        # Use scores as weights for averaging
        weights = np.array(consensus_scores)
        weights = weights / weights.sum()
        
        # Vote on class
        class_scores = defaultdict(float)
        for det, weight in zip(cluster, weights):
            class_scores[det.class_id] += weight
        
        voted_class = max(class_scores, key=class_scores.get)
        
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
            model_source=f"consensus_rank_{len(cluster)}"
        )