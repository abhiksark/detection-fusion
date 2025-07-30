"""
Standard Object Detection Evaluation Metrics

This module implements standard evaluation metrics for object detection including
Average Precision (AP), Mean Average Precision (mAP), precision, recall, and F1 score.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..core.detection import Detection
from ..utils.metrics import calculate_iou


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    precision: float
    recall: float
    f1_score: float
    ap: float
    tp: int
    fp: int
    fn: int
    total_gt: int
    total_pred: int


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for object detection.
    
    Supports standard metrics including AP, mAP, precision, recall, F1-score,
    and detailed analysis by confidence thresholds and IoU thresholds.
    """
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.1):
        """
        Initialize evaluation metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
    def evaluate(
        self, 
        predictions: List[Detection], 
        ground_truth: List[Detection]
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            EvaluationResult containing all metrics
        """
        # Filter predictions by confidence
        filtered_predictions = [
            pred for pred in predictions 
            if pred.confidence >= self.confidence_threshold
        ]
        
        # Calculate matches
        matches = self._match_predictions_to_gt(filtered_predictions, ground_truth)
        
        # Count true positives, false positives, false negatives
        tp = sum(1 for match in matches if match['matched'])
        fp = len(filtered_predictions) - tp
        fn = len(ground_truth) - tp
        
        # Calculate basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate Average Precision
        ap = self._calculate_ap(filtered_predictions, ground_truth)
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            ap=ap,
            tp=tp,
            fp=fp,
            fn=fn,
            total_gt=len(ground_truth),
            total_pred=len(filtered_predictions)
        )
    
    def evaluate_per_class(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ) -> Dict[int, EvaluationResult]:
        """
        Evaluate predictions per class.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            Dictionary mapping class_id to EvaluationResult
        """
        # Group by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for pred in predictions:
            pred_by_class[pred.class_id].append(pred)
            
        for gt in ground_truth:
            gt_by_class[gt.class_id].append(gt)
        
        # Get all classes
        all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())
        
        # Evaluate each class
        results = {}
        for class_id in all_classes:
            class_predictions = pred_by_class[class_id]
            class_gt = gt_by_class[class_id]
            results[class_id] = self.evaluate(class_predictions, class_gt)
            
        return results
    
    def calculate_map(
        self,
        predictions: List[Detection],  
        ground_truth: List[Detection],
        iou_thresholds: Optional[List[float]] = None
    ) -> float:
        """
        Calculate Mean Average Precision (mAP).
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            iou_thresholds: List of IoU thresholds. If None, uses single threshold.
            
        Returns:
            Mean Average Precision value
        """
        if iou_thresholds is None:
            iou_thresholds = [self.iou_threshold]
        
        # Calculate per-class results for each IoU threshold
        map_scores = []
        
        for iou_thresh in iou_thresholds:
            original_threshold = self.iou_threshold
            self.iou_threshold = iou_thresh
            
            class_results = self.evaluate_per_class(predictions, ground_truth)
            class_aps = [result.ap for result in class_results.values()]
            
            # Calculate mean AP for this IoU threshold
            mean_ap = np.mean(class_aps) if class_aps else 0.0
            map_scores.append(mean_ap)
            
            self.iou_threshold = original_threshold
        
        return np.mean(map_scores)
    
    def precision_recall_curve(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        # Sort predictions by confidence (descending)
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        # Calculate precision and recall at each threshold
        precisions = []
        recalls = []
        thresholds = []
        
        # Use confidence values as thresholds
        confidence_values = sorted(set([pred.confidence for pred in predictions]), reverse=True)
        
        for threshold in confidence_values:
            # Filter predictions by threshold
            filtered_preds = [pred for pred in sorted_predictions if pred.confidence >= threshold]
            
            # Calculate metrics at this threshold
            result = self.evaluate(filtered_preds, ground_truth)
            precisions.append(result.precision)
            recalls.append(result.recall)
            thresholds.append(threshold)
        
        return np.array(precisions), np.array(recalls), np.array(thresholds)
    
    def _match_predictions_to_gt(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ) -> List[Dict]:
        """
        Match predictions to ground truth using IoU threshold.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            List of match dictionaries with prediction info and match status
        """
        matches = []
        used_gt_indices = set()
        
        # Sort predictions by confidence (descending)
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        for pred in sorted_predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt_indices:
                    continue
                
                # Only match same class
                if pred.class_id != gt.class_id:
                    continue
                
                iou = calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                matches.append({
                    'prediction': pred,
                    'ground_truth': ground_truth[best_gt_idx],
                    'iou': best_iou,
                    'matched': True
                })
                used_gt_indices.add(best_gt_idx)
            else:
                matches.append({
                    'prediction': pred,
                    'ground_truth': None,
                    'iou': best_iou,
                    'matched': False
                })
        
        return matches
    
    def _calculate_ap(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ) -> float:
        """
        Calculate Average Precision using the 11-point interpolation method.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            
        Returns:
            Average Precision value
        """
        if not predictions or not ground_truth:
            return 0.0
        
        # Get precision-recall curve
        precisions, recalls, _ = self.precision_recall_curve(predictions, ground_truth)
        
        # 11-point interpolation
        ap = 0.0
        for recall_level in np.arange(0, 1.1, 0.1):
            # Find precisions at recall >= recall_level
            valid_precisions = precisions[recalls >= recall_level]
            if len(valid_precisions) > 0:
                ap += np.max(valid_precisions)
        
        return ap / 11.0


class APCalculator:
    """
    Specialized class for Average Precision calculations with different interpolation methods.
    """
    
    @staticmethod
    def calculate_ap_11point(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate AP using 11-point interpolation (PASCAL VOC 2007 style)."""
        ap = 0.0
        for recall_level in np.arange(0, 1.1, 0.1):
            valid_precisions = precisions[recalls >= recall_level]
            if len(valid_precisions) > 0:
                ap += np.max(valid_precisions)
        return ap / 11.0
    
    @staticmethod 
    def calculate_ap_all_points(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate AP using all points (PASCAL VOC 2010+ style)."""
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        sorted_recalls = recalls[sorted_indices]
        sorted_precisions = precisions[sorted_indices]
        
        # Compute the precision envelope
        for i in range(len(sorted_precisions) - 2, -1, -1):
            sorted_precisions[i] = max(sorted_precisions[i], sorted_precisions[i + 1])
        
        # Find points where recall changes
        unique_recalls, unique_indices = np.unique(sorted_recalls, return_index=True)
        
        # Calculate area under curve
        ap = 0.0
        for i in range(1, len(unique_recalls)):
            recall_diff = unique_recalls[i] - unique_recalls[i-1]
            precision = sorted_precisions[unique_indices[i]]
            ap += recall_diff * precision
        
        return ap
    
    @staticmethod
    def calculate_ap_coco_style(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate AP using COCO evaluation style (101-point interpolation)."""
        # Create 101 equally spaced recall levels
        recall_levels = np.linspace(0, 1, 101)
        
        # Interpolate precision at each recall level
        interpolated_precisions = []
        for recall_level in recall_levels:
            valid_precisions = precisions[recalls >= recall_level]
            if len(valid_precisions) > 0:
                interpolated_precisions.append(np.max(valid_precisions))
            else:
                interpolated_precisions.append(0.0)
        
        return np.mean(interpolated_precisions)