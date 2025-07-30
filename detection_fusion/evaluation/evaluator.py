"""
Main Evaluation Orchestrator

This module provides the main Evaluator class that orchestrates comprehensive
evaluation workflows including metrics calculation, error analysis, and reporting.
"""

import os
import json
from typing import List, Dict
import numpy as np

from ..core.detection import Detection
from ..utils.io import read_detections
from .metrics import EvaluationMetrics
from .error_analysis import ErrorAnalyzer


class Evaluator:
    """
    Main evaluation orchestrator for comprehensive object detection evaluation.
    
    Provides high-level interface for evaluating ensemble results against ground truth,
    including metrics calculation, error analysis, and detailed reporting.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.1,
        gt_dir: str = "labels/GT"
    ):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
            confidence_threshold: Minimum confidence threshold for predictions
            gt_dir: Directory containing ground truth annotations
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.gt_dir = gt_dir
        
        # Initialize evaluation components
        self.metrics_calculator = EvaluationMetrics(iou_threshold, confidence_threshold)
        self.error_analyzer = ErrorAnalyzer(iou_threshold)
        
        # Cache for ground truth data
        self._gt_cache = {}
        
    def load_ground_truth(self, gt_file: str = "detections.txt") -> List[Detection]:
        """
        Load ground truth detections.
        
        Args:
            gt_file: Name of ground truth file
            
        Returns:
            List of ground truth detections
        """
        if gt_file not in self._gt_cache:
            gt_path = os.path.join(self.gt_dir, gt_file)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
            
            self._gt_cache[gt_file] = read_detections(gt_path, model_source="GT")
        
        return self._gt_cache[gt_file]
    
    def evaluate_predictions(
        self,
        predictions: List[Detection],
        gt_file: str = "detections.txt",
        include_error_analysis: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation of predictions against ground truth.
        
        Args:
            predictions: List of predicted detections
            gt_file: Ground truth file name
            include_error_analysis: Whether to include detailed error analysis
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load ground truth
        ground_truth = self.load_ground_truth(gt_file)
        
        # Calculate metrics
        overall_result = self.metrics_calculator.evaluate(predictions, ground_truth)
        per_class_results = self.metrics_calculator.evaluate_per_class(predictions, ground_truth)
        
        # Calculate mAP with different IoU thresholds
        map_50 = self.metrics_calculator.calculate_map(predictions, ground_truth, [0.5])
        map_50_95 = self.metrics_calculator.calculate_map(
            predictions, ground_truth, 
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        
        # Prepare results
        results = {
            'overall_metrics': {
                'precision': overall_result.precision,
                'recall': overall_result.recall,
                'f1_score': overall_result.f1_score,
                'ap': overall_result.ap,
                'map_50': map_50,
                'map_50_95': map_50_95,
                'tp': overall_result.tp,
                'fp': overall_result.fp,
                'fn': overall_result.fn,
                'total_gt': overall_result.total_gt,
                'total_pred': overall_result.total_pred
            },
            'per_class_metrics': {
                class_id: {
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'ap': result.ap,
                    'tp': result.tp,
                    'fp': result.fp,
                    'fn': result.fn,
                    'total_gt': result.total_gt,
                    'total_pred': result.total_pred
                }
                for class_id, result in per_class_results.items()
            },
            'evaluation_config': {
                'iou_threshold': self.iou_threshold,
                'confidence_threshold': self.confidence_threshold,
                'gt_file': gt_file
            }
        }
        
        # Add error analysis if requested
        if include_error_analysis:
            error_instances, error_summary = self.error_analyzer.analyze_errors(predictions, ground_truth)
            
            results['error_analysis'] = {
                'summary': {
                    'false_positives': error_summary.false_positives,
                    'false_negatives': error_summary.false_negatives,
                    'localization_errors': error_summary.localization_errors,
                    'classification_errors': error_summary.classification_errors,
                    'duplicate_detections': error_summary.duplicate_detections,
                    'true_positives': error_summary.true_positives,
                    'total_errors': error_summary.total_errors,
                    'total_detections': error_summary.total_detections
                },
                'error_rate': error_summary.total_errors / error_summary.total_detections if error_summary.total_detections > 0 else 0,
                'confidence_analysis': self.error_analyzer.analyze_by_confidence(error_instances),
                'spatial_analysis': self.error_analyzer.analyze_spatial_distribution(error_instances),
                'size_analysis': self.error_analyzer.analyze_by_object_size(error_instances)
            }
        
        return results
    
    def evaluate_ensemble_strategies(
        self,
        strategy_results: Dict[str, List[Detection]],
        gt_file: str = "detections.txt"
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple ensemble strategies against ground truth.
        
        Args:
            strategy_results: Dictionary mapping strategy names to detection results
            gt_file: Ground truth file name
            
        Returns:
            Dictionary mapping strategy names to evaluation results
        """
        evaluation_results = {}
        
        for strategy_name, predictions in strategy_results.items():
            try:
                evaluation_results[strategy_name] = self.evaluate_predictions(
                    predictions, gt_file, include_error_analysis=True
                )
            except Exception as e:
                evaluation_results[strategy_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return evaluation_results
    
    def compare_with_individual_models(
        self,
        ensemble_predictions: List[Detection],
        model_predictions: Dict[str, List[Detection]],
        gt_file: str = "detections.txt"
    ) -> Dict:
        """
        Compare ensemble results with individual model performance.
        
        Args:
            ensemble_predictions: Ensemble prediction results
            model_predictions: Dictionary mapping model names to predictions
            gt_file: Ground truth file name
            
        Returns:
            Comparison results
        """
        # Evaluate ensemble
        ensemble_results = self.evaluate_predictions(ensemble_predictions, gt_file)
        
        # Evaluate individual models
        model_results = {}
        for model_name, predictions in model_predictions.items():
            model_results[model_name] = self.evaluate_predictions(predictions, gt_file)
        
        # Calculate improvement metrics
        ensemble_map = ensemble_results['overall_metrics']['map_50']
        model_maps = {name: results['overall_metrics']['map_50'] for name, results in model_results.items()}
        
        best_individual_map = max(model_maps.values()) if model_maps else 0
        average_individual_map = np.mean(list(model_maps.values())) if model_maps else 0
        
        comparison = {
            'ensemble_results': ensemble_results,
            'individual_model_results': model_results,
            'comparison_metrics': {
                'ensemble_map_50': ensemble_map,
                'best_individual_map_50': best_individual_map,
                'average_individual_map_50': average_individual_map,
                'improvement_over_best': ensemble_map - best_individual_map,
                'improvement_over_average': ensemble_map - average_individual_map,
                'relative_improvement_over_best': (ensemble_map - best_individual_map) / best_individual_map * 100 if best_individual_map > 0 else 0,
                'relative_improvement_over_average': (ensemble_map - average_individual_map) / average_individual_map * 100 if average_individual_map > 0 else 0
            }
        }
        
        return comparison
    
    def generate_precision_recall_data(
        self,
        predictions: List[Detection],
        gt_file: str = "detections.txt"
    ) -> Dict:
        """
        Generate precision-recall curve data for visualization.
        
        Args:
            predictions: List of predicted detections
            gt_file: Ground truth file name
            
        Returns:
            Dictionary containing PR curve data
        """
        ground_truth = self.load_ground_truth(gt_file)
        
        # Overall PR curve
        precisions, recalls, thresholds = self.metrics_calculator.precision_recall_curve(predictions, ground_truth)
        
        # Per-class PR curves
        per_class_pr = {}
        per_class_results = self.metrics_calculator.evaluate_per_class(predictions, ground_truth)
        
        for class_id in per_class_results.keys():
            class_predictions = [pred for pred in predictions if pred.class_id == class_id]
            class_gt = [gt for gt in ground_truth if gt.class_id == class_id]
            
            if class_predictions and class_gt:
                class_precisions, class_recalls, class_thresholds = self.metrics_calculator.precision_recall_curve(
                    class_predictions, class_gt
                )
                per_class_pr[class_id] = {
                    'precision': class_precisions.tolist(),
                    'recall': class_recalls.tolist(),
                    'thresholds': class_thresholds.tolist()
                }
        
        return {
            'overall': {
                'precision': precisions.tolist(),
                'recall': recalls.tolist(),
                'thresholds': thresholds.tolist()
            },
            'per_class': per_class_pr
        }
    
    def save_evaluation_report(
        self,
        results: Dict,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Output file path
            format: Output format ('json' or 'txt')
        """
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format.lower() == 'txt':
            with open(output_path, 'w') as f:
                self._write_text_report(results, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _write_text_report(self, results: Dict, file_handle) -> None:
        """Write evaluation results as formatted text report."""
        file_handle.write("OBJECT DETECTION EVALUATION REPORT\n")
        file_handle.write("=" * 50 + "\n\n")
        
        # Overall metrics
        overall = results.get('overall_metrics', {})
        file_handle.write("OVERALL PERFORMANCE:\n")
        file_handle.write("-" * 20 + "\n")
        file_handle.write(f"Precision: {overall.get('precision', 0):.4f}\n")
        file_handle.write(f"Recall: {overall.get('recall', 0):.4f}\n")
        file_handle.write(f"F1-Score: {overall.get('f1_score', 0):.4f}\n")
        file_handle.write(f"Average Precision: {overall.get('ap', 0):.4f}\n")
        file_handle.write(f"mAP@0.5: {overall.get('map_50', 0):.4f}\n")
        file_handle.write(f"mAP@0.5:0.95: {overall.get('map_50_95', 0):.4f}\n")
        file_handle.write(f"True Positives: {overall.get('tp', 0)}\n")
        file_handle.write(f"False Positives: {overall.get('fp', 0)}\n")
        file_handle.write(f"False Negatives: {overall.get('fn', 0)}\n\n")
        
        # Per-class metrics
        per_class = results.get('per_class_metrics', {})
        if per_class:
            file_handle.write("PER-CLASS PERFORMANCE:\n")
            file_handle.write("-" * 20 + "\n")
            file_handle.write(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AP':<10}\n")
            file_handle.write("-" * 50 + "\n")
            
            for class_id, metrics in per_class.items():
                file_handle.write(f"{class_id:<8} {metrics.get('precision', 0):<10.4f} "
                                f"{metrics.get('recall', 0):<10.4f} {metrics.get('f1_score', 0):<10.4f} "
                                f"{metrics.get('ap', 0):<10.4f}\n")
            file_handle.write("\n")
        
        # Error analysis
        error_analysis = results.get('error_analysis')
        if error_analysis:
            summary = error_analysis.get('summary', {})
            file_handle.write("ERROR ANALYSIS:\n")
            file_handle.write("-" * 20 + "\n")
            file_handle.write(f"True Positives: {summary.get('true_positives', 0)}\n")
            file_handle.write(f"False Positives: {summary.get('false_positives', 0)}\n")
            file_handle.write(f"False Negatives: {summary.get('false_negatives', 0)}\n")
            file_handle.write(f"Localization Errors: {summary.get('localization_errors', 0)}\n")
            file_handle.write(f"Classification Errors: {summary.get('classification_errors', 0)}\n")
            file_handle.write(f"Duplicate Detections: {summary.get('duplicate_detections', 0)}\n")
            file_handle.write(f"Total Errors: {summary.get('total_errors', 0)}\n")
            file_handle.write(f"Error Rate: {error_analysis.get('error_rate', 0):.4f}\n\n")
        
        # Configuration
        config = results.get('evaluation_config', {})
        file_handle.write("EVALUATION CONFIGURATION:\n")
        file_handle.write("-" * 20 + "\n")
        file_handle.write(f"IoU Threshold: {config.get('iou_threshold', 0.5)}\n")
        file_handle.write(f"Confidence Threshold: {config.get('confidence_threshold', 0.1)}\n")
        file_handle.write(f"Ground Truth File: {config.get('gt_file', 'detections.txt')}\n")