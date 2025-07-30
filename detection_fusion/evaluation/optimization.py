"""
Strategy Optimization using Ground Truth Feedback

This module provides functionality to optimize ensemble strategies based on
ground truth evaluation, including automatic strategy selection and parameter tuning.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.ensemble import AdvancedEnsemble
from .evaluator import Evaluator


@dataclass
class OptimizationResult:
    """Results from strategy optimization."""
    best_strategy: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: Dict[str, Dict]
    optimization_metric: str


class StrategyOptimizer:
    """
    Optimize ensemble strategies using ground truth feedback.
    
    Provides automatic strategy selection and parameter tuning to maximize
    specified evaluation metrics like mAP, precision, recall, or F1-score.
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        optimization_metric: str = 'map_50',
        maximize: bool = True
    ):
        """
        Initialize strategy optimizer.
        
        Args:
            evaluator: Evaluator instance for calculating metrics
            optimization_metric: Metric to optimize ('map_50', 'precision', 'recall', 'f1_score')
            maximize: Whether to maximize (True) or minimize (False) the metric
        """
        self.evaluator = evaluator
        self.optimization_metric = optimization_metric
        self.maximize = maximize
        
        # Define default parameter grids for different strategies
        self.default_param_grids = {
            'majority_vote_2': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'min_votes': [2]
            },
            'majority_vote_3': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'min_votes': [3]
            },
            'weighted_vote': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'use_model_weights': [True, False]
            },
            'affirmative_nms': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'min_models': [2, 3]
            },
            'dbscan': {
                'eps': [0.05, 0.1, 0.15],
                'min_samples': [2, 3]
            },
            'soft_voting': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'temperature': [0.5, 1.0, 2.0]
            },
            'bayesian': {
                'iou_threshold': [0.3, 0.5, 0.7]
            },
            'adaptive_threshold': {
                'small_threshold': [0.2, 0.3],
                'large_threshold': [0.6, 0.7],
                'size_cutoff': [0.05, 0.1]
            },
            'multi_scale': {
                'iou_threshold': [0.3, 0.5, 0.7]
            },
            'confidence_threshold': {
                'iou_threshold': [0.3, 0.5, 0.7],
                'base_confidence': [0.3, 0.5, 0.7],
                'adaptive_threshold': [True, False]
            }
        }
    
    def optimize_single_strategy(
        self,
        ensemble: AdvancedEnsemble,
        strategy_name: str,
        param_grid: Optional[Dict] = None,
        gt_file: str = "detections.txt",
        max_evaluations: int = 50
    ) -> OptimizationResult:
        """
        Optimize parameters for a single strategy.
        
        Args:
            ensemble: AdvancedEnsemble instance with loaded detections
            strategy_name: Name of strategy to optimize
            param_grid: Parameter grid to search. If None, uses default.
            gt_file: Ground truth file name
            max_evaluations: Maximum number of parameter combinations to evaluate
            
        Returns:
            OptimizationResult with best parameters and performance
        """
        if param_grid is None:
            param_grid = self.default_param_grids.get(strategy_name, {})
        
        if not param_grid:
            # No parameters to optimize, just evaluate with defaults
            results = ensemble.run_strategy(strategy_name)
            evaluation = self.evaluator.evaluate_predictions(results, gt_file)
            score = self._extract_metric_score(evaluation)
            
            return OptimizationResult(
                best_strategy=strategy_name,
                best_params={},
                best_score=score,
                all_results={strategy_name: evaluation},
                optimization_metric=self.optimization_metric
            )
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Limit evaluations if too many combinations
        if len(param_combinations) > max_evaluations:
            # Random sample for large parameter spaces
            import random
            param_combinations = random.sample(param_combinations, max_evaluations)
        
        # Evaluate each parameter combination
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = {}
        all_results = {}
        
        for params in param_combinations:
            try:
                # Set strategy parameters
                ensemble.set_strategy_params(strategy_name, **params)
                
                # Run strategy
                results = ensemble.run_strategy(strategy_name)
                
                # Evaluate
                evaluation = self.evaluator.evaluate_predictions(results, gt_file)
                score = self._extract_metric_score(evaluation)
                
                # Track results
                param_key = f"{strategy_name}_{self._params_to_string(params)}"
                all_results[param_key] = {
                    'evaluation': evaluation,
                    'parameters': params,
                    'score': score
                }
                
                # Update best if improved
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"Error evaluating {strategy_name} with params {params}: {e}")
                continue
        
        return OptimizationResult(
            best_strategy=strategy_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric
        )
    
    def optimize_all_strategies(
        self,
        ensemble: AdvancedEnsemble,
        strategy_names: Optional[List[str]] = None,
        custom_param_grids: Optional[Dict[str, Dict]] = None,
        gt_file: str = "detections.txt",
        parallel: bool = True,
        max_workers: int = 4
    ) -> OptimizationResult:
        """
        Optimize all available strategies and select the best one.
        
        Args:
            ensemble: AdvancedEnsemble instance with loaded detections
            strategy_names: List of strategy names to optimize. If None, uses all available.
            custom_param_grids: Custom parameter grids for strategies
            gt_file: Ground truth file name
            parallel: Whether to run optimizations in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            OptimizationResult with best strategy and parameters
        """
        if strategy_names is None:
            strategy_names = list(ensemble.strategies.keys())
        
        if custom_param_grids is None:
            custom_param_grids = {}
        
        all_optimization_results = {}
        
        if parallel and len(strategy_names) > 1:
            # Parallel optimization
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_strategy = {}
                
                for strategy_name in strategy_names:
                    param_grid = custom_param_grids.get(strategy_name)
                    future = executor.submit(
                        self.optimize_single_strategy,
                        ensemble, strategy_name, param_grid, gt_file
                    )
                    future_to_strategy[future] = strategy_name
                
                for future in as_completed(future_to_strategy):
                    strategy_name = future_to_strategy[future]
                    try:
                        result = future.result()
                        all_optimization_results[strategy_name] = result
                    except Exception as e:
                        print(f"Error optimizing {strategy_name}: {e}")
        else:
            # Sequential optimization
            for strategy_name in strategy_names:
                param_grid = custom_param_grids.get(strategy_name)
                try:
                    result = self.optimize_single_strategy(
                        ensemble, strategy_name, param_grid, gt_file
                    )
                    all_optimization_results[strategy_name] = result
                except Exception as e:
                    print(f"Error optimizing {strategy_name}: {e}")
        
        # Find overall best strategy
        best_overall_score = float('-inf') if self.maximize else float('inf')
        best_strategy_name = None
        best_strategy_result = None
        
        for strategy_name, result in all_optimization_results.items():
            if (self.maximize and result.best_score > best_overall_score) or \
               (not self.maximize and result.best_score < best_overall_score):
                best_overall_score = result.best_score
                best_strategy_name = strategy_name
                best_strategy_result = result
        
        # Combine all results
        combined_results = {}
        for strategy_name, result in all_optimization_results.items():
            combined_results.update(result.all_results)
        
        if best_strategy_result is None:
            raise ValueError("No successful strategy optimizations")
        
        return OptimizationResult(
            best_strategy=best_strategy_name,
            best_params=best_strategy_result.best_params,
            best_score=best_overall_score,
            all_results=combined_results,
            optimization_metric=self.optimization_metric
        )
    
    def suggest_strategy_for_data(
        self,
        ensemble: AdvancedEnsemble,
        gt_file: str = "detections.txt",
        data_characteristics: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Suggest optimal strategy based on data characteristics.
        
        Args:
            ensemble: AdvancedEnsemble instance with loaded detections
            gt_file: Ground truth file name
            data_characteristics: Optional characteristics of the data
            
        Returns:
            Tuple of (strategy_name, suggested_params)
        """
        # Analyze data characteristics if not provided
        if data_characteristics is None:
            data_characteristics = self._analyze_data_characteristics(ensemble, gt_file)
        
        # Strategy suggestions based on characteristics
        strategy_suggestions = []
        
        # Small objects -> use adaptive strategies
        if data_characteristics.get('avg_object_size', 0.1) < 0.05:
            strategy_suggestions.extend(['adaptive_threshold', 'multi_scale'])
        
        # High density -> use clustering strategies
        if data_characteristics.get('detection_density', 0) > 50:
            strategy_suggestions.extend(['dbscan', 'density_adaptive'])
        
        # Low agreement -> use consensus strategies
        if data_characteristics.get('model_agreement', 0.5) < 0.3:
            strategy_suggestions.extend(['bayesian', 'soft_voting'])
        
        # High confidence variance -> use confidence-based strategies
        if data_characteristics.get('confidence_variance', 0.1) > 0.2:
            strategy_suggestions.extend(['confidence_threshold', 'high_confidence_first'])
        
        # Default suggestions if no specific characteristics
        if not strategy_suggestions:
            strategy_suggestions = ['weighted_vote', 'affirmative_nms', 'majority_vote_2']
        
        # Optimize top suggestions
        top_strategies = strategy_suggestions[:3]  # Limit to top 3 for efficiency
        optimization_result = self.optimize_all_strategies(
            ensemble, top_strategies, gt_file=gt_file
        )
        
        return optimization_result.best_strategy, optimization_result.best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all combinations of parameters from grid."""
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for value_combination in itertools.product(*values):
            param_dict = dict(zip(keys, value_combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _extract_metric_score(self, evaluation: Dict) -> float:
        """Extract the optimization metric score from evaluation results."""
        if self.optimization_metric in evaluation.get('overall_metrics', {}):
            return evaluation['overall_metrics'][self.optimization_metric]
        else:
            raise ValueError(f"Metric {self.optimization_metric} not found in evaluation results")
    
    def _params_to_string(self, params: Dict) -> str:
        """Convert parameters dictionary to string representation."""
        return "_".join([f"{k}={v}" for k, v in sorted(params.items())])
    
    def _analyze_data_characteristics(
        self,
        ensemble: AdvancedEnsemble,
        gt_file: str
    ) -> Dict:
        """Analyze characteristics of the detection data."""
        # Load ground truth
        ground_truth = self.evaluator.load_ground_truth(gt_file)
        
        # Get all model detections
        all_detections = []
        for model_detections in ensemble.detections.values():
            all_detections.extend(model_detections)
        
        characteristics = {}
        
        if ground_truth:
            # Average object size in ground truth
            gt_areas = [gt.area for gt in ground_truth]
            characteristics['avg_object_size'] = np.mean(gt_areas) if gt_areas else 0.1
            characteristics['object_size_variance'] = np.var(gt_areas) if gt_areas else 0.0
        
        if all_detections:
            # Detection density
            characteristics['detection_density'] = len(all_detections)
            
            # Confidence statistics
            confidences = [det.confidence for det in all_detections]
            characteristics['avg_confidence'] = np.mean(confidences)
            characteristics['confidence_variance'] = np.var(confidences)
        
        # Model agreement (simplified - could be more sophisticated)
        if len(ensemble.detections) > 1:
            # Quick agreement estimate based on detection counts
            detection_counts = [len(dets) for dets in ensemble.detections.values()]
            count_variance = np.var(detection_counts)
            avg_count = np.mean(detection_counts)
            characteristics['model_agreement'] = 1.0 / (1.0 + count_variance / max(avg_count, 1))
        else:
            characteristics['model_agreement'] = 1.0
        
        return characteristics


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning using Bayesian optimization or grid search.
    """
    
    def __init__(self, evaluator: Evaluator):
        """Initialize hyperparameter tuner."""
        self.evaluator = evaluator
    
    def bayesian_optimization(
        self,
        ensemble: AdvancedEnsemble,
        strategy_name: str,
        param_bounds: Dict[str, Tuple[float, float]],
        gt_file: str = "detections.txt",
        n_iterations: int = 20
    ) -> Dict:
        """
        Bayesian optimization for continuous parameters.
        
        Note: This is a placeholder for Bayesian optimization.
        In practice, you would use libraries like scikit-optimize or Optuna.
        
        Args:
            ensemble: AdvancedEnsemble instance
            strategy_name: Strategy to optimize
            param_bounds: Bounds for continuous parameters
            gt_file: Ground truth file
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Placeholder implementation
        # In practice, integrate with scikit-optimize or similar
        
        best_params = {}
        best_score = float('-inf')
        
        # Simple random search as placeholder
        import random
        
        for _ in range(n_iterations):
            # Sample random parameters within bounds
            params = {}
            for param_name, (low, high) in param_bounds.items():
                params[param_name] = random.uniform(low, high)
            
            try:
                # Set parameters and evaluate
                ensemble.set_strategy_params(strategy_name, **params)
                results = ensemble.run_strategy(strategy_name)
                evaluation = self.evaluator.evaluate_predictions(results, gt_file)
                score = evaluation['overall_metrics']['map_50']
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception:
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'strategy': strategy_name
        }