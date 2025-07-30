from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

from .detection import Detection
from ..utils.io import read_detections, save_detections, load_ground_truth, validate_ground_truth_structure
from ..strategies import (
    MajorityVoting, WeightedVoting, NMSStrategy, 
    AffirmativeNMS, DBSCANClustering, SoftVoting, BayesianFusion,
    DistanceWeightedVoting, CentroidClustering,
    ConfidenceThresholdVoting, ConfidenceWeightedNMS, HighConfidenceFirst,
    AdaptiveThresholdStrategy, DensityAdaptiveStrategy, MultiScaleStrategy, ConsensusRankingStrategy
)


class EnsembleVoting:
    """Main ensemble class for combining object detection results with ground truth support."""
    
    def __init__(self, labels_dir: str = "labels", output_dir: str = "labels/unified", gt_dir: Optional[str] = None):
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = []
        self.detections = {}
        
        # Ground truth support
        self.gt_dir = gt_dir or str(self.labels_dir / "GT")
        self.ground_truth = {}
        self._gt_cache = {}
        
        # Initialize basic strategies
        self.strategies = {
            'majority_vote_2': MajorityVoting(min_votes=2),
            'weighted_vote': WeightedVoting(),
            'nms': NMSStrategy(),
            'unanimous': MajorityVoting(min_votes=999)  # Will be adjusted based on model count
        }
    
    def load_detections(self, filename: str = "detections.txt") -> Dict[str, List[Detection]]:
        """Load detections from all model directories."""
        self.detections = {}
        self.models = []
        
        for model_dir in self.labels_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in ["unified", "GT"]:
                model_name = model_dir.name
                self.models.append(model_name)
                file_path = model_dir / filename
                
                if file_path.exists():
                    self.detections[model_name] = read_detections(str(file_path), model_name)
                    print(f"Loaded {len(self.detections[model_name])} detections from {model_name}")
                else:
                    print(f"Warning: {file_path} not found")
                    self.detections[model_name] = []
        
        # Adjust unanimous strategy
        if 'unanimous' in self.strategies and isinstance(self.strategies['unanimous'], MajorityVoting):
            self.strategies['unanimous'].min_votes = len(self.models)
        
        return self.detections
    
    def load_all_image_detections(self, default_confidence: float = 1.0) -> Dict[str, List[Detection]]:
        """Load detections for all images from all model directories (image mode)."""
        from collections import defaultdict
        
        self.detections = {}
        self.models = []
        image_detections = defaultdict(dict)
        
        # Find all model directories
        model_dirs = [d for d in self.labels_dir.iterdir() 
                     if d.is_dir() and d.name not in ["unified", "__pycache__", "GT"]]
        
        print(f"Loading detections from {len(model_dirs)} models...")
        for model_dir in tqdm(model_dirs, desc="Loading models"):
            model_name = model_dir.name
            self.models.append(model_name)
            
            # Count txt files first
            txt_files = list(model_dir.glob("*.txt"))
            
            # Load all .txt files in the model directory
            for txt_file in tqdm(txt_files, desc=f"  {model_name}", leave=False):
                image_name = txt_file.stem
                detections = read_detections(str(txt_file), model_name, default_confidence, image_name)
                image_detections[image_name][model_name] = detections
        
        # Flatten detections for compatibility with existing ensemble code
        for image_name, model_data in image_detections.items():
            for model_name, image_dets in model_data.items():
                if model_name not in self.detections:
                    self.detections[model_name] = []
                self.detections[model_name].extend(image_dets)
        
        # Print summary
        for model_name, dets in self.detections.items():
            print(f"Loaded {len(dets)} detections from {model_name}")
        
        # Adjust unanimous strategy
        if 'unanimous' in self.strategies and isinstance(self.strategies['unanimous'], MajorityVoting):
            self.strategies['unanimous'].min_votes = len(self.models)
        
        return self.detections
    
    def add_strategy(self, name: str, strategy):
        """Add a custom strategy."""
        self.strategies[name] = strategy
    
    def run_strategy(self, strategy_name: str, **kwargs) -> List[Detection]:
        """Run a specific ensemble strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.merge(self.detections, **kwargs)
    
    def run_all_strategies(self, filename: str = "detections.txt", 
                          save_results: bool = True) -> Dict[str, List[Detection]]:
        """Run all registered strategies."""
        self.load_detections(filename)
        
        results = {}
        for strategy_name, strategy in self.strategies.items():
            print(f"\nRunning {strategy_name} strategy...")
            merged = strategy.merge(self.detections)
            results[strategy_name] = merged
            
            if save_results:
                output_filename = f"{filename.split('.')[0]}_{strategy_name}.txt"
                save_detections(merged, str(self.output_dir / output_filename))
                print(f"  {strategy_name}: {len(merged)} detections saved")
        
        return results
    
    def run_strategy_per_image(self, strategy_name: str, image_detections: Dict[str, Dict[str, List[Detection]]] = None, **kwargs) -> Dict[str, List[Detection]]:
        """Run a specific ensemble strategy on each image separately.
        
        Args:
            strategy_name: Name of the strategy to run
            image_detections: Optional pre-loaded image detections. If None, will load from disk.
            **kwargs: Additional arguments for the strategy
            
        Returns:
            Dict mapping image names to merged detections for that image
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Load image detections if not provided
        if image_detections is None:
            from collections import defaultdict
            image_detections = defaultdict(dict)
            
            # Load all image detections
            for model_dir in self.labels_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in ["unified", "__pycache__", "GT"]:
                    model_name = model_dir.name
                    
                    for txt_file in model_dir.glob("*.txt"):
                        image_name = txt_file.stem
                        detections = read_detections(str(txt_file), model_name, 1.0, image_name)
                        image_detections[image_name][model_name] = detections
        
        strategy = self.strategies[strategy_name]
        results = {}
        
        # Process each image separately
        for image_name, model_dets in image_detections.items():
            # Run strategy on this image's detections
            if model_dets:  # Only process if we have detections for this image
                merged = strategy.merge(model_dets, **kwargs)
                
                # Ensure all merged detections have the correct image_name
                for det in merged:
                    det.image_name = image_name
                
                results[image_name] = merged
        
        return results
    
    def save_statistics(self, results: Dict[str, List[Detection]], 
                       filename: str = "ensemble_stats.json"):
        """Save statistics about ensemble results."""
        from collections import Counter
        import numpy as np
        
        stats = {}
        for strategy, detections in results.items():
            class_counts = Counter([det.class_id for det in detections])
            stats[strategy] = {
                "total_detections": len(detections),
                "unique_classes": len(class_counts),
                "avg_confidence": float(np.mean([det.confidence for det in detections])) if detections else 0,
                "class_distribution": dict(class_counts)
            }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {output_path}")
    
    # Ground Truth Support Methods
    
    def load_ground_truth(self, gt_file: str = "detections.txt") -> List[Detection]:
        """Load ground truth detections."""
        return load_ground_truth(self.gt_dir, gt_file, self._gt_cache)
    
    def has_ground_truth(self, gt_file: str = "detections.txt") -> bool:
        """Check if ground truth is available."""
        try:
            self.load_ground_truth(gt_file)
            return True
        except FileNotFoundError:
            return False
    
    def validate_gt_structure(self) -> Dict[str, bool]:
        """Validate ground truth directory structure."""
        return validate_ground_truth_structure(str(self.labels_dir))
    
    def evaluate_strategy_with_gt(self, strategy_name: str, gt_file: str = "detections.txt") -> Optional[Dict]:
        """Evaluate a strategy against ground truth."""
        if not self.has_ground_truth(gt_file):
            print(f"Warning: Ground truth not available at {self.gt_dir}/{gt_file}")
            return None
        
        try:
            # Import here to avoid circular imports
            from ..evaluation.evaluator import Evaluator
            
            # Run strategy
            strategy_results = self.run_strategy(strategy_name)
            
            # Set up evaluator
            evaluator = Evaluator(gt_dir=self.gt_dir)
            
            # Evaluate
            evaluation_results = evaluator.evaluate_predictions(strategy_results, gt_file)
            evaluation_results['strategy_name'] = strategy_name
            
            return evaluation_results
            
        except Exception as e:
            print(f"Error evaluating {strategy_name} with GT: {e}")
            return None
    
    def evaluate_all_strategies_with_gt(self, gt_file: str = "detections.txt") -> Dict[str, Dict]:
        """Evaluate all strategies against ground truth."""
        if not self.has_ground_truth(gt_file):
            print(f"Warning: Ground truth not available at {self.gt_dir}/{gt_file}")
            return {}
        
        results = {}
        for strategy_name in self.strategies.keys():
            print(f"Evaluating {strategy_name} against ground truth...")
            evaluation = self.evaluate_strategy_with_gt(strategy_name, gt_file)
            if evaluation:
                results[strategy_name] = evaluation
        
        return results
    
    def find_best_strategy_with_gt(self, gt_file: str = "detections.txt", metric: str = 'f1_score') -> Optional[Tuple[str, Dict]]:
        """Find the best performing strategy using ground truth evaluation."""
        evaluations = self.evaluate_all_strategies_with_gt(gt_file)
        
        if not evaluations:
            return None
        
        best_strategy = None
        best_score = -1
        best_results = None
        
        for strategy_name, results in evaluations.items():
            if 'overall_metrics' in results and metric in results['overall_metrics']:
                score = results['overall_metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
                    best_results = results
        
        if best_strategy:
            print(f"Best strategy: {best_strategy} ({metric}: {best_score:.4f})")
            return best_strategy, best_results
        else:
            print(f"Could not determine best strategy using metric: {metric}")
            return None


class AdvancedEnsemble(EnsembleVoting):
    """Advanced ensemble with additional strategies and ground truth support."""
    
    def __init__(self, labels_dir: str = "labels", output_dir: str = "labels/unified", gt_dir: Optional[str] = None):
        super().__init__(labels_dir, output_dir, gt_dir)
        
        # Add advanced strategies
        self.strategies.update({
            'affirmative_nms': AffirmativeNMS(min_models=2),
            'dbscan': DBSCANClustering(),
            'soft_voting': SoftVoting(),
            'bayesian': BayesianFusion(),
            'distance_weighted': DistanceWeightedVoting(),
            'centroid_clustering': CentroidClustering(),
            'confidence_threshold': ConfidenceThresholdVoting(),
            'confidence_weighted_nms': ConfidenceWeightedNMS(),
            'high_confidence_first': HighConfidenceFirst(),
            'adaptive_threshold': AdaptiveThresholdStrategy(),
            'density_adaptive': DensityAdaptiveStrategy(),
            'multi_scale': MultiScaleStrategy(),
            'consensus_ranking': ConsensusRankingStrategy()
        })
    
    def set_strategy_params(self, strategy_name: str, **params):
        """Update parameters for a specific strategy."""
        if strategy_name in self.strategies:
            for key, value in params.items():
                if hasattr(self.strategies[strategy_name], key):
                    setattr(self.strategies[strategy_name], key, value)