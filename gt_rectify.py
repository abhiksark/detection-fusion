"""
GT Rectification System for DetectionFusion

This module provides functionality to identify potential ground truth annotation errors
by analyzing disagreements between ensemble consensus and ground truth labels.
"""

import shutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import json
from tqdm import tqdm

from detection_fusion.core.detection import Detection
from detection_fusion.core.ensemble import AdvancedEnsemble
from detection_fusion.evaluation.evaluator import Evaluator
from detection_fusion.utils.io import save_detections, load_yaml_config


class GTRectificationError:
    """Represents a potential error in ground truth annotation."""
    
    def __init__(self, image_name: str, gt_detection: Optional[Detection], 
                 consensus_detection: Optional[Detection], error_type: str,
                 confidence_score: float, strategy_agreement: Dict[str, bool],
                 supporting_models: List[str]):
        self.image_name = image_name
        self.gt_detection = gt_detection
        self.consensus_detection = consensus_detection
        self.error_type = error_type  # 'missing_in_gt', 'extra_in_gt', 'wrong_class', 'wrong_box'
        self.confidence_score = confidence_score
        self.strategy_agreement = strategy_agreement
        self.supporting_models = supporting_models
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'image_name': self.image_name,
            'gt_detection': self.gt_detection.to_dict() if self.gt_detection else None,
            'consensus_detection': self.consensus_detection.to_dict() if self.consensus_detection else None,
            'error_type': self.error_type,
            'confidence_score': self.confidence_score,
            'strategy_agreement': self.strategy_agreement,
            'supporting_models': self.supporting_models
        }


class GTRectifier:
    """
    Ground Truth Rectification System
    
    Identifies potential annotation errors by comparing ensemble consensus
    with ground truth labels across all available strategies.
    
    Supports two modes:
    - minimize_error: Conservative approach, only flags high-confidence errors
    - maximize_error: Aggressive approach, identifies more potential issues for review
    """
    
    def __init__(self, labels_dir: str, gt_dir: str, images_dir: str, output_dir: str,
                 iou_threshold: float = 0.5, confidence_threshold: float = 0.3,
                 min_strategy_agreement: int = 3, mode: str = "minimize_error"):
        """
        Initialize GT Rectifier.
        
        Args:
            labels_dir: Directory containing model predictions
            gt_dir: Directory containing ground truth labels
            images_dir: Directory containing images
            output_dir: Directory for ensemble outputs
            iou_threshold: IoU threshold for matching detections
            confidence_threshold: Minimum confidence for considering detections
            min_strategy_agreement: Minimum strategies that must agree for consensus
            mode: Rectification mode ('minimize_error' or 'maximize_error')
                - minimize_error: Conservative, only flags high-confidence errors
                - maximize_error: Aggressive, flags more potential issues for review
        """
        self.labels_dir = labels_dir
        self.gt_dir = gt_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.min_strategy_agreement = min_strategy_agreement
        self.mode = mode
        
        # Validate mode
        if mode not in ["minimize_error", "maximize_error"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'minimize_error' or 'maximize_error'")
        
        # Set mode-specific parameters
        self._setup_mode_parameters()
        
        # Initialize ensemble and evaluator
        self.ensemble = AdvancedEnsemble(labels_dir, output_dir, gt_dir)
        self.evaluator = Evaluator(iou_threshold=iou_threshold, gt_dir=gt_dir)
        
        # Storage for analysis results
        self.rectification_errors: List[GTRectificationError] = []
        self.image_scores: Dict[str, float] = {}
        self.strategy_results: Dict[str, Dict[str, List[Detection]]] = {}
    
    def _setup_mode_parameters(self) -> None:
        """Set up mode-specific parameters for error detection."""
        if self.mode == "minimize_error":
            # Conservative mode: Only flag high-confidence errors
            self.error_confidence_threshold = 0.7    # Higher threshold for reporting errors
            self.consensus_strength_weight = 0.8     # Strong emphasis on consensus strength
            self.strategy_diversity_weight = 0.6     # Moderate emphasis on strategy diversity
            self.isolation_weight = 0.4              # Lower emphasis on isolation
            self.min_error_reports = 2               # Require multiple error indicators
            
        elif self.mode == "maximize_error":
            # Aggressive mode: Flag more potential issues for review
            self.error_confidence_threshold = 0.3    # Lower threshold for reporting errors
            self.consensus_strength_weight = 0.4     # Lower emphasis on consensus strength
            self.strategy_diversity_weight = 0.8     # Higher emphasis on strategy diversity
            self.isolation_weight = 0.7              # Higher emphasis on isolation
            self.min_error_reports = 1               # Single error indicator sufficient
    
    def load_data(self, detection_filename: str = "detections.txt", image_mode: bool = True) -> None:
        """Load all required data for rectification analysis.
        
        Args:
            detection_filename: Name of detection file (used in single-file mode)
            image_mode: If True, load per-image detections; if False, use single file per model
        """
        print("Loading model predictions...")
        if image_mode:
            # Load all image detections
            self.ensemble.load_all_image_detections()
            self.image_detections = self._organize_detections_by_image()
        else:
            # Legacy single-file mode
            self.ensemble.load_detections(detection_filename)
            self.image_detections = None
        
        print("Loading ground truth data...")
        self.evaluator.load_ground_truth()
        
        print(f"Loaded {len(self.ensemble.models)} models and GT for {len(self.evaluator.ground_truth)} images")
    
    def _organize_detections_by_image(self) -> Dict[str, Dict[str, List[Detection]]]:
        """Organize loaded detections by image name."""
        from collections import defaultdict
        image_detections = defaultdict(dict)
        
        # The ensemble has already loaded detections with image_name set
        for model_name, detections in self.ensemble.detections.items():
            for det in detections:
                if det.image_name:
                    if model_name not in image_detections[det.image_name]:
                        image_detections[det.image_name][model_name] = []
                    image_detections[det.image_name][model_name].append(det)
        
        return dict(image_detections)
    
    def run_all_strategies(self) -> Dict[str, Dict[str, List[Detection]]]:
        """
        Run all ensemble strategies and organize results by image.
        
        Returns:
            Dictionary mapping strategy names to image-wise detection results
        """
        print("Running all ensemble strategies...")
        
        organized_results = {}
        
        if self.image_detections:
            # Image mode: run strategies per image
            for strategy_name in tqdm(self.ensemble.strategies.keys(), desc="Running strategies"):
                organized_results[strategy_name] = {}
                
                # Process each image separately
                for image_name, model_dets in self.image_detections.items():
                    if model_dets:  # Only process if we have detections
                        strategy = self.ensemble.strategies[strategy_name]
                        merged = strategy.merge(model_dets)
                        
                        # Ensure detections have image_name set
                        for det in merged:
                            det.image_name = image_name
                        
                        organized_results[strategy_name][image_name] = merged
        else:
            # Legacy mode: use existing method
            strategy_results = self.ensemble.run_all_strategies(save_results=False)
            
            # Organize by image name
            for strategy_name, detections in strategy_results.items():
                organized_results[strategy_name] = {}
                
                # Group detections by image
                image_groups = defaultdict(list)
                for detection in detections:
                    image_name = getattr(detection, 'image_name', 'default_image')
                    image_groups[image_name].append(detection)
                
                organized_results[strategy_name] = dict(image_groups)
        
        self.strategy_results = organized_results
        return organized_results
    
    def build_consensus(self, image_name: str) -> List[Detection]:
        """
        Build consensus detections for a specific image across all strategies.
        
        Args:
            image_name: Name of the image to analyze
            
        Returns:
            List of consensus detections with confidence scores
        """
        # Collect all detections from all strategies for this image
        all_detections = []
        strategy_votes = defaultdict(list)  # detection_key -> [strategy_names]
        
        for strategy_name, image_results in self.strategy_results.items():
            if image_name in image_results:
                detections = image_results[image_name]
                for detection in detections:
                    if detection.confidence >= self.confidence_threshold:
                        # Create unique key for this detection
                        detection_key = self._create_detection_key(detection)
                        strategy_votes[detection_key].append(strategy_name)
                        all_detections.append((detection, strategy_name))
        
        # Build consensus based on strategy agreement
        consensus_detections = []
        processed_keys = set()
        
        for detection, strategy_name in all_detections:
            detection_key = self._create_detection_key(detection)
            
            if detection_key in processed_keys:
                continue
            
            supporting_strategies = strategy_votes[detection_key]
            
            if len(supporting_strategies) >= self.min_strategy_agreement:
                # Calculate consensus confidence (average of supporting strategies)
                consensus_confidence = np.mean([
                    det.confidence for det, strat in all_detections 
                    if self._create_detection_key(det) == detection_key
                ])
                
                # Create consensus detection
                consensus_det = Detection(
                    detection.class_id,
                    detection.x, detection.y, detection.w, detection.h,
                    consensus_confidence,
                    f"consensus_{len(supporting_strategies)}_strategies"
                )
                consensus_det.supporting_strategies = supporting_strategies
                consensus_detections.append(consensus_det)
                processed_keys.add(detection_key)
        
        return consensus_detections
    
    def _create_detection_key(self, detection: Detection) -> str:
        """Create unique key for detection matching."""
        # Round coordinates for fuzzy matching
        x_round = round(detection.x, 2)
        y_round = round(detection.y, 2)
        w_round = round(detection.w, 2)
        h_round = round(detection.h, 2)
        
        return f"{detection.class_id}_{x_round}_{y_round}_{w_round}_{h_round}"
    
    def analyze_image_errors(self, image_name: str) -> List[GTRectificationError]:
        """
        Analyze potential GT errors for a specific image.
        
        Args:
            image_name: Name of the image to analyze
            
        Returns:
            List of potential rectification errors
        """
        errors = []
        
        # Get ground truth and consensus for this image
        gt_detections = self.evaluator.ground_truth.get(image_name, [])
        consensus_detections = self.build_consensus(image_name)
        
        # Find GT detections missing from consensus (potential false positives in GT)
        for gt_det in gt_detections:
            matched = False
            best_match = None
            best_iou = 0
            
            for cons_det in consensus_detections:
                if gt_det.class_id == cons_det.class_id:
                    from detection_fusion.utils.metrics import calculate_iou
                    iou = calculate_iou(gt_det.bbox, cons_det.bbox)
                    if iou > self.iou_threshold:
                        matched = True
                        break
                    elif iou > best_iou:
                        best_iou = iou
                        best_match = cons_det
            
            if not matched:
                # Calculate confidence that this GT detection is wrong
                confidence_score = self._calculate_error_confidence(
                    gt_det, None, "extra_in_gt", image_name
                )
                
                if confidence_score > self.error_confidence_threshold:
                    error = GTRectificationError(
                        image_name, gt_det, best_match, "extra_in_gt",
                        confidence_score, {}, []
                    )
                    errors.append(error)
        
        # Find consensus detections missing from GT (potential false negatives in GT)
        for cons_det in consensus_detections:
            matched = False
            best_match = None
            best_iou = 0
            
            for gt_det in gt_detections:
                if cons_det.class_id == gt_det.class_id:
                    from detection_fusion.utils.metrics import calculate_iou
                    iou = calculate_iou(cons_det.bbox, gt_det.bbox)
                    if iou > self.iou_threshold:
                        matched = True
                        break
                    elif iou > best_iou:
                        best_iou = iou
                        best_match = gt_det
            
            if not matched:
                # Calculate confidence that GT is missing this detection
                confidence_score = self._calculate_error_confidence(
                    None, cons_det, "missing_in_gt", image_name
                )
                
                if confidence_score > self.error_confidence_threshold:
                    supporting_strategies = getattr(cons_det, 'supporting_strategies', [])
                    error = GTRectificationError(
                        image_name, best_match, cons_det, "missing_in_gt",
                        confidence_score, {}, supporting_strategies
                    )
                    errors.append(error)
        
        return errors
    
    def _calculate_error_confidence(self, gt_det: Optional[Detection], 
                                  cons_det: Optional[Detection], 
                                  error_type: str, image_name: str) -> float:
        """
        Calculate confidence score for a potential GT error.
        
        Args:
            gt_det: Ground truth detection (if any)
            cons_det: Consensus detection (if any)
            error_type: Type of error
            image_name: Image name
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []
        
        if error_type == "missing_in_gt" and cons_det:
            # Factors supporting that GT is missing this detection
            
            # 1. Consensus confidence (weighted by mode)
            confidence_factors.append(cons_det.confidence * self.consensus_strength_weight)
            
            # 2. Number of supporting strategies (weighted by mode)
            supporting_strategies = getattr(cons_det, 'supporting_strategies', [])
            strategy_factor = min(len(supporting_strategies) / len(self.strategy_results), 1.0)
            confidence_factors.append(strategy_factor * self.consensus_strength_weight)
            
            # 3. Diversity of supporting strategies (weighted by mode)
            diversity_factor = self._calculate_strategy_diversity(supporting_strategies)
            confidence_factors.append(diversity_factor * self.strategy_diversity_weight)
            
        elif error_type == "extra_in_gt" and gt_det:
            # Factors supporting that this GT detection is wrong
            
            # 1. Low consensus support (inverted, weighted by mode)
            consensus_support = self._get_consensus_support_for_detection(gt_det, image_name)
            confidence_factors.append((1.0 - consensus_support) * self.consensus_strength_weight)
            
            # 2. Isolation factor (weighted by mode)
            isolation_factor = self._calculate_isolation_factor(gt_det, image_name)
            confidence_factors.append(isolation_factor * self.isolation_weight)
        
        # Combine factors using weighted average
        if not confidence_factors:
            return 0.0
        
        return np.mean(confidence_factors)
    
    def _calculate_strategy_diversity(self, supporting_strategies: List[str]) -> float:
        """Calculate diversity score based on strategy types."""
        strategy_types = {
            'voting': ['majority_vote_2', 'majority_vote_3', 'weighted_voting'],
            'nms': ['nms', 'affirmative_nms'],
            'clustering': ['dbscan'],
            'probabilistic': ['soft_voting', 'bayesian'],
            'adaptive': ['adaptive_threshold', 'density_adaptive', 'multi_scale'],
            'confidence': ['confidence_threshold', 'confidence_weighted_nms', 'high_confidence_first']
        }
        
        represented_types = set()
        for strategy in supporting_strategies:
            for strategy_type, strategies in strategy_types.items():
                if strategy in strategies:
                    represented_types.add(strategy_type)
                    break
        
        return len(represented_types) / len(strategy_types)
    
    def _get_consensus_support_for_detection(self, detection: Detection, image_name: str) -> float:
        """Get consensus support score for a specific detection."""
        support_count = 0
        total_strategies = len(self.strategy_results)
        
        for strategy_name, image_results in self.strategy_results.items():
            if image_name in image_results:
                strategy_detections = image_results[image_name]
                for strat_det in strategy_detections:
                    from detection_fusion.utils.metrics import calculate_iou
                    if (detection.class_id == strat_det.class_id and 
                        calculate_iou(detection.bbox, strat_det.bbox) > self.iou_threshold):
                        support_count += 1
                        break
        
        return support_count / total_strategies if total_strategies > 0 else 0.0
    
    def _calculate_isolation_factor(self, detection: Detection, image_name: str) -> float:
        """Calculate how isolated a detection is from other detections."""
        gt_detections = self.evaluator.ground_truth.get(image_name, [])
        
        if len(gt_detections) <= 1:
            return 1.0  # Highly isolated
        
        distances = []
        for other_det in gt_detections:
            if other_det != detection:
                # Calculate center distance
                dist = np.sqrt((detection.x - other_det.x)**2 + (detection.y - other_det.y)**2)
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Normalize by image diagonal
        avg_distance = np.mean(distances)
        image_diagonal = np.sqrt(2)  # For normalized coordinates
        
        return min(avg_distance / (image_diagonal / 4), 1.0)
    
    def run_full_analysis(self, detection_filename: str = "detections.txt", image_mode: bool = True) -> Dict[str, Any]:
        """
        Run complete GT rectification analysis.
        
        Args:
            detection_filename: Name of detection files to analyze
            image_mode: If True, use per-image detection files
            
        Returns:
            Complete analysis results
        """
        print("Starting GT Rectification Analysis...")
        
        # Load data
        self.load_data(detection_filename, image_mode)
        
        # Run all strategies
        self.run_all_strategies()
        
        # Analyze each image
        all_errors = []
        image_scores = {}
        
        # Get all unique image names
        all_images = set()
        all_images.update(self.evaluator.ground_truth.keys())
        
        # Add images from strategy results
        for strategy_results in self.strategy_results.values():
            all_images.update(strategy_results.keys())
        
        print(f"Analyzing {len(all_images)} images...")
        
        for image_name in tqdm(all_images, desc="Analyzing images"):
            print(f"Analyzing image: {image_name}")
            
            # Analyze errors for this image
            image_errors = self.analyze_image_errors(image_name)
            all_errors.extend(image_errors)
            
            # Calculate overall correctness score for this image
            correctness_score = self._calculate_image_correctness_score(image_name, image_errors)
            image_scores[image_name] = correctness_score
        
        self.rectification_errors = all_errors
        self.image_scores = image_scores
        
        # Generate summary statistics
        analysis_results = {
            'total_images': len(all_images),
            'total_errors_found': len(all_errors),
            'error_types': Counter([error.error_type for error in all_errors]),
            'most_problematic_images': self._get_most_problematic_images(5),
            'most_reliable_images': self._get_most_reliable_images(5),
            'image_scores': image_scores,
            'detailed_errors': [error.to_dict() for error in all_errors]
        }
        
        print(f"Analysis complete! Found {len(all_errors)} potential GT errors across {len(all_images)} images")
        
        return analysis_results
    
    def _calculate_image_correctness_score(self, image_name: str, 
                                         image_errors: List[GTRectificationError]) -> float:
        """
        Calculate F1-based correctness score for an image accounting for both 
        missing labels (recall) and extra labels (precision).
        
        Returns:
            F1-based correctness score (0-1, higher is better)
        """
        # Get GT and consensus detections for this image
        gt_detections = self.evaluator.ground_truth.get(image_name, [])
        consensus_detections = self.build_consensus(image_name)
        
        if not gt_detections and not consensus_detections:
            return 1.0  # Perfect match: both empty
        
        if not gt_detections:
            return 0.0  # No GT but consensus has detections (all false positives)
        
        if not consensus_detections:
            return 0.0  # GT has detections but no consensus (all false negatives)
        
        # Calculate matches using IoU threshold
        from detection_fusion.utils.metrics import calculate_iou
        
        gt_matched = set()
        consensus_matched = set()
        
        # Find matches between GT and consensus
        for i, gt_det in enumerate(gt_detections):
            for j, cons_det in enumerate(consensus_detections):
                if gt_det.class_id == cons_det.class_id:
                    iou = calculate_iou(gt_det.bbox, cons_det.bbox)
                    if iou > self.iou_threshold:
                        gt_matched.add(i)
                        consensus_matched.add(j)
                        break  # Each GT detection matches at most one consensus
        
        # Calculate precision, recall, and F1
        true_positives = len(gt_matched)
        false_positives = len(consensus_detections) - len(consensus_matched)
        false_negatives = len(gt_detections) - len(gt_matched)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score
    
    def _calculate_detailed_f1_metrics(self, image_name: str) -> Dict[str, float]:
        """Calculate detailed F1 metrics (precision, recall, TP, FP, FN) for an image."""
        # Get GT and consensus detections for this image
        gt_detections = self.evaluator.ground_truth.get(image_name, [])
        consensus_detections = self.build_consensus(image_name)
        
        if not gt_detections and not consensus_detections:
            return {'precision': 1.0, 'recall': 1.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        
        if not gt_detections:
            return {'precision': 0.0, 'recall': 1.0, 'true_positives': 0, 'false_positives': len(consensus_detections), 'false_negatives': 0}
        
        if not consensus_detections:
            return {'precision': 1.0, 'recall': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': len(gt_detections)}
        
        # Calculate matches using IoU threshold
        from detection_fusion.utils.metrics import calculate_iou
        
        gt_matched = set()
        consensus_matched = set()
        
        # Find matches between GT and consensus
        for i, gt_det in enumerate(gt_detections):
            for j, cons_det in enumerate(consensus_detections):
                if gt_det.class_id == cons_det.class_id:
                    iou = calculate_iou(gt_det.bbox, cons_det.bbox)
                    if iou > self.iou_threshold:
                        gt_matched.add(i)
                        consensus_matched.add(j)
                        break  # Each GT detection matches at most one consensus
        
        # Calculate metrics
        true_positives = len(gt_matched)
        false_positives = len(consensus_detections) - len(consensus_matched)
        false_negatives = len(gt_detections) - len(gt_matched)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        
        return {
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _get_most_problematic_images(self, top_n: int) -> List[Tuple[str, float]]:
        """Get images with lowest F1-based correctness scores."""
        sorted_images = sorted(self.image_scores.items(), key=lambda x: x[1])
        return sorted_images[:top_n]
    
    def _get_most_reliable_images(self, top_n: int) -> List[Tuple[str, float]]:
        """Get images with highest F1-based correctness scores."""
        sorted_images = sorted(self.image_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_images[:top_n]
    
    def create_rectified_dataset(self, output_dir: str, 
                               include_most_correct: int = 50,
                               include_most_incorrect: int = 50) -> None:
        """
        Create organized dataset directory with most correct and incorrect labels.
        
        Args:
            output_dir: Output directory for rectified dataset
            include_most_correct: Number of most correct images to include
            include_most_incorrect: Number of most incorrect images to include
        """
        print(f"Creating rectified dataset in {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        correct_dir = output_path / "most_correct"
        incorrect_dir = output_path / "most_incorrect"
        
        for subdir in [correct_dir, incorrect_dir]:
            (subdir / "images").mkdir(parents=True, exist_ok=True)
            (subdir / "labels").mkdir(parents=True, exist_ok=True)
            (subdir / "analysis").mkdir(parents=True, exist_ok=True)
            (subdir / "unified").mkdir(parents=True, exist_ok=True)
        
        # Get most reliable and problematic images
        most_reliable = self._get_most_reliable_images(include_most_correct)
        most_problematic = self._get_most_problematic_images(include_most_incorrect)
        
        # Copy most correct dataset
        print(f"Copying {len(most_reliable)} most correct images...")
        self._copy_image_dataset(most_reliable, correct_dir, "correct")
        
        # Copy most incorrect dataset  
        print(f"Copying {len(most_problematic)} most incorrect images...")
        self._copy_image_dataset(most_problematic, incorrect_dir, "incorrect")
        
        # Generate summary reports
        self._generate_dataset_summary(output_path, most_reliable, most_problematic)
        
        print(f"Rectified dataset created successfully in {output_dir}")
    
    def _copy_image_dataset(self, image_list: List[Tuple[str, float]], 
                           target_dir: Path, dataset_type: str) -> None:
        """Copy images and labels to target directory."""
        
        for i, (image_name, score) in enumerate(tqdm(image_list, desc=f"Copying {dataset_type} dataset")):
            # Copy image if it exists
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_copied = False
            
            for ext in image_extensions:
                image_path = Path(self.images_dir) / f"{image_name}{ext}"
                if image_path.exists():
                    target_image = target_dir / "images" / f"{i:03d}_{image_name}{ext}"
                    shutil.copy2(image_path, target_image)
                    image_copied = True
                    break
            
            if not image_copied:
                print(f"Warning: Image not found for {image_name}")
            
            # Copy GT labels
            gt_detections = self.evaluator.ground_truth.get(image_name, [])
            if gt_detections:
                target_label = target_dir / "labels" / f"{i:03d}_{image_name}.txt"
                save_detections(gt_detections, str(target_label))
            
            # Generate analysis file
            analysis_file = target_dir / "analysis" / f"{i:03d}_{image_name}_analysis.json"
            self._generate_image_analysis(image_name, score, analysis_file)
            
            # Save consensus detections as unified predictions
            consensus_detections = self.build_consensus(image_name)
            if consensus_detections:
                unified_label = target_dir / "unified" / f"{i:03d}_{image_name}.txt"
                save_detections(consensus_detections, str(unified_label))
    
    def _generate_image_analysis(self, image_name: str, score: float, 
                               output_file: Path) -> None:
        """Generate detailed analysis for a specific image."""
        # Get errors for this image
        image_errors = [error for error in self.rectification_errors 
                       if error.image_name == image_name]
        
        # Get GT and consensus detections
        gt_detections = self.evaluator.ground_truth.get(image_name, [])
        consensus_detections = self.build_consensus(image_name)
        
        # Calculate detailed F1 metrics for this image
        f1_metrics = self._calculate_detailed_f1_metrics(image_name)
        
        analysis = {
            'image_name': image_name,
            'f1_score': score,  # This is now the F1-based correctness score
            'precision': f1_metrics['precision'],
            'recall': f1_metrics['recall'],
            'true_positives': f1_metrics['true_positives'],
            'false_positives': f1_metrics['false_positives'],
            'false_negatives': f1_metrics['false_negatives'],
            'gt_detection_count': len(gt_detections),
            'consensus_detection_count': len(consensus_detections),
            'errors_found': len(image_errors),
            'error_details': [error.to_dict() for error in image_errors],
            'gt_detections': [det.to_dict() for det in gt_detections],
            'consensus_detections': [det.to_dict() for det in consensus_detections],
            'recommendations': self._generate_recommendations(image_name, image_errors)
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _generate_recommendations(self, image_name: str, 
                                errors: List[GTRectificationError]) -> List[str]:
        """Generate recommendations for fixing GT errors."""
        recommendations = []
        
        for error in errors:
            if error.error_type == "missing_in_gt":
                recommendations.append(
                    f"Consider adding detection: class {error.consensus_detection.class_id} "
                    f"at ({error.consensus_detection.x:.3f}, {error.consensus_detection.y:.3f}) "
                    f"with size ({error.consensus_detection.w:.3f}, {error.consensus_detection.h:.3f}). "
                    f"Supported by {len(error.supporting_models)} strategies with confidence {error.confidence_score:.3f}"
                )
            elif error.error_type == "extra_in_gt":
                recommendations.append(
                    f"Consider removing GT detection: class {error.gt_detection.class_id} "
                    f"at ({error.gt_detection.x:.3f}, {error.gt_detection.y:.3f}) "
                    f"with size ({error.gt_detection.w:.3f}, {error.gt_detection.h:.3f}). "
                    f"Low consensus support with confidence {error.confidence_score:.3f}"
                )
        
        if not recommendations:
            recommendations.append("No specific recommendations - labels appear correct.")
        
        return recommendations
    
    def _generate_dataset_summary(self, output_dir: Path, 
                                most_reliable: List[Tuple[str, float]],
                                most_problematic: List[Tuple[str, float]]) -> None:
        """Generate summary reports for the rectified dataset."""
        
        summary = {
            'dataset_creation_info': {
                'total_images_analyzed': len(self.image_scores),
                'total_errors_found': len(self.rectification_errors),
                'most_reliable_count': len(most_reliable),
                'most_problematic_count': len(most_problematic),
                'analysis_parameters': {
                    'iou_threshold': self.iou_threshold,
                    'confidence_threshold': self.confidence_threshold,
                    'min_strategy_agreement': self.min_strategy_agreement,
                    'mode': self.mode
                }
            },
            'error_statistics': {
                'error_types': dict(Counter([error.error_type for error in self.rectification_errors])),
                'avg_confidence_per_error_type': self._calculate_avg_confidence_per_error_type(),
                'most_common_error_classes': self._get_most_common_error_classes()
            },
            'most_reliable_images': [
                {'image_name': name, 'correctness_score': score}
                for name, score in most_reliable
            ],
            'most_problematic_images': [
                {'image_name': name, 'correctness_score': score}
                for name, score in most_problematic
            ],
            'recommendations': self._generate_global_recommendations()
        }
        
        # Save summary as JSON
        summary_file = output_dir / "rectification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable report
        self._generate_human_readable_report(output_dir, summary)
    
    def _calculate_avg_confidence_per_error_type(self) -> Dict[str, float]:
        """Calculate average confidence score per error type."""
        error_confidences = defaultdict(list)
        
        for error in self.rectification_errors:
            error_confidences[error.error_type].append(error.confidence_score)
        
        return {
            error_type: np.mean(confidences)
            for error_type, confidences in error_confidences.items()
        }
    
    def _get_most_common_error_classes(self) -> List[Tuple[int, int]]:
        """Get most common classes involved in errors."""
        error_classes = []
        
        for error in self.rectification_errors:
            if error.gt_detection:
                error_classes.append(error.gt_detection.class_id)
            if error.consensus_detection:
                error_classes.append(error.consensus_detection.class_id)
        
        return Counter(error_classes).most_common(10)
    
    def _generate_global_recommendations(self) -> List[str]:
        """Generate global recommendations for improving GT quality."""
        recommendations = []
        
        # Analyze error patterns
        error_types = Counter([error.error_type for error in self.rectification_errors])
        
        if error_types.get('missing_in_gt', 0) > error_types.get('extra_in_gt', 0):
            recommendations.append(
                "The dataset appears to have more missing annotations than false positives. "
                "Consider reviewing labeling guidelines to ensure complete object annotation."
            )
        elif error_types.get('extra_in_gt', 0) > error_types.get('missing_in_gt', 0):
            recommendations.append(
                "The dataset appears to have more false positive annotations. "
                "Consider reviewing labeling guidelines to ensure annotation quality."
            )
        
        # Class-specific recommendations
        common_error_classes = self._get_most_common_error_classes()
        if common_error_classes:
            recommendations.append(
                f"Classes with most errors: {[cls for cls, count in common_error_classes[:3]]}. "
                "Consider additional training or clearer guidelines for these classes."
            )
        
        # Overall quality assessment based on F1 scores
        avg_f1_score = np.mean(list(self.image_scores.values()))
        if avg_f1_score > 0.9:
            recommendations.append(f"Excellent dataset quality! Average F1 score: {avg_f1_score:.3f} (>90%).")
        elif avg_f1_score > 0.8:
            recommendations.append(f"Good dataset quality. Average F1 score: {avg_f1_score:.3f} (>80%).")
        elif avg_f1_score > 0.6:
            recommendations.append(f"Moderate dataset quality. Average F1 score: {avg_f1_score:.3f}. Consider reviewing annotations.")
        else:
            recommendations.append(f"Dataset quality needs improvement. Average F1 score: {avg_f1_score:.3f} (<60%).")
        
        return recommendations
    
    def _generate_human_readable_report(self, output_dir: Path, summary: Dict[str, Any]) -> None:
        """Generate human-readable analysis report."""
        
        report_file = output_dir / "rectification_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("GT RECTIFICATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            info = summary['dataset_creation_info']
            f.write(f"Total Images Analyzed: {info['total_images_analyzed']}\n")
            f.write(f"Total Errors Found: {info['total_errors_found']}\n")
            f.write(f"Most Reliable Images: {info['most_reliable_count']}\n")
            f.write(f"Most Problematic Images: {info['most_problematic_count']}\n")
            f.write(f"Analysis Mode: {info['analysis_parameters']['mode']}\n")
            f.write(f"Scoring Method: F1-based (precision & recall)\n\n")
            
            # Error statistics
            f.write("ERROR STATISTICS\n")
            f.write("-" * 20 + "\n")
            error_stats = summary['error_statistics']
            f.write("Error Types:\n")
            for error_type, count in error_stats['error_types'].items():
                f.write(f"  {error_type}: {count}\n")
            
            f.write("\nAverage Confidence per Error Type:\n")
            for error_type, conf in error_stats['avg_confidence_per_error_type'].items():
                f.write(f"  {error_type}: {conf:.3f}\n")
            
            f.write(f"\nMost Common Error Classes: {error_stats['most_common_error_classes'][:5]}\n\n")
            
            # Top problematic images
            f.write("MOST PROBLEMATIC IMAGES\n")
            f.write("-" * 25 + "\n")
            for img_info in summary['most_problematic_images'][:10]:
                f.write(f"  {img_info['image_name']}: {img_info['correctness_score']:.3f}\n")
            
            f.write("\nMOST RELIABLE IMAGES\n")
            f.write("-" * 20 + "\n")
            for img_info in summary['most_reliable_images'][:10]:
                f.write(f"  {img_info['image_name']}: {img_info['correctness_score']:.3f}\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Review the 'most_incorrect' and 'most_correct' directories for detailed analysis.\n")
            f.write("Each image has an associated analysis JSON file with specific recommendations.\n")


def main():
    """Main CLI interface for GT rectification."""
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="DetectionFusion GT Rectification System (image-by-image mode by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration files (recommended)
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_balanced.yaml

  # Configuration with command line overrides
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml --mode maximize_error
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_custom.yaml --most-incorrect 100

  # Basic rectification analysis (conservative mode)
  python gt_rectify.py --labels-dir labels --gt-dir gt --images-dir images --output-dir rectified

  # Aggressive mode for comprehensive error detection
  python gt_rectify.py --labels-dir labels --gt-dir gt --images-dir images \\
    --output-dir rectified --mode maximize_error

  # Conservative mode with custom parameters
  python gt_rectify.py --labels-dir labels --gt-dir gt --images-dir images \\
    --output-dir rectified --mode minimize_error --iou-threshold 0.6 \\
    --confidence-threshold 0.4 --min-strategy-agreement 4

Directory Structure Expected:
  labels/
    ├── model1/
    │   ├── image1.txt  (image mode - default)
    │   ├── image2.txt
    │   └── ...
    ├── model2/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── model3/
        ├── image1.txt
        ├── image2.txt
        └── ...
  
  gt/
    ├── image1.txt  (or GT/detections.txt for single file)
    ├── image2.txt
    └── ...
  
  images/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg

Output Structure:
  rectified_dataset/
    ├── most_correct/
    │   ├── images/         # Most reliable images
    │   ├── labels/         # Corresponding GT labels
    │   └── analysis/       # Detailed analysis per image
    ├── most_incorrect/
    │   ├── images/         # Most problematic images  
    │   ├── labels/         # Corresponding GT labels
    │   └── analysis/       # Detailed analysis per image
    ├── rectification_summary.json
    └── rectification_report.txt
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file. If provided, config values are used as defaults and can be overridden by command line arguments."
    )
    
    # Required arguments (can be provided via config file)
    parser.add_argument(
        "--labels-dir", 
        help="Directory containing model prediction subdirectories"
    )
    parser.add_argument(
        "--gt-dir", 
        help="Directory containing ground truth labels"
    )
    parser.add_argument(
        "--images-dir", 
        help="Directory containing images"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory for rectified dataset and analysis"
    )
    
    # Optional parameters
    parser.add_argument(
        "--detection-file", 
        default="detections.txt", 
        help="Name of detection files to analyze (default: detections.txt)"
    )
    parser.add_argument(
        "--iou-threshold", 
        type=float, 
        default=0.5, 
        help="IoU threshold for matching detections (default: 0.5)"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.3, 
        help="Minimum confidence threshold for considering detections (default: 0.3)"
    )
    parser.add_argument(
        "--min-strategy-agreement", 
        type=int, 
        default=3, 
        help="Minimum number of strategies that must agree for consensus (default: 3)"
    )
    parser.add_argument(
        "--most-correct", 
        type=int, 
        default=50, 
        help="Number of most correct images to include in dataset (default: 50)"
    )
    parser.add_argument(
        "--most-incorrect", 
        type=int, 
        default=50, 
        help="Number of most incorrect images to include in dataset (default: 50)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["minimize_error", "maximize_error"],
        default="minimize_error",
        help="Rectification mode: minimize_error (conservative) or maximize_error (aggressive) (default: minimize_error)"
    )
    
    # Analysis options
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output during analysis"
    )
    parser.add_argument(
        "--skip-dataset-creation", 
        action="store_true", 
        help="Only run analysis without creating rectified dataset"
    )
    parser.add_argument(
        "--single-file-mode",
        action="store_true",
        help="Use single detection file per model (legacy behavior, default is image-by-image)"
    )
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    config = {}
    if args.config:
        try:
            config = load_yaml_config(args.config)
        except Exception as e:
            print(f"❌ Error loading configuration file: {e}")
            sys.exit(1)
    
    # Get configuration values with command line overrides
    def get_config_value(arg_name, config_path, default_value=None):
        """Get value from args, config, or default (in that order of priority)."""
        arg_value = getattr(args, arg_name.replace('-', '_'), None)
        if arg_value is not None:
            return arg_value
        
        # Navigate config path (e.g., 'gt_rectification.mode')
        config_value = config
        for key in config_path.split('.'):
            if isinstance(config_value, dict) and key in config_value:
                config_value = config_value[key]
            else:
                config_value = None
                break
        
        return config_value if config_value is not None else default_value
    
    # Extract configuration values
    labels_dir = get_config_value('labels-dir', 'gt_rectification.labels_dir', 'labels')
    gt_dir = get_config_value('gt-dir', 'gt_rectification.gt_dir', 'labels/GT')
    images_dir = get_config_value('images-dir', 'gt_rectification.images_dir', 'images')
    output_dir = get_config_value('output-dir', 'gt_rectification.output_dir', 'rectified_dataset')
    
    # Validate required parameters
    if not all([labels_dir, gt_dir, images_dir, output_dir]):
        missing = []
        if not labels_dir:
            missing.append('--labels-dir')
        if not gt_dir:
            missing.append('--gt-dir')  
        if not images_dir:
            missing.append('--images-dir')
        if not output_dir:
            missing.append('--output-dir')
        
        print(f"❌ Required arguments missing: {', '.join(missing)}")
        print("Provide them via command line arguments or configuration file.")
        sys.exit(1)
    
    # Validate input directories
    labels_path = Path(labels_dir)
    gt_path = Path(gt_dir)
    images_path = Path(images_dir)
    
    if not labels_path.exists():
        print(f"Error: Labels directory '{labels_dir}' does not exist")
        sys.exit(1)
    
    if not gt_path.exists():
        print(f"Error: Ground truth directory '{gt_dir}' does not exist")
        sys.exit(1)
    
    if not images_path.exists():
        print(f"Error: Images directory '{images_dir}' does not exist")  
        sys.exit(1)
    
    # Check for model subdirectories
    model_dirs = [d for d in labels_path.iterdir() if d.is_dir()]
    if not model_dirs:
        print(f"Error: No model subdirectories found in '{labels_dir}'")
        print("Expected structure: labels/model1/, labels/model2/, etc.")
        sys.exit(1)
    
    # Get other configuration values
    detection_file = get_config_value('detection-file', 'gt_rectification.detection_file', 'detections.txt')
    iou_threshold = get_config_value('iou-threshold', 'gt_rectification.iou_threshold', 0.5)
    confidence_threshold = get_config_value('confidence-threshold', 'gt_rectification.confidence_threshold', 0.3)
    min_strategy_agreement = get_config_value('min-strategy-agreement', 'gt_rectification.min_strategy_agreement', 3)
    most_correct = get_config_value('most-correct', 'gt_rectification.output.include_most_correct', 50)
    most_incorrect = get_config_value('most-incorrect', 'gt_rectification.output.include_most_incorrect', 50)
    mode = get_config_value('mode', 'gt_rectification.mode', 'minimize_error')
    skip_dataset = get_config_value('skip-dataset-creation', 'gt_rectification.skip_dataset_creation', False) or args.skip_dataset_creation
    
    print("=" * 60)
    print("DetectionFusion GT Rectification System")
    print("=" * 60)
    if args.config:
        print(f"Configuration file: {args.config}")
    print(f"Labels directory: {labels_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Detection file: {detection_file}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Min strategy agreement: {min_strategy_agreement}")
    print(f"Rectification mode: {mode}")
    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Load strategies from config if available
    strategies = None
    if 'gt_rectification' in config and 'strategies' in config['gt_rectification']:
        strategies = config['gt_rectification']['strategies']
        print(f"Using {len(strategies)} strategies from config: {', '.join(strategies)}")
    
    if not skip_dataset:
        print(f"Will create dataset with {most_correct} most correct and {most_incorrect} most incorrect images")
    
    print("-" * 60)
    
    try:
        # Initialize rectifier
        print("Initializing GT Rectifier...")
        rectifier = GTRectifier(
            labels_dir, 
            gt_dir, 
            images_dir,
            output_dir,
            iou_threshold, 
            confidence_threshold, 
            min_strategy_agreement,
            mode
        )
        
        # Run analysis
        print("Running comprehensive GT rectification analysis...")
        image_mode = not args.single_file_mode  # Use image mode by default
        results = rectifier.run_full_analysis(args.detection_file, image_mode)
        
        # Print summary results
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Total images analyzed: {results['total_images']}")
        print(f"Total potential errors found: {results['total_errors_found']}")
        
        print("\nError breakdown:")
        for error_type, count in results['error_types'].items():
            print(f"  {error_type}: {count}")
        
        print("\nMost problematic images (lowest F1 scores):")
        for image_name, score in results['most_problematic_images']:
            print(f"  {image_name}: {score:.3f}")
        
        print("\nMost reliable images (highest F1 scores):")
        for image_name, score in results['most_reliable_images']:
            print(f"  {image_name}: {score:.3f}")
        
        # Create rectified dataset if requested
        if not skip_dataset:
            print("\n" + "-" * 60)
            print("Creating rectified dataset...")
            rectifier.create_rectified_dataset(
                output_dir, 
                most_correct, 
                most_incorrect
            )
            
            print("\n✅ Rectified dataset created successfully!")
            print(f"📁 Output directory: {output_dir}")
            print(f"📊 Summary report: {output_dir}/rectification_report.txt")
            print(f"📋 Detailed analysis: {output_dir}/rectification_summary.json")
            print(f"🖼️  Most correct images: {output_dir}/most_correct/")
            print(f"⚠️  Most incorrect images: {output_dir}/most_incorrect/")
        else:
            print("\n✅ Analysis completed (dataset creation skipped)")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        # Generate quick recommendations
        error_types = results['error_types']
        total_errors = results['total_errors_found']
        total_images = results['total_images']
        
        print(f"📊 Analysis mode: {args.mode}")
        if args.mode == "minimize_error":
            print("   → Conservative approach: Only high-confidence errors flagged")
        else:
            print("   → Aggressive approach: More potential issues flagged for review")
        print()
        
        if total_errors == 0:
            if args.mode == "minimize_error":
                print("🎉 No high-confidence GT errors detected! Your ground truth appears reliable.")
                print("💡 Tip: Try --mode maximize_error for more comprehensive screening")
            else:
                print("🎉 No potential GT errors detected even in aggressive mode!")
                print("    Your ground truth appears very reliable.")
        else:
            error_rate = total_errors / total_images
            print(f"📈 Error rate: {error_rate:.1%} ({total_errors} errors in {total_images} images)")
            
            if args.mode == "minimize_error":
                if error_rate < 0.05:
                    print("✅ Very low error rate - GT quality is excellent")
                elif error_rate < 0.15:
                    print("✅ Low error rate - GT quality appears good")
                elif error_rate < 0.3:
                    print("⚠️  Moderate error rate - consider selective review")
                else:
                    print("🚨 High error rate - comprehensive GT review recommended")
                    print("💡 Note: These are high-confidence errors only")
            else:
                if error_rate < 0.1:
                    print("✅ Low error rate even in aggressive mode - GT quality is good")
                elif error_rate < 0.4:
                    print("⚠️  Moderate error rate - many potential issues to review")
                    print("💡 Tip: Use --mode minimize_error to focus on high-confidence errors")
                else:
                    print("🚨 High error rate - extensive GT review needed")
                    print("💡 Tip: Start with --mode minimize_error for prioritized fixes")
            
            # Specific recommendations based on error types
            missing_gt = error_types.get('missing_in_gt', 0)
            extra_gt = error_types.get('extra_in_gt', 0)
            
            if missing_gt > extra_gt * 2:
                print("📍 Main issue: Missing annotations in GT")
                print("   → Review labeling completeness guidelines")
            elif extra_gt > missing_gt * 2:
                print("📍 Main issue: False positive annotations in GT") 
                print("   → Review labeling accuracy guidelines")
            else:
                print("📍 Mixed annotation issues detected")
                print("   → Review both completeness and accuracy")
        
        print("\n💡 Next steps:")
        if not args.skip_dataset_creation:
            print(f"   1. Review images in {args.output_dir}/most_incorrect/")
            print("   2. Check analysis files for specific recommendations")
            print(f"   3. Use high-quality images from {args.output_dir}/most_correct/ as reference")
        print(f"   4. Read detailed report: {args.output_dir}/rectification_report.txt")
        
        print("\n🎯 GT Rectification Analysis Complete!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()