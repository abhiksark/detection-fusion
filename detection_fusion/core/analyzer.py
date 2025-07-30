import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

from .detection import Detection
from ..utils.io import read_detections, load_class_names, load_ground_truth, validate_ground_truth_structure
from ..utils.metrics import calculate_iou


class MultiModelAnalyzer:
    """Analyzer for comparing detections across multiple models with ground truth support."""
    
    def __init__(self, labels_dir: str = "labels", iou_threshold: float = 0.5, gt_dir: Optional[str] = None):
        self.labels_dir = Path(labels_dir)
        self.iou_threshold = iou_threshold
        self.models = []
        self.detections = {}
        self.class_names = {}
        
        # Ground truth support
        self.gt_dir = gt_dir or str(self.labels_dir / "GT")
        self.ground_truth = {}
        self._gt_cache = {}
    
    def load_detections(self, filename: str = "detections.txt", default_confidence: float = 1.0) -> Dict[str, List[Detection]]:
        """Load detections from all model directories."""
        self.models = []
        self.detections = {}
        
        for model_dir in self.labels_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in ["unified", "__pycache__", "GT"]:
                model_name = model_dir.name
                self.models.append(model_name)
                file_path = model_dir / filename
                
                if file_path.exists():
                    self.detections[model_name] = read_detections(str(file_path), model_name, default_confidence)
                else:
                    print(f"Warning: {file_path} not found")
                    self.detections[model_name] = []
        
        return self.detections
    
    def load_all_image_detections(self, default_confidence: float = 1.0) -> Dict[str, Dict[str, List[Detection]]]:
        """Load detections for all images from all model directories.
        
        Args:
            default_confidence: Default confidence value for detections missing confidence scores
        
        Returns:
            Dict with structure: {image_name: {model_name: [detections]}}
        """
        self.models = []
        self.image_detections = defaultdict(dict)
        
        # Find all model directories
        model_dirs = [d for d in self.labels_dir.iterdir() 
                     if d.is_dir() and d.name not in ["unified", "__pycache__", "GT"]]
        
        print(f"Loading detections from {len(model_dirs)} models...")
        for model_dir in tqdm(model_dirs, desc="Loading models"):
            model_name = model_dir.name
            self.models.append(model_name)
            
            # Count txt files first for progress bar
            txt_files = list(model_dir.glob("*.txt"))
            
            # Load all .txt files in the model directory
            for txt_file in tqdm(txt_files, desc=f"  {model_name}", leave=False):
                image_name = txt_file.stem
                detections = read_detections(str(txt_file), model_name, default_confidence, image_name)
                self.image_detections[image_name][model_name] = detections
        
        # Convert to regular dict for return
        return dict(self.image_detections)
    
    def load_class_names(self, class_names_file: Optional[str] = None):
        """Load class names from file."""
        self.class_names = load_class_names(class_names_file)
    
    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """Compare detections between two specific models."""
        detections1 = self.detections[model1]
        detections2 = self.detections[model2]
        
        if not detections1 or not detections2:
            return {
                'total_matches': 0,
                'model1_unique': len(detections1),
                'model2_unique': len(detections2),
                'avg_iou': 0.0,
                'class_matches': {}
            }
        
        # Match detections
        matched1 = set()
        matched2 = set()
        matches = []
        class_matches = defaultdict(int)
        
        for i, det1 in enumerate(detections1):
            best_match_idx = -1
            best_iou = 0
            
            for j, det2 in enumerate(detections2):
                if j in matched2:
                    continue
                    
                if det1.class_id == det2.class_id:
                    iou = calculate_iou(det1.bbox, det2.bbox)
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_idx = j
            
            if best_match_idx != -1:
                matched1.add(i)
                matched2.add(best_match_idx)
                matches.append((i, best_match_idx, best_iou))
                class_matches[det1.class_id] += 1
        
        avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0
        
        return {
            'total_matches': len(matches),
            'model1_unique': len(detections1) - len(matched1),
            'model2_unique': len(detections2) - len(matched2),
            'avg_iou': float(avg_iou),
            'class_matches': dict(class_matches),
            'matches': matches
        }
    
    def compare_all_models(self) -> pd.DataFrame:
        """Generate comparison matrix for all model pairs."""
        results = []
        
        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models):
                if i < j:
                    comparison = self.compare_models(model1, model2)
                    results.append({
                        'model1': model1,
                        'model2': model2,
                        'total_matches': comparison['total_matches'],
                        'model1_unique': comparison['model1_unique'],
                        'model2_unique': comparison['model2_unique'],
                        'avg_iou': comparison['avg_iou']
                    })
        
        return pd.DataFrame(results)
    
    def get_class_statistics(self) -> pd.DataFrame:
        """Calculate detection statistics per class across all models."""
        class_stats = defaultdict(lambda: {model: 0 for model in self.models})
        
        for model, detections in self.detections.items():
            for det in detections:
                class_stats[det.class_id][model] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(class_stats).T
        df.index.name = 'class_id'
        df['class_name'] = df.index.map(lambda x: self.class_names.get(x, f"Class_{x}"))
        df['total'] = df[self.models].sum(axis=1)
        df['mean'] = df[self.models].mean(axis=1)
        df['std'] = df[self.models].std(axis=1)
        df['variance'] = df[self.models].var(axis=1)
        
        return df.sort_values('total', ascending=False)
    
    def get_confidence_statistics(self) -> pd.DataFrame:
        """Calculate confidence score statistics for each model."""
        conf_stats = []
        
        for model, detections in self.detections.items():
            if detections:
                confidences = [det.confidence for det in detections]
                conf_stats.append({
                    'model': model,
                    'count': len(confidences),
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences)),
                    'median': float(np.median(confidences)),
                    'q1': float(np.percentile(confidences, 25)),
                    'q3': float(np.percentile(confidences, 75))
                })
        
        return pd.DataFrame(conf_stats)
    
    def find_consensus_detections(self, min_models: int = 2) -> List[Detection]:
        """Find detections that multiple models agree on."""
        from ..strategies.voting import MajorityVoting
        
        voter = MajorityVoting(self.iou_threshold, min_votes=min_models)
        consensus = voter.merge(self.detections)
        
        return consensus
    
    def generate_report(self, output_file: str = "analysis_report.txt"):
        """Generate comprehensive analysis report."""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Multi-Model Object Detection Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Models analyzed: {', '.join(self.models)}\n")
            f.write(f"IoU threshold: {self.iou_threshold}\n\n")
            
            # Model statistics
            f.write("-"*80 + "\n")
            f.write("Model Statistics\n")
            f.write("-"*80 + "\n")
            for model in self.models:
                f.write(f"\n{model}:\n")
                f.write(f"  Total detections: {len(self.detections[model])}\n")
            
            # Confidence statistics
            f.write("\n" + "-"*80 + "\n")
            f.write("Confidence Statistics\n")
            f.write("-"*80 + "\n")
            conf_stats = self.get_confidence_statistics()
            f.write(conf_stats.to_string())
            
            # Pairwise comparisons
            f.write("\n\n" + "-"*80 + "\n")
            f.write("Pairwise Model Comparisons\n")
            f.write("-"*80 + "\n")
            comparison_df = self.compare_all_models()
            f.write(comparison_df.to_string())
            
            # Class statistics
            f.write("\n\n" + "-"*80 + "\n")
            f.write("Top 20 Most Detected Classes\n")
            f.write("-"*80 + "\n")
            class_stats = self.get_class_statistics()
            f.write(class_stats.head(20).to_string())
            
            # High variance classes
            f.write("\n\n" + "-"*80 + "\n")
            f.write("Classes with High Variance Between Models\n")
            f.write("-"*80 + "\n")
            high_variance = class_stats[class_stats['variance'] > class_stats['variance'].quantile(0.9)]
            f.write(high_variance.to_string())
            
        print(f"Report saved to {output_file}")
    
    # Ground Truth Support Methods
    
    def load_ground_truth(self, gt_file: str = "detections.txt") -> List[Detection]:
        """Load ground truth detections."""
        return load_ground_truth(self.gt_dir, gt_file, self._gt_cache)
    
    def validate_gt_structure(self) -> Dict[str, bool]:
        """Validate ground truth directory structure."""
        return validate_ground_truth_structure(str(self.labels_dir))
    
    def has_ground_truth(self, gt_file: str = "detections.txt") -> bool:
        """Check if ground truth is available."""
        try:
            self.load_ground_truth(gt_file)
            return True
        except FileNotFoundError:
            return False
    
    def compare_with_gt(self, model_name: str, gt_file: str = "detections.txt") -> Dict[str, Any]:
        """Compare a specific model's detections with ground truth."""
        if model_name not in self.detections:
            raise ValueError(f"Model {model_name} not found in loaded detections")
        
        try:
            ground_truth = self.load_ground_truth(gt_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_dir}/{gt_file}")
        
        model_detections = self.detections[model_name]
        
        # Use the same comparison logic as compare_models but with GT
        matched_pred = set()
        matched_gt = set()
        matches = []
        class_matches = defaultdict(int)
        
        for i, pred in enumerate(model_detections):
            best_match_idx = -1
            best_iou = 0
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                    
                if pred.class_id == gt.class_id:
                    iou = calculate_iou(pred.bbox, gt.bbox)
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_idx = j
            
            if best_match_idx != -1:
                matched_pred.add(i)
                matched_gt.add(best_match_idx)
                matches.append((i, best_match_idx, best_iou))
                class_matches[pred.class_id] += 1
        
        avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0
        
        # Calculate metrics
        tp = len(matches)
        fp = len(model_detections) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'model': model_name,
            'total_matches': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': float(avg_iou),
            'class_matches': dict(class_matches),
            'matches': matches,
            'total_gt': len(ground_truth),
            'total_pred': len(model_detections)
        }
    
    def compare_all_models_with_gt(self, gt_file: str = "detections.txt") -> pd.DataFrame:
        """Compare all models against ground truth."""
        if not self.has_ground_truth(gt_file):
            raise FileNotFoundError(f"Ground truth not available: {self.gt_dir}/{gt_file}")
        
        results = []
        for model_name in self.models:
            try:
                comparison = self.compare_with_gt(model_name, gt_file)
                results.append(comparison)
            except Exception as e:
                print(f"Error comparing {model_name} with GT: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('f1_score', ascending=False)
    
    def analyze_consensus_vs_gt(self, min_models: int = 2, gt_file: str = "detections.txt") -> Dict[str, Any]:
        """Analyze how consensus detections perform against ground truth."""
        if not self.has_ground_truth(gt_file):
            raise FileNotFoundError(f"Ground truth not available: {self.gt_dir}/{gt_file}")
        
        # Get consensus detections
        consensus_detections = self.find_consensus_detections(min_models)
        
        # Load ground truth
        ground_truth = self.load_ground_truth(gt_file)
        
        # Compare consensus with GT (similar to compare_with_gt but for consensus)
        matched_cons = set()
        matched_gt = set()
        matches = []
        
        for i, cons in enumerate(consensus_detections):
            best_match_idx = -1
            best_iou = 0
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                    
                if cons.class_id == gt.class_id:
                    iou = calculate_iou(cons.bbox, gt.bbox)
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_idx = j
            
            if best_match_idx != -1:
                matched_cons.add(i)
                matched_gt.add(best_match_idx)
                matches.append((i, best_match_idx, best_iou))
        
        # Calculate metrics
        tp = len(matches)
        fp = len(consensus_detections) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0
        
        return {
            'consensus_type': f'min_{min_models}_models',
            'consensus_detections': len(consensus_detections),
            'ground_truth_objects': len(ground_truth),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': float(avg_iou),
            'matches': matches
        }
    
    def generate_gt_report(self, output_file: str = "gt_analysis_report.txt", gt_file: str = "detections.txt"):
        """Generate comprehensive ground truth analysis report."""
        if not self.has_ground_truth(gt_file):
            print("Warning: Ground truth not available. Generating model-only report.")
            self.generate_report(output_file)
            return
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Multi-Model Object Detection Analysis Report (with Ground Truth)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Models analyzed: {', '.join(self.models)}\n")
            f.write(f"Ground truth file: {gt_file}\n")
            f.write(f"IoU threshold: {self.iou_threshold}\n\n")
            
            # Ground truth statistics
            try:
                gt_detections = self.load_ground_truth(gt_file)
                f.write("-"*80 + "\n")
                f.write("Ground Truth Statistics\n")
                f.write("-"*80 + "\n")
                f.write(f"Total ground truth objects: {len(gt_detections)}\n")
                
                # GT class distribution
                gt_classes = Counter([det.class_id for det in gt_detections])
                f.write(f"Classes in GT: {len(gt_classes)}\n")
                f.write("Top GT classes:\n")
                for class_id, count in gt_classes.most_common(10):
                    class_name = self.class_names.get(class_id, f"Class_{class_id}")
                    f.write(f"  {class_name}: {count}\n")
                f.write("\n")
                
            except Exception as e:
                f.write(f"Error loading ground truth: {e}\n\n")
            
            # Model vs GT comparisons
            f.write("-"*80 + "\n")
            f.write("Model Performance vs Ground Truth\n")
            f.write("-"*80 + "\n")
            try:
                gt_comparison_df = self.compare_all_models_with_gt(gt_file)
                f.write(gt_comparison_df.to_string())
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error in GT comparison: {e}\n\n")
            
            # Consensus vs GT analysis
            f.write("-"*80 + "\n")
            f.write("Consensus Detection Analysis vs Ground Truth\n")
            f.write("-"*80 + "\n")
            try:
                for min_models in [2, 3, len(self.models)]:
                    if min_models <= len(self.models):
                        consensus_analysis = self.analyze_consensus_vs_gt(min_models, gt_file)
                        f.write(f"\nConsensus (min {min_models} models):\n")
                        f.write(f"  Precision: {consensus_analysis['precision']:.4f}\n")
                        f.write(f"  Recall: {consensus_analysis['recall']:.4f}\n")
                        f.write(f"  F1-Score: {consensus_analysis['f1_score']:.4f}\n")
                        f.write(f"  Avg IoU: {consensus_analysis['avg_iou']:.4f}\n")
            except Exception as e:
                f.write(f"Error in consensus analysis: {e}\n")
            
            # Regular analysis sections (model comparisons, etc.)
            f.write("\n" + "-"*80 + "\n")
            f.write("Inter-Model Comparisons\n")
            f.write("-"*80 + "\n")
            comparison_df = self.compare_all_models()
            f.write(comparison_df.to_string())
            
        print(f"Ground truth analysis report saved to {output_file}")