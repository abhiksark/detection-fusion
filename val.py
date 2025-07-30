#!/usr/bin/env python3
"""
DetectionFusion Assessment Tool

A comprehensive analysis and assessment tool for object detection models 
with ground truth evaluation capabilities. Provides detailed insights into model 
performance, error analysis, and optimal ensemble strategies.

Usage:
    python val.py --models model1 model2 model3 --analyze agreement
    python val.py --config configs/default_config.yaml --report full
    python val.py --models-dir labels/ --compare pairwise --plot
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from detection_fusion import MultiModelAnalyzer, AdvancedEnsemble, Evaluator, StrategyOptimizer
from detection_fusion.utils import load_yaml_config, validate_ground_truth_structure
from detection_fusion.evaluation.error_analysis import ErrorAnalyzer
from detection_fusion.core.detection import Detection


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assess and analyze object detection models (image-by-image comparison by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model assessment (image-by-image comparison by default)
  python val.py --models model1 model2 model3 --analyze agreement

  # Comprehensive analysis with plots (image mode)
  python val.py --models-dir labels/ --report full --plot --save-plots results/

  # Use single detection file per model (legacy mode)
  python val.py --models model1 model2 --single-file-mode --analyze agreement

  # Handle detections with missing confidence values  
  python val.py --models model1 model2 --default-confidence 0.8

  # Compare specific strategies
  python val.py --models model1 model2 --strategies weighted_vote bayesian --compare strategies

  # Pairwise model comparison
  python val.py --models-dir labels/ --compare pairwise --metric iou

  # Class-wise analysis
  python val.py --models model1 model2 model3 --analyze class-wise --classes person car bike

  # Confidence analysis
  python val.py --models-dir labels/ --analyze confidence --confidence-bins 10
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--models", 
        nargs="+", 
        help="List of model names to analyze"
    )
    input_group.add_argument(
        "--models-dir", 
        type=str,
        help="Directory containing model detection files"
    )
    input_group.add_argument(
        "--config", 
        type=str,
        help="YAML configuration file path"
    )
    
    # Analysis types
    parser.add_argument(
        "--analyze", 
        choices=["agreement", "confidence", "class-wise", "spatial", "performance", "evaluation", "error-analysis", "all"],
        default="agreement",
        help="Type of analysis to perform (default: agreement)"
    )
    parser.add_argument(
        "--compare", 
        choices=["pairwise", "strategies", "ensemble"],
        help="Comparison mode"
    )
    parser.add_argument(
        "--report", 
        choices=["summary", "full", "json"],
        default="summary",
        help="Report detail level (default: summary)"
    )
    
    # Strategy options (for strategy comparison)
    parser.add_argument(
        "--strategies", 
        nargs="+",
        help="Strategies to compare (requires --compare strategies)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--iou-threshold", 
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float,
        default=0.1,
        help="Minimum confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "--default-confidence",
        type=float,
        default=1.0,
        help="Default confidence value for detections missing confidence scores"
    )
    parser.add_argument(
        "--confidence-bins", 
        type=int,
        default=10,
        help="Number of confidence bins for analysis (default: 10)"
    )
    parser.add_argument(
        "--classes", 
        nargs="+",
        help="Specific classes to analyze (if not provided, analyze all)"
    )
    
    # Ground truth evaluation parameters
    parser.add_argument(
        "--gt", 
        action="store_true",
        help="Enable ground truth evaluation mode"
    )
    parser.add_argument(
        "--gt-dir", 
        type=str,
        help="Ground truth directory (default: labels/GT)"
    )
    parser.add_argument(
        "--gt-file", 
        type=str,
        default="detections.txt",
        help="Ground truth file name (default: detections.txt)"
    )
    parser.add_argument(
        "--metrics", 
        nargs="+",
        choices=["precision", "recall", "f1", "ap", "map", "map_50", "map_50_95"],
        help="Specific metrics to compute (requires --gt)"
    )
    parser.add_argument(
        "--error-analysis", 
        action="store_true",
        help="Enable detailed error analysis (requires --gt)"
    )
    parser.add_argument(
        "--optimize-strategy", 
        action="store_true",
        help="Find optimal strategy using ground truth (requires --gt)"
    )
    parser.add_argument(
        "--benchmark-strategies", 
        action="store_true",
        help="Benchmark all strategies against ground truth (requires --gt)"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str,
        default="assessment_report",
        help="Output file prefix (default: assessment_report)"
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--save-plots", 
        type=str,
        help="Directory to save plots (enables plotting)"
    )
    parser.add_argument(
        "--show-plots", 
        action="store_true",
        help="Display plots interactively"
    )
    
    # Utility options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Quiet mode"
    )
    parser.add_argument(
        "--single-file-mode",
        action="store_true",
        help="Use single detection file per model (default is image-by-image comparison)"
    )
    
    return parser.parse_args()


def setup_analyzer(args) -> MultiModelAnalyzer:
    """Set up the multi-model analyzer."""
    if args.config:
        config = load_yaml_config(args.config)
        ensemble_config = config.get("ensemble", {})
        labels_dir = ensemble_config.get("labels_dir", "labels")
    elif args.models_dir:
        labels_dir = args.models_dir
    else:
        labels_dir = "labels"
    
    analyzer = MultiModelAnalyzer(labels_dir=labels_dir)
    return analyzer


def load_and_filter_detections(analyzer: MultiModelAnalyzer, args) -> Dict[str, List[Detection]]:
    """Load and filter detections based on arguments."""
    if not args.single_file_mode:
        # Load detections for all images
        image_detections = analyzer.load_all_image_detections(args.default_confidence)
        
        # Flatten detections for compatibility with existing code
        detections = {}
        for image_name, model_data in image_detections.items():
            for model_name, image_dets in model_data.items():
                if args.models and model_name not in args.models:
                    continue
                    
                if model_name not in detections:
                    detections[model_name] = []
                
                # Apply confidence filtering
                filtered_dets = [
                    d for d in image_dets 
                    if d.confidence >= args.confidence_threshold
                ]
                detections[model_name].extend(filtered_dets)
        
        if not args.quiet:
            for model_name, model_dets in detections.items():
                print(f"✓ {model_name}: {len(model_dets)} total detections across all images")
        
        # Store image_detections for later use
        analyzer.image_detections = image_detections
        
    else:
        # Original behavior - load single detection file per model
        all_detections = analyzer.load_detections(default_confidence=args.default_confidence)
        
        if args.models:
            # Filter to specific models
            detections = {}
            for model_name in args.models:
                if model_name in all_detections:
                    model_detections = all_detections[model_name]
                    # Apply confidence filtering
                    filtered_detections = [
                        d for d in model_detections 
                        if d.confidence >= args.confidence_threshold
                    ]
                    detections[model_name] = filtered_detections
                    
                    if not args.quiet:
                        print(f"✓ {model_name}: {len(filtered_detections)} detections "
                              f"(filtered from {len(model_detections)})")
                else:
                    print(f"⚠️  Model '{model_name}' not found")
                    continue
        else:
            # Use all loaded models
            detections = {}
            for model_name, model_detections in all_detections.items():
                filtered_detections = [
                    d for d in model_detections 
                    if d.confidence >= args.confidence_threshold
                ]
                detections[model_name] = filtered_detections
    
    return detections


def analyze_model_agreement(detections: Dict[str, List[Detection]], args) -> Dict:
    """Analyze agreement between models."""
    if not args.quiet:
        print("🔍 Analyzing model agreement...")
    
    results = {
        "total_models": len(detections),
        "model_stats": {},
        "agreement_matrix": {},
        "consensus_detections": 0
    }
    
    # Individual model stats
    for model_name, model_detections in detections.items():
        results["model_stats"][model_name] = {
            "total_detections": len(model_detections),
            "avg_confidence": np.mean([d.confidence for d in model_detections]) if model_detections else 0,
            "confidence_std": np.std([d.confidence for d in model_detections]) if model_detections else 0,
            "class_distribution": {}
        }
        
        # Class distribution
        if model_detections:
            class_counts = {}
            for det in model_detections:
                class_counts[det.class_id] = class_counts.get(det.class_id, 0) + 1
            results["model_stats"][model_name]["class_distribution"] = class_counts
    
    # Pairwise agreement analysis
    model_names = list(detections.keys())
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            agreement_score = calculate_pairwise_agreement(
                detections[model1], detections[model2], args.iou_threshold
            )
            results["agreement_matrix"][f"{model1}_{model2}"] = agreement_score
    
    return results


def calculate_pairwise_agreement(detections1: List[Detection], detections2: List[Detection], 
                                iou_threshold: float) -> Dict:
    """Calculate agreement metrics between two sets of detections."""
    if not detections1 or not detections2:
        return {"jaccard": 0.0, "matches": 0, "model1_unique": len(detections1), 
                "model2_unique": len(detections2)}
    
    matches = 0
    matched_indices1 = set()
    matched_indices2 = set()
    
    # Find matches using IoU
    for i, det1 in enumerate(detections1):
        for j, det2 in enumerate(detections2):
            if j in matched_indices2:
                continue
            
            if det1.class_id == det2.class_id:
                iou = calculate_iou(det1, det2)
                if iou >= iou_threshold:
                    matches += 1
                    matched_indices1.add(i)
                    matched_indices2.add(j)
                    break
    
    # Calculate metrics
    union = len(detections1) + len(detections2) - matches
    jaccard = matches / union if union > 0 else 0.0
    
    return {
        "jaccard": jaccard,
        "matches": matches,
        "model1_unique": len(detections1) - len(matched_indices1),
        "model2_unique": len(detections2) - len(matched_indices2)
    }


def calculate_iou(det1: Detection, det2: Detection) -> float:
    """Calculate IoU between two detections."""
    # Convert to absolute coordinates for calculation
    x1_min, y1_min, x1_max, y1_max = det1.x, det1.y, det1.x + det1.w, det1.y + det1.h
    x2_min, y2_min, x2_max, y2_max = det2.x, det2.y, det2.x + det2.w, det2.y + det2.h
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = det1.w * det1.h
    area2 = det2.w * det2.h
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def analyze_confidence_distribution(detections: Dict[str, List[Detection]], args) -> Dict:
    """Analyze confidence score distributions."""
    if not args.quiet:
        print("📊 Analyzing confidence distributions...")
    
    results = {
        "model_confidence_stats": {},
        "confidence_bins": {},
        "calibration_analysis": {}
    }
    
    for model_name, model_detections in detections.items():
        if not model_detections:
            continue
        
        confidences = [d.confidence for d in model_detections]
        
        # Basic statistics
        results["model_confidence_stats"][model_name] = {
            "mean": np.mean(confidences),
            "median": np.median(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "percentiles": {
                "25": np.percentile(confidences, 25),
                "75": np.percentile(confidences, 75),
                "90": np.percentile(confidences, 90),
                "95": np.percentile(confidences, 95)
            }
        }
        
        # Histogram bins
        hist, bin_edges = np.histogram(confidences, bins=args.confidence_bins)
        results["confidence_bins"][model_name] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
    
    return results


def analyze_class_wise_performance(detections: Dict[str, List[Detection]], args) -> Dict:
    """Analyze performance by object class."""
    if not args.quiet:
        print("🏷️  Analyzing class-wise performance...")
    
    results = {
        "class_statistics": {},
        "model_class_performance": {}
    }
    
    # Collect all classes
    all_classes = set()
    for model_detections in detections.values():
        for det in model_detections:
            all_classes.add(det.class_id)
    
    # Filter classes if specified
    if args.classes:
        target_classes = [int(c) if c.isdigit() else c for c in args.classes]
        all_classes = all_classes.intersection(set(target_classes))
    
    # Analyze each class
    for class_id in sorted(all_classes):
        class_stats = {
            "total_detections": 0,
            "models_detecting": 0,
            "avg_confidence": 0,
            "model_contributions": {}
        }
        
        class_confidences = []
        models_with_class = 0
        
        for model_name, model_detections in detections.items():
            class_detections = [d for d in model_detections if d.class_id == class_id]
            
            if class_detections:
                models_with_class += 1
                model_conf = np.mean([d.confidence for d in class_detections])
                class_stats["model_contributions"][model_name] = {
                    "count": len(class_detections),
                    "avg_confidence": model_conf
                }
                class_confidences.extend([d.confidence for d in class_detections])
        
        class_stats["total_detections"] = len(class_confidences)
        class_stats["models_detecting"] = models_with_class
        class_stats["avg_confidence"] = np.mean(class_confidences) if class_confidences else 0
        
        results["class_statistics"][class_id] = class_stats
    
    return results


def compare_strategies(detections: Dict[str, List[Detection]], strategies: List[str], args) -> Dict:
    """Compare different ensemble strategies."""
    if not args.quiet:
        print(f"⚖️  Comparing {len(strategies)} strategies...")
    
    # Set up ensemble
    ensemble = AdvancedEnsemble()
    ensemble.detections = detections
    
    results = {
        "strategy_comparison": {},
        "best_strategy": None,
        "metrics_comparison": {}
    }
    
    strategy_results = {}
    
    for strategy_name in strategies:
        if strategy_name not in ensemble.strategies:
            print(f"⚠️  Strategy '{strategy_name}' not available")
            continue
        
        try:
            strategy_detections = ensemble.run_strategy(strategy_name)
            
            # Calculate metrics
            metrics = {
                "detection_count": len(strategy_detections),
                "avg_confidence": np.mean([d.confidence for d in strategy_detections]) if strategy_detections else 0,
                "confidence_std": np.std([d.confidence for d in strategy_detections]) if strategy_detections else 0,
                "spatial_coverage": calculate_spatial_coverage(strategy_detections)
            }
            
            strategy_results[strategy_name] = strategy_detections
            results["strategy_comparison"][strategy_name] = metrics
            
        except Exception as e:
            print(f"❌ Error running {strategy_name}: {e}")
    
    # Find best strategy (by detection count for now)
    if strategy_results:
        best_strategy = max(results["strategy_comparison"].keys(), 
                          key=lambda x: results["strategy_comparison"][x]["detection_count"])
        results["best_strategy"] = best_strategy
    
    return results


def calculate_spatial_coverage(detections: List[Detection]) -> float:
    """Calculate spatial coverage of detections."""
    if not detections:
        return 0.0
    
    x_coords = [d.x + d.w/2 for d in detections]  # Center coordinates
    y_coords = [d.y + d.h/2 for d in detections]
    
    x_range = max(x_coords) - min(x_coords) if len(set(x_coords)) > 1 else 0
    y_range = max(y_coords) - min(y_coords) if len(set(y_coords)) > 1 else 0
    
    return x_range * y_range


def generate_plots(analysis_results: Dict, args):
    """Generate visualization plots."""
    if args.save_plots:
        plot_dir = Path(args.save_plots)
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Model comparison
    if "model_stats" in analysis_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(analysis_results["model_stats"].keys())
        detection_counts = [analysis_results["model_stats"][m]["total_detections"] for m in models]
        avg_confidences = [analysis_results["model_stats"][m]["avg_confidence"] for m in models]
        
        # Detection counts
        ax1.bar(models, detection_counts)
        ax1.set_title("Detection Counts by Model")
        ax1.set_ylabel("Number of Detections")
        ax1.tick_params(axis='x', rotation=45)
        
        # Average confidences
        ax2.bar(models, avg_confidences)
        ax2.set_title("Average Confidence by Model")
        ax2.set_ylabel("Average Confidence")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(plot_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        if args.show_plots:
            plt.show()
        plt.close()
    
    # Plot 2: Agreement matrix
    if "agreement_matrix" in analysis_results:
        agreement_data = analysis_results["agreement_matrix"]
        if agreement_data:
            # Create matrix for heatmap
            models = set()
            for key in agreement_data.keys():
                m1, m2 = key.split('_', 1)
                models.add(m1)
                models.add(m2)
            
            models = sorted(list(models))
            n_models = len(models)
            
            if n_models > 1:
                matrix = np.zeros((n_models, n_models))
                
                for i, m1 in enumerate(models):
                    for j, m2 in enumerate(models):
                        if i == j:
                            matrix[i, j] = 1.0  # Perfect self-agreement
                        else:
                            key1 = f"{m1}_{m2}"
                            key2 = f"{m2}_{m1}"
                            if key1 in agreement_data:
                                matrix[i, j] = agreement_data[key1]["jaccard"]
                            elif key2 in agreement_data:
                                matrix[i, j] = agreement_data[key2]["jaccard"]
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(matrix, annot=True, xticklabels=models, yticklabels=models,
                           cmap='Blues', vmin=0, vmax=1)
                plt.title("Model Agreement Matrix (Jaccard Index)")
                
                if args.save_plots:
                    plt.savefig(plot_dir / "agreement_matrix.png", dpi=300, bbox_inches='tight')
                if args.show_plots:
                    plt.show()
                plt.close()


def print_summary_report(analysis_results: Dict, args):
    """Print summary report to console."""
    print("\n" + "="*60)
    print("📋 ASSESSMENT SUMMARY REPORT")
    print("="*60)
    
    if "model_stats" in analysis_results:
        print("\n📊 Model Overview:")
        print(f"  Total models analyzed: {analysis_results['total_models']}")
        
        for model_name, stats in analysis_results["model_stats"].items():
            print(f"\n  {model_name}:")
            print(f"    Detections: {stats['total_detections']}")
            print(f"    Avg confidence: {stats['avg_confidence']:.3f} ± {stats['confidence_std']:.3f}")
            if stats['class_distribution']:
                top_classes = sorted(stats['class_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top classes: {dict(top_classes)}")
    
    if "agreement_matrix" in analysis_results:
        print("\n🤝 Model Agreement:")
        agreement_scores = [data["jaccard"] for data in analysis_results["agreement_matrix"].values()]
        if agreement_scores:
            print(f"    Average agreement: {np.mean(agreement_scores):.3f}")
            print(f"    Agreement range: {np.min(agreement_scores):.3f} - {np.max(agreement_scores):.3f}")
    
    if "strategy_comparison" in analysis_results:
        print("\n⚖️  Strategy Comparison:")
        for strategy, metrics in analysis_results["strategy_comparison"].items():
            print(f"    {strategy}: {metrics['detection_count']} detections "
                  f"(conf: {metrics['avg_confidence']:.3f})")


def save_full_report(analysis_results: Dict, output_path: str, args):
    """Save detailed analysis report."""
    if args.report == "json":
        output_file = f"{output_path}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"💾 Full report saved to: {output_file}")
    
    else:
        output_file = f"{output_path}.txt"
        with open(output_file, 'w') as f:
            f.write("OBJECT DETECTION MODEL ASSESSMENT REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Write detailed analysis
            if "model_stats" in analysis_results:
                f.write("MODEL STATISTICS:\n")
                f.write("-"*20 + "\n")
                for model, stats in analysis_results["model_stats"].items():
                    f.write(f"\n{model}:\n")
                    f.write(f"  Total detections: {stats['total_detections']}\n")
                    f.write(f"  Average confidence: {stats['avg_confidence']:.4f}\n")
                    f.write(f"  Confidence std: {stats['confidence_std']:.4f}\n")
                    f.write(f"  Class distribution: {stats['class_distribution']}\n")
            
            if "agreement_matrix" in analysis_results:
                f.write("\nAGREEMENT ANALYSIS:\n")
                f.write("-"*20 + "\n")
                for pair, metrics in analysis_results["agreement_matrix"].items():
                    f.write(f"{pair}: Jaccard={metrics['jaccard']:.4f}, "
                           f"Matches={metrics['matches']}\n")
        
        print(f"💾 Full report saved to: {output_file}")


def run_gt_evaluation(detections: Dict[str, List[Detection]], args) -> Dict:
    """Run ground truth evaluation."""
    if not args.gt:
        return {}
    
    print("🔍 Running ground truth evaluation...")
    
    # Set up evaluator
    gt_dir = args.gt_dir or "labels/GT"
    evaluator = Evaluator(
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        gt_dir=gt_dir
    )
    
    # Validate GT structure
    validation = validate_ground_truth_structure(args.models_dir or "labels")
    if not validation.get('structure_valid', False):
        print("❌ Invalid ground truth structure:")
        print(f"  GT directory exists: {validation.get('gt_dir_exists', False)}")
        print(f"  GT files found: {validation.get('gt_files_found', [])}")
        return {}
    
    results = {}
    
    # Individual model evaluations
    if not args.quiet:
        print(f"📊 Evaluating {len(detections)} models against ground truth...")
    
    for model_name, model_detections in tqdm(detections.items(), desc="Evaluating models", disable=args.quiet):
        try:
            evaluation = evaluator.evaluate_predictions(
                model_detections, 
                args.gt_file,
                include_error_analysis=args.error_analysis
            )
            results[model_name] = evaluation
            
            if not args.quiet:
                metrics = evaluation.get('overall_metrics', {})
                print(f"  {model_name}: mAP={metrics.get('map_50', 0):.3f}, "
                      f"F1={metrics.get('f1_score', 0):.3f}")
        
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    return results


def run_strategy_evaluation(args) -> Dict:
    """Run strategy evaluation against ground truth."""
    if not args.gt or not (args.benchmark_strategies or args.optimize_strategy):
        return {}
    
    print("🎯 Running strategy evaluation...")
    
    # Set up ensemble
    ensemble = AdvancedEnsemble(
        labels_dir=args.models_dir or "labels",
        gt_dir=args.gt_dir
    )
    ensemble.load_detections("detections.txt")
    
    results = {}
    
    if args.benchmark_strategies:
        print("📈 Benchmarking all strategies against ground truth...")
        strategy_evaluations = ensemble.evaluate_all_strategies_with_gt(args.gt_file)
        results['strategy_benchmark'] = strategy_evaluations
        
        if not args.quiet and strategy_evaluations:
            print("\n📊 Strategy Performance Summary:")
            print(f"{'Strategy':<25} {'mAP@0.5':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
            print("-" * 65)
            
            for strategy, eval_result in strategy_evaluations.items():
                metrics = eval_result.get('overall_metrics', {})
                print(f"{strategy:<25} {metrics.get('map_50', 0):<10.3f} "
                      f"{metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<10.3f} "
                      f"{metrics.get('f1_score', 0):<10.3f}")
    
    if args.optimize_strategy:
        print("🔧 Finding optimal strategy...")
        try:
            evaluator = Evaluator(gt_dir=args.gt_dir or "labels/GT")
            optimizer = StrategyOptimizer(evaluator, optimization_metric='map_50')
            
            optimization_result = optimizer.optimize_all_strategies(ensemble, gt_file=args.gt_file)
            results['optimization'] = optimization_result
            
            if not args.quiet:
                print(f"✅ Best strategy: {optimization_result.best_strategy}")
                print(f"   Best score: {optimization_result.best_score:.4f}")
                print(f"   Best params: {optimization_result.best_params}")
        
        except Exception as e:
            print(f"❌ Error in strategy optimization: {e}")
    
    return results


def run_error_analysis(detections: Dict[str, List[Detection]], args) -> Dict:
    """Run detailed error analysis."""
    if not args.gt or not args.error_analysis:
        return {}
    
    print("🔬 Running detailed error analysis...")
    
    # Set up error analyzer
    error_analyzer = ErrorAnalyzer(iou_threshold=args.iou_threshold)
    
    # Load ground truth
    from detection_fusion.utils.io import load_ground_truth
    gt_dir = args.gt_dir or "labels/GT"
    
    try:
        ground_truth = load_ground_truth(gt_dir, args.gt_file)
    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {gt_dir}/{args.gt_file}")
        return {}
    
    results = {}
    
    for model_name, model_detections in detections.items():
        if not args.quiet:
            print(f"  Analyzing errors for {model_name}...")
        
        try:
            error_instances, error_summary = error_analyzer.analyze_errors(
                model_detections, ground_truth
            )
            
            # Additional analyses
            confidence_analysis = error_analyzer.analyze_by_confidence(error_instances)
            spatial_analysis = error_analyzer.analyze_spatial_distribution(error_instances)
            size_analysis = error_analyzer.analyze_by_object_size(error_instances)
            
            results[model_name] = {
                'error_summary': {
                    'false_positives': error_summary.false_positives,
                    'false_negatives': error_summary.false_negatives,
                    'localization_errors': error_summary.localization_errors,
                    'classification_errors': error_summary.classification_errors,
                    'duplicate_detections': error_summary.duplicate_detections,
                    'true_positives': error_summary.true_positives,
                    'error_rate': error_summary.total_errors / error_summary.total_detections if error_summary.total_detections > 0 else 0
                },
                'confidence_analysis': confidence_analysis,
                'spatial_analysis': spatial_analysis,
                'size_analysis': size_analysis
            }
            
            if not args.quiet:
                print(f"    Error rate: {results[model_name]['error_summary']['error_rate']:.3f}")
                print(f"    FP: {error_summary.false_positives}, "
                      f"FN: {error_summary.false_negatives}, "
                      f"TP: {error_summary.true_positives}")
        
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
    
    return results


def print_gt_summary_report(gt_results: Dict, strategy_results: Dict, error_results: Dict, args):
    """Print ground truth evaluation summary report."""
    print("\n" + "="*60)
    print("📋 GROUND TRUTH EVALUATION SUMMARY")
    print("="*60)
    
    # Model performance summary
    if gt_results:
        print("\n📊 Model Performance:")
        print(f"{'Model':<15} {'mAP@0.5':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 55)
        
        for model_name, evaluation in gt_results.items():
            if isinstance(evaluation, dict) and 'overall_metrics' in evaluation:
                metrics = evaluation['overall_metrics']
                print(f"{model_name:<15} {metrics.get('map_50', 0):<10.3f} "
                      f"{metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<10.3f} "
                      f"{metrics.get('f1_score', 0):<10.3f}")
    
    # Strategy optimization results
    if 'optimization' in strategy_results:
        opt_result = strategy_results['optimization']
        print("\n🎯 Strategy Optimization Results:")
        print(f"  Best strategy: {opt_result.best_strategy}")
        print(f"  Best score: {opt_result.best_score:.4f}")
        if opt_result.best_params:
            print(f"  Best parameters: {opt_result.best_params}")
        
    # Error analysis summary
    if error_results:
        print("\n🔬 Error Analysis Summary:")
        for model_name, error_data in error_results.items():
            if 'error_summary' in error_data:
                summary = error_data['error_summary']
                print(f"  {model_name}:")
                print(f"    Error rate: {summary['error_rate']:.3f}")
                print(f"    False positives: {summary['false_positives']}")
                print(f"    False negatives: {summary['false_negatives']}")


def save_gt_evaluation_report(gt_results: Dict, strategy_results: Dict, error_results: Dict, args):
    """Save comprehensive ground truth evaluation report."""
    all_results = {
        'model_evaluations': gt_results,
        'strategy_evaluations': strategy_results,
        'error_analysis': error_results,
        'evaluation_config': {
            'gt_dir': args.gt_dir or "labels/GT",
            'gt_file': args.gt_file,
            'iou_threshold': args.iou_threshold,
            'confidence_threshold': args.confidence_threshold
        }
    }
    
    # Save in requested format
    if args.report == 'json':
        output_file = f"{args.output}_gt.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"💾 GT evaluation results saved to: {output_file}")
    
    elif args.report in ['full', 'summary']:
        output_file = f"{args.output}_gt.txt"
        from detection_fusion.utils.io import save_evaluation_results
        save_evaluation_results(all_results, output_file, 'txt')
        print(f"💾 GT evaluation report saved to: {output_file}")


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.save_plots:
        args.plot = True  # Enable plotting if save directory specified
    
    # Setup analyzer
    try:
        analyzer = setup_analyzer(args)
    except Exception as e:
        print(f"❌ Error setting up analyzer: {e}")
        sys.exit(1)
    
    # Load detections
    try:
        detections = load_and_filter_detections(analyzer, args)
        if not detections:
            print("❌ No detections loaded")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading detections: {e}")
        sys.exit(1)
    
    # Run analysis
    analysis_results = {}
    
    if args.analyze in ["agreement", "all"]:
        analysis_results.update(analyze_model_agreement(detections, args))
    
    if args.analyze in ["confidence", "all"]:
        analysis_results.update(analyze_confidence_distribution(detections, args))
    
    if args.analyze in ["class-wise", "all"]:
        analysis_results.update(analyze_class_wise_performance(detections, args))
    
    if args.compare == "strategies" and args.strategies:
        analysis_results.update(compare_strategies(detections, args.strategies, args))
    
    # Ground Truth Evaluation
    gt_results = {}
    strategy_results = {}
    error_results = {}
    
    if args.gt:
        # Run ground truth evaluation
        if args.analyze in ["evaluation", "all"] or args.gt:
            gt_results = run_gt_evaluation(detections, args)
        
        # Run strategy evaluation
        strategy_results = run_strategy_evaluation(args)
        
        # Run error analysis
        if args.error_analysis or args.analyze in ["error-analysis", "all"]:
            error_results = run_error_analysis(detections, args)
    
    # Generate outputs
    if args.plot:
        generate_plots(analysis_results, args)
    
    if args.report in ["summary", "full"]:
        print_summary_report(analysis_results, args)
        
        # Print GT summary if GT evaluation was run
        if args.gt and (gt_results or strategy_results or error_results):
            print_gt_summary_report(gt_results, strategy_results, error_results, args)
    
    if args.report in ["full", "json"]:
        save_full_report(analysis_results, args.output, args)
        
        # Save GT evaluation report if GT evaluation was run
        if args.gt and (gt_results or strategy_results or error_results):
            save_gt_evaluation_report(gt_results, strategy_results, error_results, args)
    
    if not args.quiet:
        completion_msg = "✅ Assessment complete!"
        if args.gt:
            completion_msg += " (with ground truth evaluation)"
        print(f"\n{completion_msg}")


if __name__ == "__main__":
    main()