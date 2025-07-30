#!/usr/bin/env python3
"""
DetectionFusion Merging Tool

A command-line interface for fusing predictions from multiple object detection 
models using various ensemble strategies with ground truth optimization. 
This tool provides an intuitive way to combine model outputs and improve 
detection quality through consensus-based learning.

Usage:
    python merge.py --models model1 model2 model3 --strategy weighted_vote --output results.txt
    python merge.py --config configs/high_precision_config.yaml
    python merge.py --models-dir labels/ --strategy multi_scale --iou 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from detection_fusion import AdvancedEnsemble
from detection_fusion.utils import load_yaml_config, save_detections, validate_ground_truth_structure, read_detections
from detection_fusion.core.detection import Detection


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge object detection predictions using ensemble strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merging with 3 models
  python merge.py --models model1 model2 model3 --strategy weighted_vote

  # Use configuration file
  python merge.py --config configs/high_precision_config.yaml

  # Merge all models in directory
  python merge.py --models-dir labels/ --strategy multi_scale --output ensemble_results.txt

  # Run multiple strategies
  python merge.py --models-dir labels/ --strategies weighted_vote affirmative_nms bayesian

  # Custom parameters
  python merge.py --models model1 model2 --strategy majority_vote_2 --iou 0.7 --confidence 0.3
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--models", 
        nargs="+", 
        help="List of model names (should match directories in labels/)"
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
    
    # Strategy options
    parser.add_argument(
        "--strategy", 
        type=str,
        default="weighted_vote",
        help="Ensemble strategy to use (default: weighted_vote)"
    )
    parser.add_argument(
        "--strategies", 
        nargs="+",
        help="Multiple strategies to run and compare"
    )
    
    # Parameters
    parser.add_argument(
        "--iou", 
        type=float,
        default=0.5,
        help="IoU threshold for matching detections (default: 0.5)"
    )
    parser.add_argument(
        "--confidence", 
        type=float,
        default=0.1,
        help="Minimum confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "--min-votes", 
        type=int,
        default=2,
        help="Minimum votes required (for voting strategies, default: 2)"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str,
        default="ensemble_results.txt",
        help="Output file path (default: ensemble_results.txt)"
    )
    parser.add_argument(
        "--format", 
        choices=["yolo", "json"],
        default="yolo",
        help="Output format (default: yolo)"
    )
    parser.add_argument(
        "--save-individual", 
        action="store_true",
        help="Save results from each strategy separately"
    )
    
    # Utility options
    parser.add_argument(
        "--list-strategies", 
        action="store_true",
        help="List all available strategies and exit"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Quiet mode (minimal output)"
    )
    parser.add_argument(
        "--image-mode",
        action="store_true",
        default=True,
        help="Compare detections image by image (default behavior)"
    )
    parser.add_argument(
        "--single-file-mode",
        action="store_true",
        help="Use single detection file per model (legacy behavior)"
    )
    
    return parser.parse_args()


def list_available_strategies():
    """List all available ensemble strategies."""
    strategies = {
        "Basic Voting": [
            "majority_vote_2", "majority_vote_3", "majority_vote_all",
            "weighted_vote", "unanimous"
        ],
        "NMS-Based": [
            "nms", "affirmative_nms", "confidence_weighted_nms"
        ],
        "Clustering": [
            "dbscan", "centroid_clustering", "distance_weighted"
        ],
        "Probabilistic": [
            "soft_voting", "bayesian"
        ],
        "Confidence-Based": [
            "confidence_threshold", "high_confidence_first"
        ],
        "Adaptive": [
            "adaptive_threshold", "density_adaptive", 
            "multi_scale", "consensus_ranking"
        ]
    }
    
    print("📋 Available Ensemble Strategies:")
    print("=" * 50)
    
    for category, strategy_list in strategies.items():
        print(f"\n{category}:")
        for strategy in strategy_list:
            print(f"  • {strategy}")
    
    print(f"\nTotal: {sum(len(s) for s in strategies.values())} strategies available")
    print("\nFor detailed strategy information, see docs/STRATEGY_GUIDE.md")


def setup_ensemble(args) -> AdvancedEnsemble:
    """Set up the ensemble based on arguments."""
    if args.config:
        # Load from configuration file
        config = load_yaml_config(args.config)
        ensemble_config = config.get("ensemble", {})
        
        ensemble = AdvancedEnsemble(
            labels_dir=ensemble_config.get("labels_dir", "labels"),
            output_dir=ensemble_config.get("output_dir", "output")
        )
        
        # Apply strategy configurations
        strategies_config = ensemble_config.get("strategies", {})
        for strategy_name, params in strategies_config.items():
            if strategy_name in ensemble.strategies:
                ensemble.set_strategy_params(strategy_name, **params)
        
        return ensemble
    
    else:
        # Set up from command line arguments
        if args.models_dir:
            labels_dir = args.models_dir
        else:
            labels_dir = "labels"
        
        gt_dir = args.gt_dir if hasattr(args, 'gt_dir') and args.gt_dir else None
        ensemble = AdvancedEnsemble(labels_dir=labels_dir, gt_dir=gt_dir)
        
        # Set common parameters
        common_params = {
            "iou_threshold": args.iou,
            "min_votes": args.min_votes
        }
        
        # Apply to relevant strategies
        strategy_name = args.strategy
        if strategy_name in ensemble.strategies:
            ensemble.set_strategy_params(strategy_name, **common_params)
        
        return ensemble


def load_model_detections(ensemble: AdvancedEnsemble, args) -> Dict[str, List[Detection]]:
    """Load detections from specified models."""
    # Load all detections first - use image mode by default
    if hasattr(args, 'single_file_mode') and args.single_file_mode:
        all_detections = ensemble.load_detections()
    else:
        # Default to image mode
        all_detections = ensemble.load_all_image_detections()
    
    if args.models:
        # Filter to specific models
        detections = {}
        for model_name in args.models:
            if model_name in all_detections:
                model_detections = all_detections[model_name]
                detections[model_name] = model_detections
                if not args.quiet:
                    print(f"✓ Loaded {len(model_detections)} detections from {model_name}")
            else:
                print(f"⚠️  Model '{model_name}' not found in labels directory")
                continue
        
        return detections
    
    else:
        # Use all loaded models
        if not args.quiet:
            total_detections = sum(len(dets) for dets in all_detections.values())
            print(f"✓ Loaded {total_detections} total detections from {len(all_detections)} models")
        
        return all_detections


def run_single_strategy(ensemble: AdvancedEnsemble, strategy_name: str, args) -> List[Detection]:
    """Run a single ensemble strategy."""
    if strategy_name not in ensemble.strategies:
        available = list(ensemble.strategies.keys())
        print(f"❌ Strategy '{strategy_name}' not found.")
        print(f"Available strategies: {', '.join(available)}")
        return []
    
    if args.verbose:
        print(f"🔄 Running strategy: {strategy_name}")
    
    try:
        # Check if we should run per-image (when outputting to unified directory)
        output_path = Path(args.output)
        if output_path.name == "unified" or (not args.single_file_mode and output_path.suffix == ""):
            # Run per-image strategy for unified output
            return run_strategy_per_image(ensemble, strategy_name, args)
        else:
            # Run traditional single-file strategy
            results = ensemble.run_strategy(strategy_name)
            
            if not args.quiet:
                avg_conf = sum(d.confidence for d in results) / len(results) if results else 0
                print(f"✓ {strategy_name}: {len(results)} detections (avg conf: {avg_conf:.3f})")
            
            return results
    
    except Exception as e:
        print(f"❌ Error running {strategy_name}: {e}")
        return []


def run_strategy_per_image(ensemble: AdvancedEnsemble, strategy_name: str, args) -> List[Detection]:
    """Run ensemble strategy on each image separately."""
    # Get the pre-loaded image detections structure
    from collections import defaultdict
    image_detections = defaultdict(dict)
    
    # We need to reconstruct the image detections from the loaded detections
    # This is a bit hacky but necessary given the current structure
    for model_dir in ensemble.labels_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ["unified", "__pycache__", "GT"]:
            model_name = model_dir.name
            
            txt_files = list(model_dir.glob("*.txt"))
            for txt_file in tqdm(txt_files, desc=f"  Loading {model_name}", leave=False):
                image_name = txt_file.stem
                detections = read_detections(str(txt_file), model_name, 1.0, image_name)
                
                # Filter to only the models we're using
                if hasattr(args, 'models') and args.models and model_name not in args.models:
                    continue
                    
                image_detections[image_name][model_name] = detections
    
    # Run strategy per image
    image_results = ensemble.run_strategy_per_image(strategy_name, dict(image_detections))
    
    # Flatten results for compatibility but preserve image_name
    all_results = []
    for image_name, detections in image_results.items():
        all_results.extend(detections)
    
    if not args.quiet:
        avg_conf = sum(d.confidence for d in all_results) / len(all_results) if all_results else 0
        print(f"✓ {strategy_name}: {len(all_results)} detections across {len(image_results)} images (avg conf: {avg_conf:.3f})")
    
    return all_results


def find_best_strategy_with_gt(ensemble: AdvancedEnsemble, args) -> Optional[str]:
    """Find the best strategy using ground truth evaluation."""
    if not hasattr(args, 'gt') or not args.gt and not (hasattr(args, 'auto_strategy') and args.auto_strategy):
        return None
    
    # Validate GT structure
    validation = validate_ground_truth_structure(args.models_dir or "labels")
    
    if not validation.get('structure_valid', False):
        print("❌ Invalid ground truth structure:")
        print(f"  GT directory exists: {validation.get('gt_dir_exists', False)}")
        print(f"  GT files found: {validation.get('gt_files_found', [])}")
        return None
    
    if not args.quiet:
        print("🎯 Finding optimal strategy using ground truth evaluation...")
    
    try:
        # Use ensemble's built-in GT evaluation
        best_result = ensemble.find_best_strategy_with_gt(
            gt_file=args.gt_file, 
            metric=args.evaluation_metric
        )
        
        if best_result:
            best_strategy, best_evaluation = best_result
            
            if not args.quiet:
                metrics = best_evaluation.get('overall_metrics', {})
                print(f"✅ Best strategy found: {best_strategy}")
                print(f"   Performance: {args.evaluation_metric}={metrics.get(args.evaluation_metric, 0):.4f}")
                print(f"   Precision: {metrics.get('precision', 0):.3f}, "
                      f"Recall: {metrics.get('recall', 0):.3f}, "
                      f"F1: {metrics.get('f1_score', 0):.3f}")
            
            return best_strategy
        else:
            print("❌ Could not determine best strategy from ground truth evaluation")
            return None
    
    except Exception as e:
        print(f"❌ Error in GT-based strategy optimization: {e}")
        return None


def run_gt_strategy_comparison(ensemble: AdvancedEnsemble, strategies: List[str], args) -> Dict[str, Dict]:
    """Compare multiple strategies using ground truth evaluation."""
    if not args.gt:
        return {}
    
    print(f"📊 Comparing {len(strategies)} strategies using ground truth...")
    
    strategy_evaluations = {}
    
    for strategy_name in tqdm(strategies, desc="Comparing strategies", disable=not args.verbose):
        if strategy_name not in ensemble.strategies:
            continue
        
        try:
            evaluation = ensemble.evaluate_strategy_with_gt(strategy_name, args.gt_file)
            if evaluation:
                strategy_evaluations[strategy_name] = evaluation
                
                if args.verbose:
                    metrics = evaluation.get('overall_metrics', {})
                    print(f"  {strategy_name}: {args.evaluation_metric}={metrics.get(args.evaluation_metric, 0):.4f}")
        
        except Exception as e:
            print(f"❌ Error evaluating {strategy_name}: {e}")
            continue
    
    return strategy_evaluations


def print_gt_strategy_comparison(evaluations: Dict[str, Dict], metric: str):
    """Print GT-based strategy comparison results."""
    if not evaluations:
        return
    
    print("\n📈 Ground Truth Strategy Performance Comparison:")
    print(f"{'Strategy':<25} {metric.upper():<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 65)
    
    # Sort by the selected metric
    sorted_strategies = sorted(
        evaluations.items(), 
        key=lambda x: x[1].get('overall_metrics', {}).get(metric, 0), 
        reverse=True
    )
    
    for strategy_name, evaluation in sorted_strategies:
        metrics = evaluation.get('overall_metrics', {})
        print(f"{strategy_name:<25} {metrics.get(metric, 0):<10.3f} "
              f"{metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<10.3f} "
              f"{metrics.get('f1_score', 0):<10.3f}")
    
    best_strategy = sorted_strategies[0][0] if sorted_strategies else None
    if best_strategy:
        print(f"\n🏆 Best performing strategy: {best_strategy}")


def save_results(results: List[Detection], output_path: str, format_type: str, args):
    """Save ensemble results to file."""
    try:
        # If output path is a directory or ends with /, save per-image
        output_path_obj = Path(output_path)
        
        # Check if we should save in unified directory format
        if output_path_obj.name == "unified" or (not args.single_file_mode and output_path_obj.suffix == ""):
            # Save per-image in unified directory format
            save_unified_results(results, output_path_obj, format_type, args)
        else:
            # Save to single file
            if format_type == "json":
                import json
                results_dict = [det.to_dict() for det in results]
                with open(output_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
            else:
                # YOLO format (default)
                save_detections(results, output_path)
            
            if not args.quiet:
                print(f"💾 Results saved to: {output_path}")
    
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def save_unified_results(results: List[Detection], output_dir: Path, format_type: str, args):
    """Save results in unified directory format (per-image files)."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group detections by image
    image_detections = {}
    for det in results:
        # Extract image name from detection (assuming it's stored during loading)
        # For now, we'll use a placeholder - this needs to be enhanced
        image_name = "detections"  # Default name if no image info
        if hasattr(det, 'image_name'):
            image_name = det.image_name
        
        if image_name not in image_detections:
            image_detections[image_name] = []
        image_detections[image_name].append(det)
    
    # Save each image's detections
    total_files = 0
    for image_name, detections in image_detections.items():
        if format_type == "json":
            output_file = output_dir / f"{image_name}.json"
            import json
            results_dict = [det.to_dict() for det in detections]
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
        else:
            # YOLO format
            output_file = output_dir / f"{image_name}.txt"
            save_detections(detections, str(output_file))
        total_files += 1
    
    if not args.quiet:
        print(f"💾 Results saved to {total_files} files in: {output_dir}/")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Handle special modes
    if args.list_strategies:
        list_available_strategies()
        return
    
    # Setup ensemble
    try:
        ensemble = setup_ensemble(args)
    except Exception as e:
        print(f"❌ Error setting up ensemble: {e}")
        sys.exit(1)
    
    # Load detections
    try:
        detections = load_model_detections(ensemble, args)
        if not detections:
            print("❌ No detections loaded. Check your input paths.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading detections: {e}")
        sys.exit(1)
    
    # Handle ground truth guided strategy selection
    selected_strategy = None
    
    if hasattr(args, 'auto_strategy') and args.auto_strategy or (hasattr(args, 'gt') and hasattr(args, 'optimize_strategy') and args.gt and args.optimize_strategy):
        selected_strategy = find_best_strategy_with_gt(ensemble, args)
        if selected_strategy:
            if not args.quiet:
                print(f"🤖 Auto-selected strategy: {selected_strategy}")
        else:
            print("⚠️  Could not auto-select strategy, falling back to default")
            selected_strategy = args.strategy
    
    # Run strategies
    if args.strategies:
        # Multiple strategies
        if not args.quiet:
            print(f"🎯 Running {len(args.strategies)} strategies...")
        
        all_results = {}
        for strategy_name in tqdm(args.strategies, desc="Running strategies", disable=args.quiet):
            results = run_single_strategy(ensemble, strategy_name, args)
            if results:
                all_results[strategy_name] = results
                
                if args.save_individual:
                    output_name = f"{Path(args.output).stem}_{strategy_name}{Path(args.output).suffix}"
                    save_results(results, output_name, args.format, args)
        
        # GT-based strategy comparison if enabled
        if args.gt and all_results:
            gt_evaluations = run_gt_strategy_comparison(ensemble, list(all_results.keys()), args)
            if gt_evaluations:
                print_gt_strategy_comparison(gt_evaluations, args.evaluation_metric)
                
                # Use GT-based best strategy for output
                best_gt_strategy = max(gt_evaluations.keys(), 
                                     key=lambda x: gt_evaluations[x].get('overall_metrics', {}).get(args.evaluation_metric, 0))
                save_results(all_results[best_gt_strategy], args.output, args.format, args)
                if not args.quiet:
                    print(f"💫 GT-optimized strategy ({best_gt_strategy}) saved as main output")
            else:
                # Fallback to detection count based selection
                best_strategy = max(all_results.keys(), 
                                  key=lambda x: len(all_results[x]) if all_results[x] else 0)
                save_results(all_results[best_strategy], args.output, args.format, args)
                if not args.quiet:
                    print(f"💫 Best strategy ({best_strategy}) saved as main output")
        else:
            # Summary without GT
            if not args.quiet and all_results:
                print("\n📊 Strategy Comparison:")
                for strategy, results in all_results.items():
                    avg_conf = sum(d.confidence for d in results) / len(results) if results else 0
                    print(f"  {strategy:<20}: {len(results):>3} detections (conf: {avg_conf:.3f})")
            
            # Save best strategy results as main output
            if all_results:
                best_strategy = max(all_results.keys(), 
                                  key=lambda x: len(all_results[x]) if all_results[x] else 0)
                save_results(all_results[best_strategy], args.output, args.format, args)
                if not args.quiet:
                    print(f"💫 Best strategy ({best_strategy}) saved as main output")
    
    else:
        # Single strategy (or auto-selected strategy)
        final_strategy = selected_strategy or args.strategy
        
        if not args.quiet:
            strategy_label = "auto-selected" if selected_strategy else "specified"
            print(f"🎯 Running {strategy_label} strategy: {final_strategy}")
        
        results = run_single_strategy(ensemble, final_strategy, args)
        if results:
            save_results(results, args.output, args.format, args)
            
            # If GT evaluation was used, show performance metrics
            if hasattr(args, 'gt') and args.gt and selected_strategy:
                try:
                    evaluation = ensemble.evaluate_strategy_with_gt(final_strategy, args.gt_file)
                    if evaluation and not args.quiet:
                        metrics = evaluation.get('overall_metrics', {})
                        print("\n📊 Strategy Performance vs Ground Truth:")
                        print(f"  mAP@0.5: {metrics.get('map_50', 0):.3f}")
                        print(f"  Precision: {metrics.get('precision', 0):.3f}")
                        print(f"  Recall: {metrics.get('recall', 0):.3f}")
                        print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
                except Exception as e:
                    if args.verbose:
                        print(f"Note: Could not evaluate final results: {e}")
        else:
            print("❌ No results generated")
            sys.exit(1)
    
    if not args.quiet:
        completion_msg = "✅ Ensemble merging complete!"
        if args.gt:
            completion_msg += " (with ground truth optimization)"
        print(completion_msg)


if __name__ == "__main__":
    main()