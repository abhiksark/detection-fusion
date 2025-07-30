#!/usr/bin/env python3
"""
Basic usage example for DetectionFusion package.
"""

from detection_fusion import EnsembleVoting, MultiModelAnalyzer

def main():
    # Example 1: Basic ensemble voting
    print("=== Basic Ensemble Voting ===")
    
    ensemble = EnsembleVoting(labels_dir="labels", output_dir="labels/unified")
    
    # Load detections from all models
    ensemble.load_detections("detections.txt")
    
    # Run specific strategy
    majority_results = ensemble.run_strategy("majority_vote_2")
    print(f"Majority voting (2+ models): {len(majority_results)} detections")
    
    # Run all strategies
    all_results = ensemble.run_all_strategies("detections.txt")
    
    # Save statistics
    ensemble.save_statistics(all_results)
    
    
    # Example 2: Model analysis
    print("\n=== Model Analysis ===")
    
    analyzer = MultiModelAnalyzer(labels_dir="labels", iou_threshold=0.5)
    
    # Load detections
    analyzer.load_detections("detections.txt")
    
    # Get class statistics
    class_stats = analyzer.get_class_statistics()
    print("\nTop 5 most detected classes:")
    print(class_stats[['class_name', 'total', 'mean', 'std']].head())
    
    # Get confidence statistics
    conf_stats = analyzer.get_confidence_statistics()
    print("\nConfidence statistics per model:")
    print(conf_stats[['model', 'mean', 'std', 'min', 'max']])
    
    # Compare specific models
    if len(analyzer.models) >= 2:
        comparison = analyzer.compare_models(analyzer.models[0], analyzer.models[1])
        print(f"\nComparison {analyzer.models[0]} vs {analyzer.models[1]}:")
        print(f"  Matches: {comparison['total_matches']}")
        print(f"  {analyzer.models[0]} unique: {comparison['model1_unique']}")
        print(f"  {analyzer.models[1]} unique: {comparison['model2_unique']}")
        print(f"  Average IoU: {comparison['avg_iou']:.3f}")
    
    # Generate full report
    analyzer.generate_report("analysis_report.txt")
    print("\nFull report saved to: analysis_report.txt")


if __name__ == "__main__":
    main()