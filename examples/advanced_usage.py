#!/usr/bin/env python3
"""
Advanced usage example for DetectionFusion package.
"""

from detection_fusion import AdvancedEnsemble, MultiModelAnalyzer
from detection_fusion.strategies import MajorityVoting
from detection_fusion.visualization import generate_all_plots

def main():
    # Example 1: Advanced ensemble with custom parameters
    print("=== Advanced Ensemble ===")
    
    ensemble = AdvancedEnsemble(labels_dir="labels", output_dir="labels/unified")
    
    # Load detections
    ensemble.load_detections("detections.txt")
    
    # Customize strategy parameters
    ensemble.set_strategy_params("dbscan", eps=0.15, min_samples=3)
    ensemble.set_strategy_params("soft_voting", temperature=0.5)
    
    # Run specific advanced strategies
    dbscan_results = ensemble.run_strategy("dbscan")
    print(f"DBSCAN clustering: {len(dbscan_results)} detections")
    
    soft_results = ensemble.run_strategy("soft_voting")
    print(f"Soft voting: {len(soft_results)} detections")
    
    
    # Example 2: Custom strategy
    print("\n=== Custom Strategy ===")
    
    # Create custom majority voting requiring 3+ models
    custom_voter = MajorityVoting(iou_threshold=0.6, min_votes=3)
    ensemble.add_strategy("strict_majority", custom_voter)
    
    # Run custom strategy
    strict_results = ensemble.run_strategy("strict_majority")
    print(f"Strict majority (3+ models): {len(strict_results)} detections")
    
    
    # Example 3: Batch processing multiple files
    print("\n=== Batch Processing ===")
    
    files_to_process = ["detections.txt", "predictions.txt", "results.txt"]
    
    for filename in files_to_process:
        try:
            print(f"\nProcessing {filename}...")
            results = ensemble.run_all_strategies(filename)
            
            # Print summary
            for strategy, detections in results.items():
                print(f"  {strategy}: {len(detections)} detections")
                
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    
    # Example 4: Analysis with visualizations
    print("\n=== Analysis with Visualizations ===")
    
    analyzer = MultiModelAnalyzer(labels_dir="labels", iou_threshold=0.5)
    analyzer.load_detections("detections.txt")
    
    # Find consensus detections (agreed by 2+ models)
    consensus = analyzer.find_consensus_detections(min_models=2)
    print(f"Consensus detections (2+ models): {len(consensus)}")
    
    # Generate all plots
    generate_all_plots(analyzer, output_dir="analysis_plots", top_n=20)
    print("Plots saved to: analysis_plots/")
    
    
    # Example 5: Export results in different formats
    print("\n=== Export Results ===")
    
    # Get results
    results = ensemble.run_strategy("weighted_vote")
    
    # Export as COCO format (example)
    coco_output = {
        "images": [{"id": 1, "file_name": "image.jpg"}],
        "annotations": [],
        "categories": []
    }
    
    for i, det in enumerate(results):
        coco_output["annotations"].append({
            "id": i,
            "image_id": 1,
            "category_id": det.class_id,
            "bbox": [det.x - det.w/2, det.y - det.h/2, det.w, det.h],
            "score": det.confidence,
            "area": det.w * det.h
        })
    
    import json
    with open("labels/unified/results_coco.json", "w") as f:
        json.dump(coco_output, f, indent=2)
    
    print("Results exported to COCO format")


if __name__ == "__main__":
    main()