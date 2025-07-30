#!/usr/bin/env python3
"""
GT Rectification System Example

This example demonstrates how to use the DetectionFusion GT rectification system
to identify potential ground truth annotation errors and create organized datasets
for human review.

The rectification system compares ensemble consensus across all 17+ strategies
with ground truth labels to identify disagreements that may indicate GT errors.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gt_rectify import GTRectifier
from detection_fusion.core.detection import Detection
from detection_fusion.utils.io import save_detections
import numpy as np
import json


def create_sample_data():
    """Create sample data for demonstration."""
    
    print("Creating sample data structure...")
    
    # Create directory structure
    base_dir = Path("sample_rectification_data")
    labels_dir = base_dir / "labels"
    gt_dir = base_dir / "gt"
    images_dir = base_dir / "images"
    
    # Create model directories
    for model_name in ["yolov8n", "yolov8s", "yolov8m"]:
        model_dir = labels_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
    
    gt_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample image names
    image_names = [f"image_{i:03d}" for i in range(1, 11)]
    
    # Create sample detection data with intentional GT errors
    np.random.seed(42)  # For reproducible results
    
    for img_idx, image_name in enumerate(image_names):
        
        # Generate base detections (simulate real objects)
        num_objects = np.random.randint(2, 6)
        base_detections = []
        
        for obj_idx in range(num_objects):
            # Random object properties
            class_id = np.random.randint(0, 5)
            x = np.random.uniform(0.2, 0.8)  
            y = np.random.uniform(0.2, 0.8)
            w = np.random.uniform(0.1, 0.3)
            h = np.random.uniform(0.1, 0.3)
            base_confidence = np.random.uniform(0.6, 0.95)
            
            base_detections.append({
                'class_id': class_id,
                'x': x, 'y': y, 'w': w, 'h': h,
                'confidence': base_confidence
            })
        
        # Generate model predictions (with noise)
        for model_name in ["yolov8n", "yolov8s", "yolov8m"]:
            model_detections = []
            
            for base_det in base_detections:
                # Add model-specific noise
                if np.random.random() > 0.1:  # 90% detection rate
                    noise_scale = 0.05 if model_name == "yolov8m" else 0.1
                    
                    detection = Detection(
                        base_det['class_id'],
                        base_det['x'] + np.random.normal(0, noise_scale),
                        base_det['y'] + np.random.normal(0, noise_scale),
                        base_det['w'] + np.random.normal(0, noise_scale * 0.5),
                        base_det['h'] + np.random.normal(0, noise_scale * 0.5),
                        base_det['confidence'] + np.random.normal(0, 0.05),
                        model_name
                    )
                    
                    # Clamp values to valid ranges
                    detection.x = np.clip(detection.x, 0, 1)
                    detection.y = np.clip(detection.y, 0, 1)
                    detection.w = np.clip(detection.w, 0.01, 1)
                    detection.h = np.clip(detection.h, 0.01, 1)
                    detection.confidence = np.clip(detection.confidence, 0, 1)
                    
                    model_detections.append(detection)
            
            # Add some false positives
            if np.random.random() > 0.7:
                fp_detection = Detection(
                    np.random.randint(0, 5),
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.05, 0.2),
                    np.random.uniform(0.3, 0.7),
                    model_name
                )
                model_detections.append(fp_detection)
            
            # Save model predictions
            model_file = labels_dir / model_name / "detections.txt"
            save_detections(model_detections, str(model_file))
        
        # Generate ground truth with intentional errors
        gt_detections = []
        
        for base_det in base_detections:
            # 95% chance to include correct detection
            if np.random.random() > 0.05:
                gt_detection = Detection(
                    base_det['class_id'],
                    base_det['x'],
                    base_det['y'], 
                    base_det['w'],
                    base_det['h'],
                    1.0,  # GT doesn't have confidence scores
                    "ground_truth"
                )
                gt_detections.append(gt_detection)
        
        # Add some GT errors for demonstration
        error_type = np.random.choice(['missing', 'extra', 'none'], p=[0.1, 0.1, 0.8])
        
        if error_type == 'missing':
            # Remove a detection (simulate missing annotation)
            if gt_detections:
                gt_detections.pop(np.random.randint(0, len(gt_detections)))
        elif error_type == 'extra':
            # Add false positive to GT (simulate incorrect annotation)
            extra_detection = Detection(
                np.random.randint(0, 5),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.05, 0.2),
                np.random.uniform(0.05, 0.2),
                1.0,
                "ground_truth"
            )
            gt_detections.append(extra_detection)
        
        # Save GT detections
        gt_file = gt_dir / "detections.txt"
        if img_idx == 0:
            # Create new file for first image
            save_detections(gt_detections, str(gt_file))
        else:
            # Append to existing file (batch GT format)
            with open(gt_file, 'a') as f:
                for det in gt_detections:
                    f.write(f"{det.class_id} {det.x:.6f} {det.y:.6f} {det.w:.6f} {det.h:.6f} {det.confidence:.6f}\n")
        
        # Create dummy image files
        dummy_image = images_dir / f"{image_name}.jpg"
        dummy_image.touch()
    
    print(f"Sample data created in {base_dir}/")
    return str(base_dir)


def run_basic_rectification():
    """Demonstrate basic rectification functionality."""
    
    print("\n" + "="*60)
    print("BASIC RECTIFICATION EXAMPLE")
    print("="*60)
    
    # Create sample data
    base_dir = create_sample_data()
    
    labels_dir = f"{base_dir}/labels"
    gt_dir = f"{base_dir}/gt"
    images_dir = f"{base_dir}/images"
    output_dir = f"{base_dir}/rectification_results"
    
    # Initialize rectifier
    print("\n1. Initializing GT Rectifier...")
    rectifier = GTRectifier(
        labels_dir=labels_dir,
        gt_dir=gt_dir,
        images_dir=images_dir,
        iou_threshold=0.5,
        confidence_threshold=0.3,
        min_strategy_agreement=2  # Lower threshold for demo
    )
    
    # Run analysis
    print("\n2. Running comprehensive analysis...")
    results = rectifier.run_full_analysis("detections.txt")
    
    # Display results
    print("\n3. Analysis Results:")
    print(f"   Total images: {results['total_images']}")
    print(f"   Total errors found: {results['total_errors_found']}")
    print(f"   Error types: {dict(results['error_types'])}")
    
    # Show most problematic images
    print("\n   Most problematic images:")
    for img_name, score in results['most_problematic_images']:
        print(f"     {img_name}: {score:.3f} correctness")
    
    # Create rectified dataset
    print("\n4. Creating rectified dataset...")
    rectifier.create_rectified_dataset(
        output_dir=output_dir,
        include_most_correct=5,
        include_most_incorrect=5
    )
    
    print("\n✅ Basic rectification complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return output_dir


def run_advanced_rectification():
    """Demonstrate advanced rectification with maximize_error mode."""
    
    print("\n" + "="*60)
    print("ADVANCED RECTIFICATION EXAMPLE (MAXIMIZE ERROR MODE)")
    print("="*60)
    
    base_dir = "sample_rectification_data"  # Reuse data from basic example
    
    labels_dir = f"{base_dir}/labels"
    gt_dir = f"{base_dir}/gt"
    images_dir = f"{base_dir}/images"
    output_dir = f"{base_dir}/advanced_rectification_results"
    
    # Initialize with maximize_error mode
    print("\n1. Initializing with maximize_error mode...")
    rectifier = GTRectifier(
        labels_dir=labels_dir,
        gt_dir=gt_dir,
        images_dir=images_dir,
        iou_threshold=0.5,           
        confidence_threshold=0.3,    
        min_strategy_agreement=2,    # Lower threshold for more sensitivity
        mode="maximize_error"        # Aggressive error detection
    )
    
    # Run analysis
    print("\n2. Running analysis with stricter parameters...")
    results = rectifier.run_full_analysis("detections.txt")
    
    # Custom analysis of results
    print("\n3. Advanced Analysis:")
    
    # Analyze error patterns
    error_confidences = []
    for error_dict in results['detailed_errors']:
        error_confidences.append(error_dict['confidence_score'])
    
    if error_confidences:
        print(f"   Average error confidence: {np.mean(error_confidences):.3f}")
        print(f"   High-confidence errors (>0.7): {sum(1 for c in error_confidences if c > 0.7)}")
    
    # Show strategy effectiveness  
    print("\n   Strategy Analysis:")
    strategy_votes = {}
    for error_dict in results['detailed_errors']:
        for strategy in error_dict.get('supporting_models', []):
            strategy_votes[strategy] = strategy_votes.get(strategy, 0) + 1
    
    if strategy_votes:
        print("   Most active strategies in error detection:")
        for strategy, count in sorted(strategy_votes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {strategy}: {count} error detections")
    
    # Create focused dataset
    print("\n4. Creating focused rectification dataset...")
    rectifier.create_rectified_dataset(
        output_dir=output_dir,
        include_most_correct=3,      # Fewer images for focused review
        include_most_incorrect=8     # More problematic images
    )
    
    print("\n✅ Advanced rectification complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return output_dir


def run_mode_comparison():
    """Compare minimize_error vs maximize_error modes side by side."""
    
    print("\n" + "="*60)
    print("MODE COMPARISON EXAMPLE")
    print("="*60)
    
    base_dir = "sample_rectification_data"
    labels_dir = f"{base_dir}/labels"
    gt_dir = f"{base_dir}/gt"
    images_dir = f"{base_dir}/images"
    
    print("\n1. Running both modes for comparison...")
    
    # Conservative mode
    print("\n   Conservative Mode (minimize_error):")
    conservative_rectifier = GTRectifier(
        labels_dir=labels_dir,
        gt_dir=gt_dir,
        images_dir=images_dir,
        mode="minimize_error"
    )
    conservative_results = conservative_rectifier.run_full_analysis("detections.txt")
    
    # Aggressive mode
    print("\n   Aggressive Mode (maximize_error):")
    aggressive_rectifier = GTRectifier(
        labels_dir=labels_dir,
        gt_dir=gt_dir,
        images_dir=images_dir,
        mode="maximize_error"
    )
    aggressive_results = aggressive_rectifier.run_full_analysis("detections.txt")
    
    # Compare results
    print("\n2. Mode Comparison Results:")
    print(f"   Conservative mode found: {conservative_results['total_errors_found']} errors")
    print(f"   Aggressive mode found: {aggressive_results['total_errors_found']} errors")
    
    # Error type breakdown
    print("\n   Error Type Comparison:")
    conservative_types = conservative_results['error_types']
    aggressive_types = aggressive_results['error_types']
    
    all_error_types = set(conservative_types.keys()) | set(aggressive_types.keys())
    for error_type in all_error_types:
        conservative_count = conservative_types.get(error_type, 0)
        aggressive_count = aggressive_types.get(error_type, 0)
        print(f"     {error_type}: {conservative_count} (conservative) vs {aggressive_count} (aggressive)")
    
    # Image score comparison
    print("\n   Image Correctness Score Comparison:")
    conservative_scores = conservative_rectifier.image_scores
    aggressive_scores = aggressive_rectifier.image_scores
    
    if conservative_scores and aggressive_scores:
        conservative_avg = np.mean(list(conservative_scores.values()))
        aggressive_avg = np.mean(list(aggressive_scores.values()))
        
        print(f"     Average correctness (conservative): {conservative_avg:.3f}")
        print(f"     Average correctness (aggressive): {aggressive_avg:.3f}")
        
        # Find images with different scores
        different_scores = []
        for img_name in conservative_scores:
            if img_name in aggressive_scores:
                conservative_score = conservative_scores[img_name]
                aggressive_score = aggressive_scores[img_name]
                if abs(conservative_score - aggressive_score) > 0.1:
                    different_scores.append((img_name, conservative_score, aggressive_score))
        
        if different_scores:
            print("\n   Images with significantly different scores:")
            for img_name, cons_score, agg_score in different_scores[:5]:
                print(f"     {img_name}: {cons_score:.3f} (conservative) vs {agg_score:.3f} (aggressive)")
    
    print("\n3. Mode Selection Recommendations:")
    conservative_errors = conservative_results['total_errors_found']
    aggressive_errors = aggressive_results['total_errors_found']
    
    if conservative_errors == 0 and aggressive_errors == 0:
        print("   ✅ Both modes found no errors - GT appears very reliable")
    elif conservative_errors == 0 and aggressive_errors > 0:
        print("   ✅ Conservative mode found no high-confidence errors")
        print("   💡 Aggressive mode found potential issues - review if time permits")
    elif conservative_errors > 0:
        print("   🚨 Conservative mode found high-confidence errors - prioritize these fixes")
        if aggressive_errors > conservative_errors * 2:
            print("   💡 Aggressive mode found many more issues - consider phased review")
    
    return {"conservative": conservative_results, "aggressive": aggressive_results}


def analyze_rectification_results(output_dir):
    """Analyze and display rectification results."""
    
    print("\n" + "="*60)
    print("RECTIFICATION RESULTS ANALYSIS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Load summary data
    summary_file = output_path / "rectification_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\n1. Summary Statistics:")
        creation_info = summary['dataset_creation_info']
        print(f"   Images analyzed: {creation_info['total_images_analyzed']}")
        print(f"   Errors found: {creation_info['total_errors_found']}")
        print(f"   Most reliable images: {creation_info['most_reliable_count']}")
        print(f"   Most problematic images: {creation_info['most_problematic_count']}")
        
        print("\n2. Error Breakdown:")
        error_stats = summary['error_statistics']
        for error_type, count in error_stats['error_types'].items():
            avg_conf = error_stats['avg_confidence_per_error_type'][error_type]
            print(f"   {error_type}: {count} errors (avg confidence: {avg_conf:.3f})")
        
        print("\n3. Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Check directory structure
    print("\n4. Output Structure:")
    for subdir in ['most_correct', 'most_incorrect']:
        subdir_path = output_path / subdir
        if subdir_path.exists():
            image_count = len(list((subdir_path / 'images').glob('*')))
            label_count = len(list((subdir_path / 'labels').glob('*')))
            analysis_count = len(list((subdir_path / 'analysis').glob('*')))
            print(f"   {subdir}/: {image_count} images, {label_count} labels, {analysis_count} analyses")
    
    # Show sample analysis
    print("\n5. Sample Image Analysis:")
    analysis_dir = output_path / "most_incorrect" / "analysis"
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.glob("*.json"))
        if analysis_files:
            with open(analysis_files[0], 'r') as f:
                sample_analysis = json.load(f)
            
            print(f"   Image: {sample_analysis['image_name']}")
            print(f"   Correctness score: {sample_analysis['correctness_score']:.3f}")  
            print(f"   GT detections: {sample_analysis['gt_detection_count']}")
            print(f"   Consensus detections: {sample_analysis['consensus_detection_count']}")
            print(f"   Errors found: {sample_analysis['errors_found']}")
            
            if sample_analysis['recommendations']:
                print(f"   Sample recommendation: {sample_analysis['recommendations'][0][:100]}...")


def cleanup_sample_data():
    """Clean up sample data directory."""
    import shutil
    
    base_dir = Path("sample_rectification_data")
    if base_dir.exists():
        print(f"\nCleaning up sample data directory: {base_dir}")
        shutil.rmtree(base_dir)
        print("✅ Cleanup complete!")


def main():
    """Run complete rectification example."""
    
    print("DetectionFusion GT Rectification System Example")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("• Creating synthetic data with GT errors")
    print("• Running basic rectification analysis (minimize_error mode)")
    print("• Advanced rectification with maximize_error mode") 
    print("• Comparing both modes side by side")
    print("• Analyzing and interpreting results")
    print("• Creating organized datasets for human review")
    
    try:
        # Run basic example
        basic_output = run_basic_rectification()
        
        # Run advanced example
        advanced_output = run_advanced_rectification()
        
        # Run mode comparison
        run_mode_comparison()
        
        # Analyze results
        print("\n" + "="*60)
        print("ANALYZING BASIC RESULTS")
        print("="*60)
        analyze_rectification_results(basic_output)
        
        print("\n" + "="*60)
        print("ANALYZING ADVANCED RESULTS")
        print("="*60) 
        analyze_rectification_results(advanced_output)
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETE!")
        print("="*60)
        print("\n🎉 GT Rectification examples completed successfully!")
        print("\nKey takeaways:")
        print("• Rectification helps identify potential GT annotation errors")
        print("• Two modes available: minimize_error (conservative) vs maximize_error (aggressive)")
        print("• Conservative mode flags only high-confidence errors for immediate attention")
        print("• Aggressive mode identifies more potential issues for comprehensive review")
        print("• Organized output makes human review efficient")
        print("• Consensus across multiple strategies increases confidence")
        print("• Error analysis helps improve overall dataset quality")
        
        print("\n📁 Check these directories for detailed results:")
        print(f"   • Basic results: {basic_output}")
        print(f"   • Advanced results: {advanced_output}")
        
        # Ask if user wants to keep or clean up data
        response = input("\nKeep sample data for manual inspection? (y/n): ")
        if response.lower() != 'y':
            cleanup_sample_data()
    
    except Exception as e:
        print(f"\n❌ Error during example: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)