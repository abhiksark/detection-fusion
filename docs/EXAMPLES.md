# Usage Examples

Comprehensive examples demonstrating various use cases of the DetectionFusion package.

## 🚀 Quick Start Examples

### Basic Ensemble Voting
```python
from detection_fusion import EnsembleVoting

# Create ensemble with default settings
ensemble = EnsembleVoting(labels_dir="labels", output_dir="results")

# Load detections from all models
ensemble.load_detections("detections.txt")

# Run all basic strategies
results = ensemble.run_all_strategies()

# Print summary
for strategy, detections in results.items():
    print(f"{strategy}: {len(detections)} detections")
```

### Model Analysis
```python
from detection_fusion import MultiModelAnalyzer

# Create analyzer
analyzer = MultiModelAnalyzer("labels", iou_threshold=0.5)
analyzer.load_detections("detections.txt")

# Get basic statistics
class_stats = analyzer.get_class_statistics()
print("Top 5 detected classes:")
print(class_stats[['total', 'variance']].head())

# Generate comprehensive report
analyzer.generate_report("analysis_report.txt")
```

## 🎯 Label-Free Model Evaluation

### Cross-Model Validation
```python
from detection_fusion import AdvancedEnsemble

# For scenarios without ground truth labels
ensemble = AdvancedEnsemble("labels")
ensemble.load_detections("detections.txt")

# Conservative approach: Only objects 3+ models detect
strict_consensus = ensemble.run_strategy("affirmative_nms", min_models=3)
print(f"High-confidence detections: {len(strict_consensus)}")

# Moderate approach: Majority voting
moderate_consensus = ensemble.run_strategy("majority_vote_2")
print(f"Moderate-confidence detections: {len(moderate_consensus)}")

# Permissive approach: Any model detection with NMS
permissive_results = ensemble.run_strategy("nms")
print(f"All detections (NMS filtered): {len(permissive_results)}")
```

### Model Reliability Assessment
```python
# Analyze which models agree most often
comparison_matrix = analyzer.compare_all_models()
print("\nModel Agreement Matrix:")
print(comparison_matrix.pivot_table(
    index='model1', columns='model2', 
    values='total_matches', fill_value=0
))

# Find models with consistent confidence patterns
conf_stats = analyzer.get_confidence_statistics()
print("\nModel Confidence Statistics:")
print(conf_stats[['model', 'mean', 'std']].sort_values('std'))
```

## 🔧 Advanced Configuration

### Custom Strategy Parameters
```python
from detection_fusion import AdvancedEnsemble

ensemble = AdvancedEnsemble("labels")

# Customize DBSCAN parameters for different object sizes
ensemble.set_strategy_params("dbscan", eps=0.05, min_samples=3)  # Small objects
results_small = ensemble.run_strategy("dbscan")

ensemble.set_strategy_params("dbscan", eps=0.15, min_samples=2)  # Large objects  
results_large = ensemble.run_strategy("dbscan")

# Customize soft voting temperature
ensemble.set_strategy_params("soft_voting", temperature=0.5)  # More decisive
results_sharp = ensemble.run_strategy("soft_voting")

ensemble.set_strategy_params("soft_voting", temperature=2.0)  # Less decisive
results_smooth = ensemble.run_strategy("soft_voting")
```

### Multi-File Processing
```python
import os
from pathlib import Path

def process_multiple_experiments():
    """Process detection files from multiple experiments."""
    
    # Different experiment files
    experiment_files = [
        "detections_epoch_10.txt",
        "detections_epoch_20.txt", 
        "detections_final.txt"
    ]
    
    ensemble = AdvancedEnsemble("labels")
    results_summary = {}
    
    for filename in experiment_files:
        if Path(f"labels/model1/{filename}").exists():
            print(f"\nProcessing {filename}...")
            
            # Load and process
            ensemble.load_detections(filename)
            results = ensemble.run_strategy("bayesian")
            
            # Store results
            epoch = filename.replace("detections_", "").replace(".txt", "")
            results_summary[epoch] = {
                'count': len(results),
                'avg_confidence': sum(d.confidence for d in results) / len(results) if results else 0
            }
            
            # Save epoch-specific results
            from detection_fusion.utils.io import save_detections
            save_detections(results, f"results/ensemble_{epoch}.txt")
    
    return results_summary

# Usage
summary = process_multiple_experiments()
for epoch, stats in summary.items():
    print(f"{epoch}: {stats['count']} detections, avg conf: {stats['avg_confidence']:.3f}")
```

## 📊 Visualization Examples

### Comprehensive Analysis Plots
```python
from detection_fusion import MultiModelAnalyzer
from detection_fusion.visualization import generate_all_plots

# Setup analyzer
analyzer = MultiModelAnalyzer("labels")
analyzer.load_detections("detections.txt")

# Load class names for better visualization
analyzer.load_class_names("class_names.txt")

# Generate all plots
generate_all_plots(analyzer, output_dir="analysis_plots", top_n=20)

# Individual plot examples
analyzer.plot_class_distribution(top_n=15, save_path="class_dist.png")
analyzer.plot_confidence_distribution(save_path="confidence_dist.png")
analyzer.plot_model_comparison_heatmap(save_path="model_similarity.png")
```

### Custom Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_ensemble_comparison():
    """Compare different ensemble strategies."""
    
    strategies = ["majority_vote_2", "weighted_vote", "bayesian", "nms"]
    detection_counts = []
    avg_confidences = []
    
    for strategy in strategies:
        results = ensemble.run_strategy(strategy)
        detection_counts.append(len(results))
        
        if results:
            avg_conf = np.mean([d.confidence for d in results])
            avg_confidences.append(avg_conf)
        else:
            avg_confidences.append(0)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Detection counts
    ax1.bar(strategies, detection_counts)
    ax1.set_title("Detection Counts by Strategy")
    ax1.set_ylabel("Number of Detections")
    ax1.tick_params(axis='x', rotation=45)
    
    # Average confidences
    ax2.bar(strategies, avg_confidences)
    ax2.set_title("Average Confidence by Strategy")
    ax2.set_ylabel("Average Confidence")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_ensemble_comparison()
```

## 🔬 Custom Strategy Development

### Simple Custom Strategy
```python
from detection_fusion.strategies.base import BaseStrategy
from detection_fusion.core.detection import Detection
from detection_fusion.utils.metrics import calculate_iou
import numpy as np

class HighConfidenceVoting(BaseStrategy):
    """Only consider detections above confidence threshold."""
    
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.7):
        super().__init__(iou_threshold)
        self.confidence_threshold = confidence_threshold
    
    @property
    def name(self):
        return f"high_confidence_{self.confidence_threshold}"
    
    def merge(self, detections, **kwargs):
        # Filter by confidence
        filtered_detections = {}
        for model, dets in detections.items():
            filtered_detections[model] = [
                d for d in dets if d.confidence >= self.confidence_threshold
            ]
        
        # Apply simple majority voting to filtered detections
        from detection_fusion.strategies import MajorityVoting
        voter = MajorityVoting(self.iou_threshold, min_votes=2)
        return voter.merge(filtered_detections)

# Usage
ensemble = AdvancedEnsemble("labels")
custom_strategy = HighConfidenceVoting(confidence_threshold=0.8)
ensemble.add_strategy("high_conf", custom_strategy)

results = ensemble.run_strategy("high_conf")
print(f"High confidence results: {len(results)}")
```

### Advanced Custom Strategy
```python
class AdaptiveThresholdVoting(BaseStrategy):
    """Voting with adaptive IoU thresholds based on object size."""
    
    def __init__(self, small_threshold=0.3, large_threshold=0.7, size_cutoff=0.1):
        super().__init__(iou_threshold=0.5)  # Default, will be overridden
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        self.size_cutoff = size_cutoff
    
    @property
    def name(self):
        return "adaptive_threshold"
    
    def merge(self, detections, **kwargs):
        # Separate by object size
        small_objects = {}
        large_objects = {}
        
        for model, dets in detections.items():
            small_objects[model] = []
            large_objects[model] = []
            
            for det in dets:
                object_size = det.w * det.h
                if object_size < self.size_cutoff:
                    small_objects[model].append(det)
                else:
                    large_objects[model].append(det)
        
        # Apply different thresholds
        from detection_fusion.strategies import MajorityVoting
        
        small_voter = MajorityVoting(self.small_threshold, min_votes=2)
        large_voter = MajorityVoting(self.large_threshold, min_votes=2)
        
        small_results = small_voter.merge(small_objects)
        large_results = large_voter.merge(large_objects)
        
        return small_results + large_results

# Usage
adaptive_strategy = AdaptiveThresholdVoting(
    small_threshold=0.3, 
    large_threshold=0.7, 
    size_cutoff=0.05
)
ensemble.add_strategy("adaptive", adaptive_strategy)
```

## 🚀 Production Workflows

### Automated Pipeline
```python
import json
from datetime import datetime
from pathlib import Path

class ProductionPipeline:
    """Production-ready ensemble pipeline."""
    
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.ensemble = AdvancedEnsemble(
            self.config["labels_dir"],
            self.config["output_dir"]
        )
        
    def run_pipeline(self, detection_file):
        """Run complete analysis pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Load and validate data
        try:
            self.ensemble.load_detections(detection_file)
            print(f"✅ Loaded detections from {len(self.ensemble.models)} models")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None
        
        # 2. Run ensemble strategies
        strategies = self.config.get("strategies", ["majority_vote_2", "bayesian"])
        results = {}
        
        for strategy in strategies:
            try:
                result = self.ensemble.run_strategy(strategy)
                results[strategy] = result
                print(f"✅ {strategy}: {len(result)} detections")
            except Exception as e:
                print(f"❌ {strategy} failed: {e}")
        
        # 3. Save results
        output_dir = Path(self.config["output_dir"]) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for strategy, detections in results.items():
            output_file = output_dir / f"{strategy}_results.txt"
            from detection_fusion.utils.io import save_detections
            save_detections(detections, str(output_file))
        
        # 4. Generate analysis report
        analyzer = MultiModelAnalyzer(self.config["labels_dir"])
        analyzer.load_detections(detection_file)
        report_path = output_dir / "analysis_report.txt"
        analyzer.generate_report(str(report_path))
        
        # 5. Save metadata
        metadata = {
            "timestamp": timestamp,
            "detection_file": detection_file,
            "models": self.ensemble.models,
            "strategies": list(results.keys()),
            "total_detections": {k: len(v) for k, v in results.items()}
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Pipeline complete. Results saved to {output_dir}")
        return results

# Usage
pipeline = ProductionPipeline("production_config.json")
results = pipeline.run_pipeline("detections.txt")
```

### Batch Processing with Monitoring
```python
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_logging():
    """Setup production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ensemble_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_single_file(args):
    """Process a single detection file."""
    file_path, config = args
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        
        # Setup ensemble
        ensemble = AdvancedEnsemble(config["labels_dir"])
        ensemble.load_detections(file_path.name)
        
        # Run strategies
        results = {}
        for strategy in config["strategies"]:
            result = ensemble.run_strategy(strategy)
            results[strategy] = len(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {file_path.name} in {processing_time:.2f}s")
        
        return file_path.name, results, processing_time
        
    except Exception as e:
        logger.error(f"Failed to process {file_path.name}: {e}")
        return file_path.name, None, 0

def batch_process_files(file_list, config, max_workers=4):
    """Process multiple files in parallel."""
    logger = setup_logging()
    
    # Prepare arguments
    args_list = [(file_path, config) for file_path in file_list]
    
    # Process in parallel
    results = {}
    total_time = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, args): args[0] 
            for args in args_list
        }
        
        # Collect results
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                filename, file_results, proc_time = future.result()
                if file_results:
                    results[filename] = file_results
                    total_time += proc_time
                    logger.info(f"✅ {filename} completed")
                else:
                    logger.error(f"❌ {filename} failed")
            except Exception as e:
                logger.error(f"❌ {file_path.name} crashed: {e}")
    
    logger.info(f"Batch processing complete. Total time: {total_time:.2f}s")
    return results

# Usage
from pathlib import Path

config = {
    "labels_dir": "labels",
    "strategies": ["majority_vote_2", "weighted_vote", "bayesian"]
}

# Find all detection files
detection_files = list(Path("experiments").glob("*/detections.txt"))
results = batch_process_files(detection_files, config, max_workers=8)
```

## 📋 Scripted Workflows

### Automated Python Pipeline
```python
#!/usr/bin/env python3
"""Automated ensemble processing script."""

import json
from datetime import datetime
from pathlib import Path
from detection_fusion import AdvancedEnsemble, MultiModelAnalyzer
from detection_fusion.visualization import generate_all_plots

def automated_ensemble_pipeline():
    """Run automated ensemble analysis pipeline."""
    
    # Configuration
    labels_dir = "experiments/final_models/labels"
    output_dir = Path(f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strategies = ["majority_vote_2", "weighted_vote", "bayesian", "soft_voting"]
    iou_threshold = 0.5
    
    print("Starting ensemble analysis...")
    
    # Setup ensemble
    ensemble = AdvancedEnsemble(labels_dir, str(output_dir))
    ensemble.load_detections("detections.txt")
    
    # Run strategies
    results = {}
    for strategy in strategies:
        try:
            result = ensemble.run_strategy(strategy)
            results[strategy] = result
            
            # Save results
            from detection_fusion.utils.io import save_detections
            save_detections(result, str(output_dir / f"{strategy}_results.txt"))
            print(f"✅ {strategy}: {len(result)} detections")
            
        except Exception as e:
            print(f"❌ {strategy} failed: {e}")
    
    # Generate analysis
    analyzer = MultiModelAnalyzer(labels_dir, iou_threshold)
    analyzer.load_detections("detections.txt")
    analyzer.generate_report(str(output_dir / "analysis_report.txt"))
    
    # Generate plots
    generate_all_plots(analyzer, str(output_dir / "plots"))
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "labels_dir": labels_dir,
        "iou_threshold": iou_threshold,
        "models": ensemble.models,
        "strategies": list(results.keys()),
        "detection_counts": {k: len(v) for k, v in results.items()}
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Pipeline complete! Results in {output_dir}")
    return results

if __name__ == "__main__":
    automated_ensemble_pipeline()
```

These examples demonstrate the flexibility and power of the DetectionFusion package across different use cases, from simple voting to complex production pipelines.