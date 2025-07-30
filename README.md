# DetectionFusion - Object Detection Ensemble Toolkit

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.1-green.svg)](https://github.com/abhiksark/detection-fusion)

**DetectionFusion: Python toolkit for fusing multiple object detection results with ground truth validation and error analysis.**

Perfect for scenarios where you have multiple object detection models - leverage the wisdom of crowds to improve detection quality through consensus-based ensemble learning, and evaluate performance with rigorous ground truth analysis.

## 🌟 Key Features

- **🤖 Advanced Ensemble Learning**: 19 strategies from simple voting to adaptive methods
- **📊 Ground Truth Evaluation**: Complete evaluation framework with standard object detection metrics
- **🔬 Error Analysis**: Detailed error classification and analysis (FP, FN, localization, classification)
- **🎯 Strategy Optimization**: Automatic best strategy selection using ground truth feedback
- **🔍 GT Rectification**: Identify potential ground truth annotation errors using ensemble consensus with F1-based quality scoring (two modes: conservative & aggressive)
- **📈 Comprehensive Analysis**: Deep insights into model performance and agreement
- **🎨 Rich Visualizations**: Automated plots, reports, and precision-recall curves
- **⚡ Professional CLI Tools**: Intuitive merge.py, val.py, and gt_rectify.py interfaces with GT support and progress tracking
- **🔧 YAML Configuration**: External, flexible configuration management
- **📈 Progress Tracking**: Real-time progress bars for all long-running operations using tqdm
- **📦 Production Ready**: Professional packaging with full documentation and testing

## 📁 Project Structure

```
detection_fusion/
├── __init__.py              # Package initialization and exports
├── core/                    # Core functionality
│   ├── detection.py         # Detection data class with GT support
│   ├── ensemble.py          # Main ensemble classes with GT evaluation
│   └── analyzer.py          # Multi-model analysis with GT comparison
├── strategies/              # Ensemble strategies (17+ available)
│   ├── base.py             # Base strategy class
│   ├── voting.py           # Voting strategies
│   ├── nms.py              # NMS-based strategies
│   ├── clustering.py       # Clustering strategies
│   ├── probabilistic.py    # Probabilistic strategies
│   └── adaptive.py         # Adaptive strategies
├── evaluation/              # Ground truth evaluation framework
│   ├── metrics.py          # EvaluationMetrics, APCalculator
│   ├── error_analysis.py   # ErrorAnalyzer for detailed error classification
│   ├── evaluator.py        # Main evaluation orchestrator
│   └── optimization.py     # StrategyOptimizer for GT-based selection
├── utils/                   # Utility functions
│   ├── io.py               # File I/O with GT loading utilities
│   └── metrics.py          # IoU and evaluation metrics
├── visualization/           # Plotting and analysis visualization
└── tests/                   # Comprehensive unit tests
configs/                     # Organized configuration files
├── README.md               # Configuration guide and documentation
├── ensemble/               # Ensemble and analysis configurations
│   ├── default_config.yaml # Standard ensemble configuration
│   ├── high_precision_config.yaml # Conservative strategies
│   ├── high_recall_config.yaml    # Permissive strategies
│   ├── small_objects_config.yaml  # Small object optimization
│   └── advanced_strategies_config.yaml # All 17+ strategies
└── gt_rectification/       # GT rectification configurations
    ├── gt_rectify_conservative.yaml # High-precision error detection
    ├── gt_rectify_aggressive.yaml  # Comprehensive error detection
    ├── gt_rectify_balanced.yaml    # Balanced approach
    └── gt_rectify_custom.yaml      # Template for customization
examples/                    # Usage examples and demonstrations
├── basic_usage.py           # Basic ensemble usage examples
├── advanced_usage.py        # Advanced strategies and GT evaluation
├── config_usage.py          # Configuration-based examples
└── rectification_example.py # GT rectification examples
merge.py                     # CLI ensemble merging tool with GT optimization
val.py                       # CLI assessment tool with GT evaluation
gt_rectify.py                # GT rectification system with CLI and implementation
```

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/abhiksark/detection-fusion
cd detection-fusion

# Production installation
pip install -r requirements.txt
pip install -e .

# Development environment with testing
pip install -r requirements-dev.txt
pip install -e .

# Alternative: use setup.py extras
pip install -e ".[dev,viz]"
```

### Quick Start - Command Line Interface

```bash
# 1. Set up your data structure
"""
labels/
├── model1/
│   └── detections.txt
├── model2/
│   └── detections.txt
├── model3/
│   └── detections.txt
└── GT/                    # Ground truth (optional)
    └── detections.txt
"""

# 2. Basic ensemble merging
python merge.py --models model1 model2 model3 --strategy weighted_vote --output results.txt

# 3. Ground truth guided strategy selection (NEW!)
# Note: GT evaluation is done via val.py, then use merge.py for the actual ensemble
python val.py --models-dir labels/ --gt --optimize-strategy --output best_strategy_report.txt
python merge.py --models-dir labels/ --strategy weighted_vote --output best_ensemble.txt

# 4. Comprehensive assessment with ground truth evaluation
python val.py --models-dir labels/ --gt --analyze evaluation --error-analysis

# 5. Ground truth rectification - identify annotation errors (NEW!)
# Using configuration files (recommended)
python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml
python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml

# Conservative mode (default): Only high-confidence errors
python gt_rectify.py --labels-dir labels/ --gt-dir GT/ --images-dir images/ --output-dir rectified_dataset

# Aggressive mode: More comprehensive error detection
python gt_rectify.py --labels-dir labels/ --gt-dir GT/ --images-dir images/ --output-dir rectified_dataset --mode maximize_error
```

### Python API Usage

```python
from detection_fusion import AdvancedEnsemble, MultiModelAnalyzer, Evaluator
from detection_fusion.utils import load_yaml_config

# Load configuration
config = load_yaml_config("configs/ensemble/default_config.yaml")

# Basic ensemble with 17+ strategies
ensemble = AdvancedEnsemble("labels")
ensemble.load_detections("detections.txt")

# Run advanced strategies
results = ensemble.run_strategy("multi_scale")
adaptive_results = ensemble.run_strategy("adaptive_threshold")
consensus_results = ensemble.run_strategy("consensus_ranking")

# Ground truth evaluation (NEW!)
if ensemble.has_ground_truth():
    best_strategy, evaluation = ensemble.find_best_strategy_with_gt()
    print(f"Best strategy: {best_strategy}")
    print(f"mAP@0.5: {evaluation['overall_metrics']['map_50']:.3f}")

# Comprehensive analysis
analyzer = MultiModelAnalyzer("labels")
analyzer.load_detections("detections.txt")
analyzer.generate_report("analysis_report.txt")

# Ground truth comparison (NEW!)
if analyzer.has_ground_truth():
    gt_comparison = analyzer.compare_all_models_with_gt()
    print(gt_comparison[['model', 'precision', 'recall', 'f1_score']])
```

## 🎯 Ground Truth Evaluation Framework (NEW in v0.2.0!)

### Complete Ground Truth Support

The toolkit now includes a comprehensive ground truth evaluation system with standard object detection metrics:

#### Setup Ground Truth
```bash
# Add ground truth to your data structure
labels/
├── model1/detections.txt
├── model2/detections.txt
├── model3/detections.txt
└── GT/                         # Ground truth annotations
    ├── detections.txt          # Single file format
    └── *.txt                   # Or per-image format
```

#### CLI Ground Truth Evaluation
```bash
# Find optimal strategy using ground truth (use val.py for GT operations)
python val.py --models-dir labels/ --gt --optimize-strategy --metrics map_50

# Comprehensive evaluation with error analysis
python val.py --models-dir labels/ --gt --analyze evaluation --error-analysis

# Benchmark all strategies against ground truth
python val.py --models-dir labels/ --gt --benchmark-strategies --optimize-strategy

# Compare strategies with detailed GT metrics
python val.py --models-dir labels/ --gt --compare strategies --strategies weighted_vote bayesian nms
```

#### Python API Ground Truth Features
```python
from detection_fusion import Evaluator, ErrorAnalyzer, StrategyOptimizer

# Evaluate predictions against ground truth
evaluator = Evaluator(gt_dir="labels/GT")
evaluation = evaluator.evaluate_predictions(predictions, "detections.txt")

print(f"mAP@0.5: {evaluation['overall_metrics']['map_50']:.3f}")
print(f"Precision: {evaluation['overall_metrics']['precision']:.3f}")
print(f"Recall: {evaluation['overall_metrics']['recall']:.3f}")

# Detailed error analysis
error_analyzer = ErrorAnalyzer()
errors, summary = error_analyzer.analyze_errors(predictions, ground_truth)

print(f"False Positives: {summary.false_positives}")
print(f"False Negatives: {summary.false_negatives}")
print(f"Localization Errors: {summary.localization_errors}")

# Strategy optimization using ground truth
optimizer = StrategyOptimizer(evaluator)
best_result = optimizer.optimize_all_strategies(ensemble)
print(f"Best strategy: {best_result.best_strategy}")

# GT Rectification - Identify annotation errors with F1-based quality scoring (NEW!)
from detection_fusion import GTRectifier

# Conservative mode (default): Only high-confidence errors
conservative_rectifier = GTRectifier(
    labels_dir="labels/",
    gt_dir="labels/GT/", 
    images_dir="images/",
    output_dir="labels/unified/",
    mode="minimize_error"  # Conservative approach
)

results = conservative_rectifier.run_full_analysis("detections.txt")
print(f"High-confidence errors found: {results['total_errors_found']}")
print(f"Average F1 score: {results['average_f1_score']:.3f}")

# Aggressive mode: More comprehensive error detection
aggressive_rectifier = GTRectifier(
    labels_dir="labels/",
    gt_dir="labels/GT/", 
    images_dir="images/",
    output_dir="labels/unified/",
    mode="maximize_error"  # Aggressive approach
)

results = aggressive_rectifier.run_full_analysis("detections.txt")
print(f"All potential errors found: {results['total_errors_found']}")

# Create organized dataset for human review with F1-based ranking
aggressive_rectifier.create_rectified_dataset(
    output_dir="rectified_dataset/",
    include_most_correct=50,    # Highest F1 scores
    include_most_incorrect=50   # Lowest F1 scores
)
```

### Available Ground Truth Metrics

- **Average Precision (AP)** with multiple interpolation methods (11-point, all-points, COCO-style)
- **Mean Average Precision (mAP)** at IoU 0.5 and 0.5:0.95
- **Precision, Recall, F1-Score** with confidence-based thresholds
- **Per-class metrics** for detailed class-wise analysis
- **Precision-Recall curves** for performance visualization

### Error Analysis Categories

- **False Positives**: Predictions with no matching ground truth
- **False Negatives**: Ground truth objects not detected
- **Localization Errors**: Correct class but poor bounding box overlap
- **Classification Errors**: Good localization but wrong class prediction
- **Duplicate Detections**: Multiple predictions for same ground truth object
- **Confidence Analysis**: Error distribution across confidence ranges

## 📋 Detection File Format

Each model's detections should be in a text file with format:
```
class_id x_center y_center width height confidence
0 0.5 0.3 0.2 0.4 0.85
1 0.7 0.6 0.1 0.3 0.92
```

Where coordinates are normalized (0-1) and in YOLO format.

## 🎯 Ensemble Strategies (19 Available!)

### Basic Voting (5 strategies)
- **Majority Voting**: Requires minimum number of models to agree (2, 3, or all)
- **Weighted Voting**: Weights detections by confidence scores and model performance
- **Unanimous**: Only keeps detections ALL models agree on (highest precision)

### NMS-Based (3 strategies)
- **NMS**: Standard Non-Maximum Suppression across all models
- **Affirmative NMS**: NMS requiring agreement from multiple models
- **Confidence-Weighted NMS**: NMS with confidence-weighted box regression

### Clustering (3 strategies)
- **DBSCAN Clustering**: Density-based spatial clustering of detections
- **Centroid Clustering**: Agglomerative clustering based on detection centers
- **Distance-Weighted Voting**: Weights detections by spatial distance to cluster centroid

### Probabilistic (2 strategies)
- **Soft Voting**: Probabilistic voting with temperature scaling
- **Bayesian Fusion**: Bayesian inference with learned class priors

### Confidence-Based (2 strategies)
- **Confidence Threshold Voting**: Adaptive confidence thresholds per model
- **High Confidence First**: Prioritizes high-confidence detections hierarchically

### Adaptive (4 strategies)
- **Adaptive Threshold**: Different IoU thresholds for small vs large objects
- **Density Adaptive**: Context-aware processing for high/low density regions
- **Multi-Scale**: Scale-specific processing (tiny/small/medium/large objects)
- **Consensus Ranking**: Combines model ranking with confidence scores

🔍 **See [Strategy Guide](docs/STRATEGY_GUIDE.md) for detailed explanations and usage examples.**

## 🖥️ Command Line Interface

### merge.py - Ensemble Merging Tool

#### Basic Usage
```bash
# Basic merging
python merge.py --models model1 model2 model3 --strategy weighted_vote

# Multiple strategies comparison
python merge.py --models-dir labels/ --strategies weighted_vote affirmative_nms bayesian

# Custom parameters
python merge.py --models model1 model2 --strategy majority_vote_2 --iou 0.7 --confidence 0.3

# List all available strategies
python merge.py --list-strategies
```

#### Ground Truth Guided Merging (NEW!)
```bash
# Find optimal strategy using ground truth (use val.py for GT evaluation)
python val.py --models-dir labels/ --gt --optimize-strategy --output best_strategy_report.txt

# Optimize for specific metric
python val.py --models-dir labels/ --gt --benchmark-strategies --metrics f1

# Compare strategies with GT evaluation
python val.py --models-dir labels/ --gt --compare strategies --strategies weighted_vote bayesian nms

# Use custom ground truth location
python val.py --models-dir labels/ --gt-dir custom_gt/ --gt --optimize-strategy
```

### val.py - Model Assessment Tool

**Default Mode**: Image-by-image comparison (each .txt file represents detections for one image)

#### Basic Analysis
```bash
# Basic model assessment (image-by-image comparison by default)
python val.py --models model1 model2 model3 --analyze agreement

# Comprehensive analysis with plots
python val.py --models-dir labels/ --report full --plot --save-plots results/

# Class-wise analysis
python val.py --models model1 model2 model3 --analyze class-wise --classes person car bike

# Confidence distribution analysis
python val.py --models-dir labels/ --analyze confidence --confidence-bins 10

# Use single file mode (legacy behavior)
python val.py --models model1 model2 --single-file-mode --analyze agreement

# Handle detections with missing confidence values
python val.py --models model1 model2 --default-confidence 0.8
```

#### Ground Truth Evaluation (NEW!)
```bash
# Complete ground truth evaluation
python val.py --models-dir labels/ --gt --analyze evaluation

# Detailed error analysis
python val.py --models-dir labels/ --gt --error-analysis --analyze error-analysis

# Strategy optimization and benchmarking
python val.py --models-dir labels/ --gt --optimize-strategy --benchmark-strategies

# Custom metrics and ground truth location
python val.py --models-dir labels/ --gt-dir custom_gt/ --metrics precision recall f1 --gt-file ground_truth.txt
```

## 📊 Analysis Capabilities

### Model Comparison
```python
# Compare all model pairs
comparison_df = analyzer.compare_all_models()
print(comparison_df)

# Ground truth comparison (NEW!)
if analyzer.has_ground_truth():
    gt_comparison = analyzer.compare_all_models_with_gt()
    print(gt_comparison[['model', 'precision', 'recall', 'f1_score', 'map_50']])
```

### Class Analysis
```python
# Get detection statistics per class
class_stats = analyzer.get_class_statistics()
print("Top detected classes:")
print(class_stats[['class_name', 'total', 'variance']].head(10))

# Find classes with high disagreement
high_variance = class_stats.nlargest(5, 'variance')
```

### Consensus Detection with Ground Truth Validation
```python
# Find detections multiple models agree on
consensus = analyzer.find_consensus_detections(min_models=2)
print(f"Found {len(consensus)} consensus detections")

# Validate consensus against ground truth (NEW!)
if analyzer.has_ground_truth():
    consensus_analysis = analyzer.analyze_consensus_vs_gt(min_models=2)
    print(f"Consensus Precision: {consensus_analysis['precision']:.3f}")
    print(f"Consensus Recall: {consensus_analysis['recall']:.3f}")
```

## 📈 Visualizations

### Automatic Plot Generation
```python
from detection_fusion.visualization import generate_all_plots

analyzer = MultiModelAnalyzer("labels")
analyzer.load_detections("detections.txt")

# Generate all plots including GT analysis (NEW!)
generate_all_plots(analyzer, output_dir="analysis_plots", include_gt=True)
```

### Available Plots
- **Class Distribution**: Detection counts across models
- **Confidence Histograms**: Score distributions per model
- **Model Similarity Heatmap**: Agreement between model pairs
- **Variance Analysis**: Classes with highest disagreement
- **Precision-Recall Curves**: Performance curves for GT evaluation (NEW!)
- **Error Analysis Plots**: Distribution of error types (NEW!)
- **Strategy Comparison**: GT-based strategy performance (NEW!)

## 📋 Configuration

### Configuration (`configs/ensemble/default_config.yaml`)
```yaml
ensemble:
  labels_dir: "labels"
  output_dir: "labels/unified"
  iou_threshold: 0.5
  # gt_dir: "labels/GT"  # Ground truth directory (can be added when needed)
  
  strategies:
    majority_vote:
      min_votes: 2
    
    weighted_vote:
      use_model_weights: true
    
    dbscan:
      eps: 0.1
      min_samples: 2
    
    soft_voting:
      temperature: 1.0

# Note: The actual default_config.yaml contains only basic ensemble settings
# For GT evaluation workflows, additional sections can be added as needed

visualization:
  figure_size: [14, 8]
  dpi: 300
  style: "seaborn"
  color_palette: "Set2"
```

## 🎨 Output Examples

### Ground Truth Evaluation Output (NEW!)
```
🔍 Running ground truth evaluation...
📊 Evaluating 3 models against ground truth...
  yolov8n: mAP=0.654, F1=0.721
  yolov8s: mAP=0.687, F1=0.743  
  yolov8m: mAP=0.702, F1=0.756

🎯 Finding optimal strategy using ground truth evaluation...
✅ Best strategy found: consensus_ranking
   Performance: map_50=0.7234
   Precision: 0.823, Recall: 0.691, F1: 0.751

📈 Ground Truth Strategy Performance Comparison:
Strategy                  MAP_50     Precision  Recall     F1        
-----------------------------------------------------------------
consensus_ranking         0.723      0.823      0.691      0.751     
bayesian                  0.718      0.834      0.672      0.744     
weighted_vote             0.695      0.798      0.688      0.739     
majority_vote_2           0.671      0.856      0.612      0.714     

🏆 Best performing strategy: consensus_ranking
```

### Error Analysis Output (NEW!)
```
🔬 Running detailed error analysis...
  Analyzing errors for yolov8n...
    Error rate: 0.234
    FP: 145, FN: 89, TP: 567

📊 Error Breakdown:
    False Positives: 145
    False Negatives: 89
    Localization Errors: 23
    Classification Errors: 12
    Duplicate Detections: 8
```

### GT Rectification Output with F1-Based Scoring (NEW!)
```
🔍 DetectionFusion GT Rectification System
📁 Labels directory: labels/
📁 Ground truth directory: gt/
📁 Images directory: images/
📁 Output directory: rectified_dataset/
📊 Scoring Method: F1-based (precision & recall)

🤖 Running comprehensive GT rectification analysis...
✅ Found 17 ensemble strategies available  
📊 Analyzing 150 images...

📈 Analysis Results:
  Total images analyzed: 150
  Total potential errors found: 23
  Average F1 score: 0.734
  Error types: {'missing_in_gt': 14, 'extra_in_gt': 9}

⚠️  Most problematic images (lowest F1 scores):
    image_042: 0.234 (Precision: 0.45, Recall: 0.16)
    image_089: 0.445 (Precision: 0.38, Recall: 0.55)  
    image_156: 0.567 (Precision: 0.72, Recall: 0.46)

✅ Most reliable images (highest F1 scores):
    image_003: 0.987 (Precision: 0.98, Recall: 0.99)
    image_021: 0.934 (Precision: 0.91, Recall: 0.96)
    image_067: 0.923 (Precision: 0.89, Recall: 0.96)

📁 Creating rectified dataset...
  📋 50 most correct images copied for reference (highest F1 scores)
  ⚠️  50 most incorrect images flagged for review (lowest F1 scores)
  📊 Individual analysis files with detailed F1 metrics per image

💡 Next steps:
   1. Review images in rectified_dataset/most_incorrect/
   2. Check analysis files for precision/recall breakdown
   3. Use high-quality images from rectified_dataset/most_correct/ as reference
   4. Read detailed report: rectified_dataset/rectification_report.txt

🎯 GT Rectification Analysis Complete!
```

### Ensemble Results
```
🤖 Auto-selected strategy: consensus_ranking
🎯 Running auto-selected strategy: consensus_ranking
✓ consensus_ranking: 1,156 detections saved

📊 Strategy Performance vs Ground Truth:
  mAP@0.5: 0.723
  Precision: 0.823
  Recall: 0.691
  F1-Score: 0.751

✅ Ensemble merging complete! (with ground truth optimization)
```

## 🛠️ Advanced Usage

### Custom Strategies with Ground Truth Support
```python
from detection_fusion.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, custom_param=0.5):
        super().__init__()
        self.custom_param = custom_param
    
    def merge(self, detections, **kwargs):
        # Your custom logic here
        merged_detections = []
        # ... implementation ...
        return merged_detections
    
    @property
    def name(self):
        return "my_custom_strategy"

# Use with ensemble and GT evaluation
ensemble = AdvancedEnsemble("labels")
ensemble.add_strategy("custom", MyCustomStrategy(custom_param=0.7))

# Evaluate custom strategy with ground truth
if ensemble.has_ground_truth():
    evaluation = ensemble.evaluate_strategy_with_gt("custom")
    print(f"Custom strategy mAP: {evaluation['overall_metrics']['map_50']:.3f}")
```

### Batch Processing with Ground Truth
```python
# Process multiple detection files with GT evaluation
files = ["detections_epoch1.txt", "detections_epoch2.txt", "detections_final.txt"]

for filename in files:
    print(f"Processing {filename}...")
    ensemble.load_detections(filename)
    
    # Find best strategy for this specific file
    if ensemble.has_ground_truth():
        best_strategy, evaluation = ensemble.find_best_strategy_with_gt()
        results = ensemble.run_strategy(best_strategy)
        
        # Save with performance metrics
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = evaluation['overall_metrics']
        
        output_name = f"results_{best_strategy}_{timestamp}_map{metrics['map_50']:.3f}.txt"
        ensemble.save_detections(results, output_name)
    else:
        # Fallback to default strategy
        results = ensemble.run_all_strategies(filename)
        ensemble.save_statistics(results, f"stats_{timestamp}.json")
```

## 🧪 Testing

```bash
# Run all tests including GT evaluation tests
pytest

# With coverage report
pytest --cov=detection_fusion --cov-report=html

# Run specific test categories
pytest tests/test_strategies.py          # Strategy tests
pytest tests/test_evaluation.py         # Ground truth evaluation tests
pytest tests/test_error_analysis.py     # Error analysis tests
```

## 📚 API Reference

### Core Classes

#### `EnsembleVoting`
Main class for basic ensemble operations with GT support.

```python
ensemble = EnsembleVoting(labels_dir="labels", output_dir="output", gt_dir="labels/GT")
ensemble.load_detections("detections.txt")

# Basic ensemble
results = ensemble.run_all_strategies()

# Ground truth evaluation (NEW!)
if ensemble.has_ground_truth():
    best_strategy, evaluation = ensemble.find_best_strategy_with_gt()
```

#### `AdvancedEnsemble`
Extended ensemble with advanced strategies and GT evaluation.

```python
ensemble = AdvancedEnsemble("labels", "output", gt_dir="labels/GT")
ensemble.set_strategy_params("dbscan", eps=0.15)

# Run strategy with GT evaluation
results = ensemble.run_strategy("dbscan")
evaluation = ensemble.evaluate_strategy_with_gt("dbscan")
```

#### `MultiModelAnalyzer`
Comprehensive analysis toolkit with GT comparison.

```python
analyzer = MultiModelAnalyzer("labels", iou_threshold=0.5, gt_dir="labels/GT")
analyzer.load_detections("detections.txt")

# Traditional analysis
stats = analyzer.get_class_statistics()

# Ground truth analysis (NEW!)
if analyzer.has_ground_truth():
    gt_comparison = analyzer.compare_all_models_with_gt()
    consensus_analysis = analyzer.analyze_consensus_vs_gt()
    analyzer.generate_gt_report("gt_analysis.txt")
```

#### `Evaluator` (NEW!)
Main ground truth evaluation orchestrator.

```python
from detection_fusion import Evaluator

evaluator = Evaluator(
    iou_threshold=0.5, 
    confidence_threshold=0.1,
    gt_dir="labels/GT"
)

evaluation = evaluator.evaluate_predictions(predictions, "detections.txt")
detailed_evaluation = evaluator.evaluate_predictions(
    predictions, 
    "detections.txt",
    include_error_analysis=True
)
```

#### `ErrorAnalyzer` (NEW!)
Detailed error classification and analysis.

```python
from detection_fusion.evaluation import ErrorAnalyzer

analyzer = ErrorAnalyzer(iou_threshold=0.5)
errors, summary = analyzer.analyze_errors(predictions, ground_truth)

# Advanced analyses
confidence_analysis = analyzer.analyze_by_confidence(errors)
spatial_analysis = analyzer.analyze_spatial_distribution(errors)
size_analysis = analyzer.analyze_by_object_size(errors)
```

#### `StrategyOptimizer` (NEW!)
Ground truth based strategy optimization.

```python
from detection_fusion.evaluation import StrategyOptimizer

optimizer = StrategyOptimizer(evaluator, optimization_metric='map_50')
optimization_result = optimizer.optimize_all_strategies(ensemble)

print(f"Best strategy: {optimization_result.best_strategy}")
print(f"Best score: {optimization_result.best_score}")
```

### Strategy Classes

All strategies inherit from `BaseStrategy` and support GT evaluation:

```python
from detection_fusion.strategies import MajorityVoting, SoftVoting, BayesianFusion

# Configure and use strategies
voter = MajorityVoting(iou_threshold=0.6, min_votes=3)
merged = voter.merge(detections_dict)

# Strategies automatically work with GT evaluation through ensemble classes
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/abhiksark/detection-fusion
cd detection-fusion

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests including GT evaluation tests
pytest
```

## 📄 What's New in v0.2.1

### Documentation and Quality Improvements
- **Complete Documentation Review**: Comprehensive review and accuracy verification of all documentation files
- **Strategy Count Accuracy**: Updated documentation to reflect actual 19 strategies (was showing 17+)
- **Example Corrections**: Fixed all code examples, import statements, and CLI command examples
- **Consistency Improvements**: Ensured terminology and examples are consistent across all files
- **F1-Based Scoring Documentation**: Enhanced GT rectification documentation with detailed F1-based quality assessment information

## 📄 What's New in v0.2.0

### Major Ground Truth Evaluation Framework
- Complete ground truth evaluation system with standard object detection metrics
- Detailed error analysis with 6 error categories (FP, FN, localization, classification, duplicates)
- Automatic strategy optimization using ground truth feedback
- Enhanced CLI tools with comprehensive GT evaluation modes
- Professional-grade evaluation metrics (AP, mAP, precision, recall, F1-score)
- COCO-style evaluation with multiple interpolation methods

### Enhanced User Experience
- Ground truth guided strategy selection (`--auto-strategy`, `--optimize-strategy`)
- Comprehensive error analysis (`--error-analysis`)
- Strategy benchmarking against ground truth (`--benchmark-strategies`)
- Improved reporting with GT performance metrics
- Enhanced configuration with evaluation sections

See [CHANGELOG.md](CHANGELOG.md) for complete details.

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{detection_fusion,
  title={DetectionFusion: Object Detection Ensemble Toolkit with Ground Truth Evaluation},
  author={DetectionFusion Team},
  year={2024},
  version={0.2.1},
  url={https://github.com/abhiksark/detection-fusion}
}
```

## 🐛 Issues and Support

- **Bug Reports**: [GitHub Issues](https://github.com/abhiksark/detection-fusion/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/abhiksark/detection-fusion/discussions)
- **Documentation**: [Wiki](https://github.com/abhiksark/detection-fusion/docs)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch and scikit-learn
- Inspired by ensemble methods in machine learning and COCO evaluation protocols
- Thanks to the object detection community for valuable feedback

---

**Happy Detection Fusion with Ground Truth! 🎯📊**