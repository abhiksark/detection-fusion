# DetectionFusion - Object Detection Ensemble Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/abhiksark/detection-fusion)

**DetectionFusion** is a Python toolkit for fusing multiple object detection results with ground truth validation and error analysis.

Perfect for scenarios where you have multiple object detection models - leverage the wisdom of crowds to improve detection quality through consensus-based ensemble learning, and evaluate performance with rigorous ground truth analysis.

**Author:** Abhik Sarkar

## Key Features

- **16 Ensemble Strategies**: From simple voting to adaptive multi-scale methods
- **Extensible Format Support**: YOLO, VOC XML, COCO - with easy extension for new formats
- **Pipeline API**: Fluent interface for chaining load -> ensemble -> evaluate operations
- **Modern CLI**: Click-based CLI with Rich output (`detection-fusion` / `dfusion`)
- **Ground Truth Evaluation**: Complete evaluation framework with standard metrics
- **GT Rectification**: Identify potential annotation errors using ensemble consensus
- **Pydantic Models**: Type-safe configuration and detection classes
- **PyPI Ready**: Modern `pyproject.toml` packaging

## Installation

```bash
# Clone the repository
git clone https://github.com/abhiksark/detection-fusion
cd detection-fusion

# Install in development mode
pip install -e ".[dev]"

# Or install with specific extras
pip install -e ".[cli]"      # CLI with Rich output
pip install -e ".[viz]"      # Visualization support
pip install -e ".[full]"     # Everything
```

## Quick Start

### Command Line Interface

```bash
# List available strategies
detection-fusion list-strategies

# List with category filter
detection-fusion list-strategies --category voting -v

# Merge detections from multiple models
detection-fusion merge -d labels/ -s weighted_vote -o unified/

# Validate against ground truth
detection-fusion validate -d labels/ --gt-dir GT/ -s weighted_vote

# Convert annotation formats
detection-fusion convert --input annotations.xml --output labels/ --input-format voc_xml --output-format yolo

# Rectify ground truth annotations
detection-fusion rectify --labels-dir labels/ --gt-dir GT/ --images-dir images/ --output rectified/

# List supported formats
detection-fusion list-formats
```

### Python API

```python
from detection_fusion import (
    Detection,
    DetectionSet,
    merge_detections,
    evaluate_detections,
    convert_annotations,
)

# Quick merge with convenience function
results = merge_detections("labels/", strategy="weighted_vote")

# Or use the Pipeline API for more control
from detection_fusion.pipeline import DetectionPipeline

pipeline = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .evaluate("GT/")
)
ctx = pipeline.run()

print(f"Merged {len(ctx.ensemble_result)} detections")
print(f"Precision: {ctx.evaluation_result.precision:.3f}")
print(f"Recall: {ctx.evaluation_result.recall:.3f}")
```

### Working with Detections

```python
from detection_fusion import Detection, DetectionSet

# Create a detection (Pydantic model - immutable)
det = Detection(
    class_id=0,
    x=0.5, y=0.5,  # Center coordinates (normalized)
    w=0.2, h=0.3,  # Width/height (normalized)
    confidence=0.95,
    model_source="yolov8n",
    image_name="image_001"
)

# Access properties
print(det.bbox)    # [0.5, 0.5, 0.2, 0.3]
print(det.xyxy)    # [0.4, 0.35, 0.6, 0.65]
print(det.area)    # 0.06
print(det.center)  # (0.5, 0.5)

# Create modified copies (immutable)
det2 = det.with_confidence(0.8)
det3 = det.with_source("yolov8s")

# Calculate IoU
iou = det.iou_with(det2)

# Work with DetectionSet for filtering/grouping
detections = {
    "model1": [det1, det2],
    "model2": [det3, det4],
}
ds = DetectionSet(detections)

# Filter and group
high_conf = ds.filter_by_confidence(0.8)
by_class = ds.group_by_class()
by_image = ds.group_by_image()

# Statistics
stats = ds.confidence_stats()
print(f"Mean confidence: {stats['mean']:.3f}")
```

### Using Strategies Directly

```python
from detection_fusion.strategies import StrategyRegistry, create_strategy

# List all strategies
strategies = StrategyRegistry.list_all()
print(f"Available: {strategies}")

# Get strategy info
info = StrategyRegistry.get_info("weighted_vote")
print(f"Category: {info.category}")
print(f"Parameters: {info.parameters}")

# Create and use a strategy
strategy = create_strategy("weighted_vote", iou_threshold=0.6)
results = strategy.merge(detections)

# Validate parameters with schema
validated = strategy.validate_params(iou_threshold=0.5, use_model_weights=True)
```

### Format Conversion

```python
from detection_fusion.data.formats import FormatRegistry

# List available formats
formats = FormatRegistry.list_formats()
print(f"Readers: {formats['readers']}")
print(f"Writers: {formats['writers']}")

# Auto-detect format
reader = FormatRegistry.auto_detect_reader("annotations/")
detections = reader.read_directory("annotations/")

# Convert formats
from detection_fusion import convert_annotations

convert_annotations(
    input_path="annotations.xml",
    output_path="labels/",
    input_format="voc_xml",
    output_format="yolo"
)
```

## Available Strategies (16)

| Category | Strategies | Description |
|----------|------------|-------------|
| **Voting** | `majority_vote`, `weighted_vote` | Consensus-based merging |
| **NMS** | `nms`, `affirmative_nms` | Non-maximum suppression variants |
| **Clustering** | `dbscan` | Density-based spatial clustering |
| **Probabilistic** | `soft_voting`, `bayesian` | Probabilistic fusion methods |
| **Distance-Based** | `distance_weighted`, `centroid_clustering` | Spatial relationship methods |
| **Confidence-Based** | `confidence_threshold`, `confidence_weighted_nms`, `high_confidence_first` | Confidence-aware processing |
| **Adaptive** | `adaptive_threshold`, `density_adaptive`, `multi_scale`, `consensus_ranking` | Context-aware strategies |

## Supported Formats

| Format | Read | Write | Auto-Detect |
|--------|------|-------|-------------|
| YOLO (.txt) | Yes | Yes | Yes |
| VOC XML (.xml) | Yes | Yes | Yes |
| COCO JSON (.json) | Yes | Yes | Yes |

## Configuration

### Strategy Config (Pydantic)

```python
from detection_fusion.config import StrategyConfig

# Builder pattern for config
config = (
    StrategyConfig()
    .with_overlap(threshold=0.6, method="iou")
    .with_voting(min_votes=3, use_weights=True)
    .with_confidence(min_threshold=0.3)
)

# Use with strategy
from detection_fusion.strategies import create_strategy
strategy = create_strategy("weighted_vote", config=config)
```

### YAML Configuration

```yaml
# configs/ensemble/default.yaml - Must match StrategyConfig Pydantic model
overlap:
  threshold: 0.5
  method: "iou"

voting:
  min_votes: 2
  use_weights: true

confidence:
  min_threshold: 0.1
  temperature: 1.0

extra: {}
```

```yaml
# configs/gt_rectification/balanced.yaml - Must match RectificationConfig model
mode: "minimize_error"

paths:
  labels_dir: "labels"
  gt_dir: "GT"
  images_dir: "images"
  output_dir: "rectified_balanced"

thresholds:
  iou: 0.5
  confidence: 0.5
  min_agreement: 3

output:
  most_correct: 50
  most_incorrect: 50
  copy_images: true
```

Load configs via CLI or Python:

```bash
# Rectify supports --config option
detection-fusion rectify --config configs/gt_rectification/balanced.yaml

# Merge uses CLI options directly
detection-fusion merge -d labels/ -s weighted_vote -o unified/ --iou-threshold 0.5
```

```python
from detection_fusion.config import ConfigLoader

# Load ensemble config for Python API
config = ConfigLoader.from_yaml("configs/ensemble/default.yaml")

# Load rectification config
rect_config = ConfigLoader.load_rectification("configs/gt_rectification/balanced.yaml")
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=detection_fusion --cov-report=html

# Run specific test file
pytest tests/strategies/test_registry.py -v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format and lint
make format
make lint

# Or manually
ruff format detection_fusion tests examples
ruff check detection_fusion tests examples

# Type checking
mypy detection_fusion
```

## Citation

```bibtex
@software{detection_fusion,
  title={DetectionFusion: Object Detection Ensemble Toolkit},
  author={Sarkar, Abhik},
  year={2025},
  version={1.0.0},
  url={https://github.com/abhiksark/detection-fusion}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Documentation

- [Strategy Guide](docs/STRATEGY_GUIDE.md) - Detailed guide to all ensemble strategies
- [API Reference](https://github.com/abhiksark/detection-fusion#python-api) - Python API documentation
- [Examples](examples/) - Example scripts and usage patterns

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Abhik Sarkar** - [GitHub](https://github.com/abhiksark)
