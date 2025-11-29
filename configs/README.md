# DetectionFusion Configuration Files

Configuration files for the DetectionFusion toolkit v1.0.

**Author:** Abhik Sarkar

## Directory Structure

```
configs/
├── README.md
├── ensemble/                          # Ensemble strategy configurations
│   ├── default.yaml                   # Default balanced configuration
│   ├── high_precision.yaml            # Conservative, high-precision settings
│   ├── high_recall.yaml               # Permissive, high-recall settings
│   ├── small_objects.yaml             # Optimized for small object detection
│   └── advanced.yaml                  # Showcase of all parameters
└── gt_rectification/                  # Ground truth rectification configurations
    ├── conservative.yaml              # Conservative error detection
    ├── aggressive.yaml                # Aggressive error detection
    ├── balanced.yaml                  # Balanced approach
    └── custom.yaml                    # Template for custom configurations
```

## Ensemble Configurations

Located in `configs/ensemble/` - Used with the Python API.

### Usage

**Python API:**
```python
from detection_fusion.config import ConfigLoader, StrategyConfig

# Load from YAML file
config = ConfigLoader.from_yaml("configs/ensemble/default.yaml")

# Use with pipeline
from detection_fusion.pipeline import DetectionPipeline

ctx = (
    DetectionPipeline()
    .with_config(config)
    .load("labels/", format="yolo")
    .ensemble("weighted_vote")
    .run()
)
```

### Configuration Schema

All ensemble configs must match the `StrategyConfig` Pydantic model:

```yaml
overlap:
  threshold: 0.5          # IoU threshold (0.0-1.0)
  method: "iou"           # Overlap method: "iou", "giou", "diou"

voting:
  min_votes: 2            # Minimum votes for majority voting (>=1)
  use_weights: true       # Use model weights in voting

confidence:
  min_threshold: 0.1      # Minimum confidence threshold (0.0-1.0)
  temperature: 1.0        # Softmax temperature for soft voting (>0)

extra: {}                 # Additional strategy-specific parameters
```

### Available Configurations

| File | Purpose | Best For |
|------|---------|----------|
| `default.yaml` | Balanced defaults | General-purpose workflows |
| `high_precision.yaml` | Conservative thresholds | Critical applications |
| `high_recall.yaml` | Permissive thresholds | Maximum detection coverage |
| `small_objects.yaml` | Lower IoU thresholds | Small object datasets |
| `advanced.yaml` | All parameters shown | Reference and customization |

## GT Rectification Configurations

Located in `configs/gt_rectification/` - Used with `detection-fusion rectify`.

### Usage

**CLI:**
```bash
# Load config file
detection-fusion rectify --config configs/gt_rectification/balanced.yaml

# Override paths via CLI
detection-fusion rectify \
    --config configs/gt_rectification/balanced.yaml \
    --labels-dir my_labels/ \
    --gt-dir my_gt/ \
    --output my_output/
```

**Python API:**
```python
from detection_fusion.config import ConfigLoader, RectificationConfig

# Load from YAML file
config = ConfigLoader.load_rectification("configs/gt_rectification/balanced.yaml")

# Access configuration values
print(f"Mode: {config.mode}")
print(f"IoU threshold: {config.thresholds.iou}")
print(f"Output dir: {config.paths.output_dir}")
```

### Configuration Schema

All rectification configs must match the `RectificationConfig` Pydantic model:

```yaml
mode: "minimize_error"    # "minimize_error" (conservative) or "maximize_error" (aggressive)

paths:
  labels_dir: "labels"    # Directory with model predictions
  gt_dir: "GT"            # Directory with ground truth labels
  images_dir: "images"    # Directory with source images
  output_dir: "rectified_dataset"  # Output directory

thresholds:
  iou: 0.5                # IoU threshold for matching (0.0-1.0)
  confidence: 0.5         # Confidence threshold (0.0-1.0)
  min_agreement: 3        # Minimum model agreement (>=1)

output:
  most_correct: 50        # Number of most correct images to include (>=0)
  most_incorrect: 50      # Number of most incorrect images to include (>=0)
  copy_images: true       # Copy source images to output
```

### Available Configurations

| File | Purpose | Best For |
|------|---------|----------|
| `conservative.yaml` | High-precision error detection | Critical datasets |
| `aggressive.yaml` | Maximum error coverage | Initial dataset audit |
| `balanced.yaml` | Balanced approach | Routine quality checks |
| `custom.yaml` | Template with all options | Custom requirements |

## Builder Pattern (Python)

Both config types support a builder pattern for programmatic modification:

```python
from detection_fusion.config import StrategyConfig, RectificationConfig

# Strategy config builder
config = (
    StrategyConfig()
    .with_overlap(threshold=0.6, method="iou")
    .with_voting(min_votes=3, use_weights=True)
    .with_confidence(min_threshold=0.3)
    .with_extra(custom_param=42)
)

# Rectification config builder
rect_config = (
    RectificationConfig()
    .with_paths(labels_dir="my_labels", output_dir="my_output")
    .with_thresholds(iou=0.6, min_agreement=4)
    .with_output(most_correct=100, most_incorrect=100)
)
```

## Quick Reference

```bash
# List available strategies
detection-fusion list-strategies

# List strategies by category
detection-fusion list-strategies --category voting

# Merge detections (config via Python API)
detection-fusion merge -d labels/ -s weighted_vote -o unified/

# Rectify with config file
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
```

## Related Documentation

- [API Reference](../docs/API.md) - Complete Python API documentation
- [Examples](../docs/EXAMPLES.md) - Usage examples with configurations
- [Changelog](../CHANGELOG.md) - Version history and updates
