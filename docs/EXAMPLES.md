# Usage Examples

Comprehensive examples demonstrating various use cases of the DetectionFusion package v1.0.0.

**Author:** Abhik Sarkar

## Quick Start Examples

### Basic Merge with Convenience Function

```python
from detection_fusion import merge_detections

# Simplest usage - one line
results = merge_detections("labels/", strategy="weighted_vote")
print(f"Merged {len(results)} detections")
```

### Pipeline API

```python
from detection_fusion.pipeline import DetectionPipeline

# Fluent pipeline for full workflow
ctx = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .run()
)

print(f"Loaded from {len(ctx.detections.model_names)} models")
print(f"Merged to {len(ctx.ensemble_result)} detections")
```

### Working with Detections

```python
from detection_fusion import Detection, DetectionSet

# Create a detection (Pydantic model - immutable)
det = Detection(
    class_id=0,
    x=0.5, y=0.5,
    w=0.2, h=0.3,
    confidence=0.95,
    model_source="yolov8n",
    image_name="image_001"
)

# Properties
print(f"BBox: {det.bbox}")      # [0.5, 0.5, 0.2, 0.3]
print(f"XYXY: {det.xyxy}")      # [0.4, 0.35, 0.6, 0.65]
print(f"Area: {det.area}")      # 0.06
print(f"Center: {det.center}")  # (0.5, 0.5)

# Create modified copies (immutable pattern)
high_conf = det.with_confidence(0.99)
different_model = det.with_source("yolov8s")
```

## DetectionSet Operations

### Filtering and Grouping

```python
from detection_fusion import Detection, DetectionSet

# Create detection set from multiple models
detections = {
    "yolov8n": [...],  # List of Detection objects
    "yolov8s": [...],
    "yolov8m": [...],
}
ds = DetectionSet(detections)

# Filter by confidence
high_conf = ds.filter_by_confidence(0.8)
print(f"High confidence: {high_conf.total_count} detections")

# Filter by class
persons = ds.filter_by_class([0])  # COCO class 0 = person
vehicles = ds.filter_by_class([2, 5, 7])  # car, bus, truck

# Chain filters
result = (
    ds.filter_by_confidence(0.7)
      .filter_by_class([0, 1])
)

# Group operations
by_image = ds.group_by_image()  # Dict[str, DetectionSet]
by_class = ds.group_by_class()  # Dict[int, DetectionSet]

# Statistics
stats = ds.confidence_stats()
print(f"Mean confidence: {stats['mean']:.3f}")
print(f"Std: {stats['std']:.3f}")

dist = ds.class_distribution()
print(f"Class distribution: {dist}")
```

### Model-Specific Operations

```python
# Get detections from a specific model
yolo_dets = ds.by_model("yolov8n")

# Get detections for a specific image
img_dets = ds.by_image("image_001")

# List all models and images
print(f"Models: {ds.model_names}")
print(f"Images: {ds.image_names}")
print(f"Total: {ds.total_count}")
```

## Strategy Usage

### Using the Registry

```python
from detection_fusion.strategies import (
    StrategyRegistry,
    create_strategy,
    list_strategies,
)

# List all available strategies
strategies = list_strategies()
print(f"Available: {strategies}")

# List by category
voting = StrategyRegistry.list_by_category("voting")
nms = StrategyRegistry.list_by_category("nms")

# Get strategy info
info = StrategyRegistry.get_info("weighted_vote")
print(f"Category: {info.category}")
print(f"Description: {info.description}")
print(f"Parameters: {info.parameters}")

# Create strategy with custom parameters
strategy = create_strategy("weighted_vote", iou_threshold=0.6)
```

### Direct Strategy Usage

```python
from detection_fusion.strategies import create_strategy

# Create and use strategies
strategy = create_strategy("weighted_vote", iou_threshold=0.5)

detections = {
    "model1": [...],
    "model2": [...],
    "model3": [...],
}

results = strategy.merge(detections)
print(f"Merged: {len(results)} detections")
```

### Comparing Strategies

```python
from detection_fusion.strategies import create_strategy, list_strategies

strategies_to_compare = ["majority_vote", "weighted_vote", "nms", "bayesian"]

detections = {
    "model1": [...],
    "model2": [...],
    "model3": [...],
}

results = {}
for name in strategies_to_compare:
    strategy = create_strategy(name)
    merged = strategy.merge(detections)
    results[name] = {
        "count": len(merged),
        "avg_conf": sum(d.confidence for d in merged) / len(merged) if merged else 0
    }

for name, stats in results.items():
    print(f"{name}: {stats['count']} detections, avg conf: {stats['avg_conf']:.3f}")
```

## Configuration

### Using StrategyConfig

```python
from detection_fusion.config import StrategyConfig

# Builder pattern
config = (
    StrategyConfig()
    .with_overlap(threshold=0.6, method="iou")
    .with_voting(min_votes=3, use_weights=True)
    .with_confidence(min_threshold=0.3)
    .with_extra(custom_param=42)
)

# Use with strategy
from detection_fusion.strategies import create_strategy
strategy = create_strategy("weighted_vote", config=config)
```

### Parameter Validation

```python
from detection_fusion.strategies import create_strategy

strategy = create_strategy("weighted_vote")

# Validate parameters against schema
try:
    validated = strategy.validate_params(
        iou_threshold=0.5,
        use_model_weights=True
    )
    print(f"Validated: {validated}")
except ValueError as e:
    print(f"Invalid params: {e}")

# Get defaults
defaults = strategy.get_param_defaults()
print(f"Defaults: {defaults}")
```

## Format Handling

### Format Conversion

```python
from detection_fusion import convert_annotations

# Convert VOC XML to YOLO
convert_annotations(
    input_path="annotations/",
    output_path="labels/",
    input_format="voc_xml",
    output_format="yolo"
)

# Convert COCO JSON to YOLO
convert_annotations(
    input_path="annotations.json",
    output_path="labels/",
    input_format="coco",
    output_format="yolo"
)
```

### Using Format Registry

```python
from detection_fusion.data.formats import FormatRegistry

# List available formats
formats = FormatRegistry.list_formats()
print(f"Readers: {formats['readers']}")
print(f"Writers: {formats['writers']}")

# Get reader for specific format
reader = FormatRegistry.get_reader("yolo")
detections = reader.read_directory("labels/model1/")

# Auto-detect format
reader = FormatRegistry.auto_detect_reader("annotations/")
detections = reader.read_directory("annotations/")

# Write detections
writer = FormatRegistry.get_writer("yolo")
writer.write_directory(detections_by_image, "output/")
```

### Custom Format Reader

```python
from detection_fusion.data.formats.base import AnnotationReader
from detection_fusion.data.formats import FormatRegistry
from detection_fusion import Detection
from pathlib import Path

@FormatRegistry.register_reader("custom")
class CustomReader(AnnotationReader):
    format_name = "custom"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        path = Path(path)
        return any(path.glob("*.custom"))

    def read_file(self, path: Path) -> list:
        detections = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                det = Detection(
                    class_id=int(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    confidence=float(parts[5]) if len(parts) > 5 else 1.0,
                    image_name=path.stem
                )
                detections.append(det)
        return detections

    def read_directory(self, path: Path) -> dict:
        path = Path(path)
        result = {}
        for file in path.glob("*.custom"):
            result[file.stem] = self.read_file(file)
        return result
```

## CLI Examples

### List Commands

```bash
# List all strategies
detection-fusion list-strategies

# List by category with verbose output
detection-fusion list-strategies --category voting -v

# List available formats
detection-fusion list-formats
```

### Merge Command

```bash
# Basic merge
detection-fusion merge --input labels/ --strategy weighted_vote --output unified/

# With custom parameters
detection-fusion merge -i labels/ -s majority_vote -o unified/ \
    --iou 0.6 --confidence 0.3

# Using specific format
detection-fusion merge --input labels/ --strategy nms --output results/ \
    --format yolo
```

### Convert Command

```bash
# VOC XML to YOLO
detection-fusion convert \
    --input annotations/ \
    --output labels/ \
    --input-format voc_xml \
    --output-format yolo

# COCO to YOLO
detection-fusion convert \
    --input annotations.json \
    --output labels/ \
    --input-format coco \
    --output-format yolo
```

### Rectify Command

```bash
# Conservative mode (default)
detection-fusion rectify \
    --labels-dir labels/ \
    --gt-dir GT/ \
    --images-dir images/ \
    --output rectified/

# Aggressive mode
detection-fusion rectify \
    --labels-dir labels/ \
    --gt-dir GT/ \
    --images-dir images/ \
    --output rectified/ \
    --mode maximize_error

# With custom thresholds
detection-fusion rectify \
    --labels-dir labels/ \
    --gt-dir GT/ \
    --images-dir images/ \
    --output rectified/ \
    --iou-threshold 0.6 \
    --confidence-threshold 0.4 \
    --min-agreement 3
```

## Production Workflows

### Automated Pipeline

```python
from detection_fusion.pipeline import DetectionPipeline
from detection_fusion.config import StrategyConfig
from pathlib import Path
import json

def production_pipeline(input_dir: str, output_dir: str):
    """Production-ready ensemble pipeline."""

    # Configure
    config = (
        StrategyConfig()
        .with_overlap(threshold=0.5)
        .with_voting(min_votes=2)
    )

    # Run pipeline
    ctx = (
        DetectionPipeline()
        .with_config(config)
        .load(input_dir, format="yolo")
        .ensemble("weighted_vote")
        .run()
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write merged detections
    from detection_fusion.data.formats import FormatRegistry
    writer = FormatRegistry.get_writer("yolo")

    # Group by image and write
    detections_by_image = {}
    for det in ctx.ensemble_result:
        img = det.image_name or "unknown"
        if img not in detections_by_image:
            detections_by_image[img] = []
        detections_by_image[img].append(det)

    writer.write_directory(detections_by_image, output_path / "labels")

    # Save metadata
    metadata = {
        "input_dir": input_dir,
        "models": ctx.detections.model_names,
        "total_input": ctx.detections.total_count,
        "total_output": len(ctx.ensemble_result),
        "strategy": "weighted_vote",
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return ctx

# Usage
ctx = production_pipeline("data/labels/", "output/")
print(f"Processed {ctx.detections.total_count} -> {len(ctx.ensemble_result)}")
```

### Multi-Strategy Comparison

```python
from detection_fusion.strategies import create_strategy, list_strategies
from detection_fusion.data.loader import FileDetectionLoader

def compare_strategies(input_dir: str, strategies: list = None):
    """Compare multiple strategies on the same data."""

    # Load data
    loader = FileDetectionLoader(input_dir, format="yolo")
    ds = loader.load_as_set()

    # Get raw detections dict
    detections = ds.raw_data

    # Default strategies
    if strategies is None:
        strategies = ["majority_vote", "weighted_vote", "nms", "bayesian"]

    results = {}
    for name in strategies:
        strategy = create_strategy(name)
        merged = strategy.merge(detections)

        results[name] = {
            "count": len(merged),
            "avg_confidence": sum(d.confidence for d in merged) / len(merged) if merged else 0,
            "class_distribution": {},
        }

        # Count per class
        for det in merged:
            cls = det.class_id
            results[name]["class_distribution"][cls] = \
                results[name]["class_distribution"].get(cls, 0) + 1

    return results

# Usage
results = compare_strategies("labels/")
for strategy, stats in results.items():
    print(f"{strategy}: {stats['count']} detections, "
          f"avg conf: {stats['avg_confidence']:.3f}")
```

## Custom Strategy Development

### Simple Custom Strategy

```python
from detection_fusion.strategies.base import BaseStrategy, StrategyMetadata
from detection_fusion.strategies.params import ParamSchema, ParamSpec
from detection_fusion import Detection

# Define parameter schema
MY_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
        ),
        ParamSpec(
            name="min_confidence",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
        ),
    ]
)

class HighConfidenceVoting(BaseStrategy):
    """Only consider high-confidence detections."""

    metadata = StrategyMetadata(
        name="high_confidence_voting",
        category="confidence_based",
        description="Filter by confidence then apply majority voting",
        params_schema=MY_SCHEMA,
    )

    def __init__(self, iou_threshold: float = 0.5, min_confidence: float = 0.7, **kwargs):
        super().__init__(iou_threshold, **kwargs)
        self.min_confidence = min_confidence

    @property
    def name(self) -> str:
        return f"high_conf_{self.min_confidence}"

    def merge(self, detections: dict, **kwargs) -> list:
        # Filter by confidence
        filtered = {}
        for model, dets in detections.items():
            filtered[model] = [d for d in dets if d.confidence >= self.min_confidence]

        # Apply majority voting
        from detection_fusion.strategies import create_strategy
        voter = create_strategy("majority_vote", iou_threshold=self.iou_threshold)
        return voter.merge(filtered)

# Usage
strategy = HighConfidenceVoting(min_confidence=0.8)
results = strategy.merge(detections)
```

### Register Custom Strategy

```python
from detection_fusion.strategies import StrategyRegistry

# Register after class definition
StrategyRegistry.register("high_conf", HighConfidenceVoting)

# Now usable via registry
strategy = StrategyRegistry.create("high_conf", min_confidence=0.9)
```

## Evaluation Examples

### Basic Evaluation

```python
from detection_fusion import evaluate_detections, merge_detections

# Merge predictions
predictions = merge_detections("labels/", strategy="weighted_vote")

# Evaluate against ground truth
evaluation = evaluate_detections(
    predictions=predictions,
    gt_path="GT/",
    iou_threshold=0.5,
)

print(f"Precision: {evaluation.precision:.3f}")
print(f"Recall: {evaluation.recall:.3f}")
print(f"F1: {evaluation.f1_score:.3f}")
```

### Pipeline with Evaluation

```python
from detection_fusion.pipeline import DetectionPipeline

ctx = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote")
    .evaluate("GT/")
    .run()
)

if ctx.evaluation_result:
    print(f"Precision: {ctx.evaluation_result.precision:.3f}")
    print(f"Recall: {ctx.evaluation_result.recall:.3f}")
    print(f"TP: {ctx.evaluation_result.true_positives}")
    print(f"FP: {ctx.evaluation_result.false_positives}")
    print(f"FN: {ctx.evaluation_result.false_negatives}")
```

---

These examples demonstrate the flexibility and power of DetectionFusion v1.0.0 across different use cases.
