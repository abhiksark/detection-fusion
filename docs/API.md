# API Documentation

Complete API reference for the DetectionFusion package v1.0.0.

**Author:** Abhik Sarkar

## Core Classes

### `Detection`

**Location**: `detection_fusion.core.detection`

Immutable Pydantic model representing a single object detection.

#### Constructor
```python
Detection(
    class_id: int,           # Class identifier (>= 0)
    x: float,                # Center x coordinate (0-1)
    y: float,                # Center y coordinate (0-1)
    w: float,                # Width (0-1)
    h: float,                # Height (0-1)
    confidence: float = 1.0, # Confidence score (0-1)
    model_source: str = "",  # Source model name
    image_name: str = "",    # Image filename
)
```

#### Properties
```python
detection.bbox      # [x, y, w, h] - center format
detection.xyxy      # [x1, y1, x2, y2] - corner format
detection.center    # (x, y) tuple
detection.area      # w * h
```

#### Methods
```python
# Create modified copies (immutable)
det2 = detection.with_confidence(0.8)
det3 = detection.with_source("yolov8n")
det4 = detection.with_image("image_001")

# IoU calculation
iou = detection.iou_with(other_detection)

# Serialization
data = detection.to_dict()
det = Detection.from_dict(data)
```

#### Example
```python
from detection_fusion import Detection

det = Detection(
    class_id=0,
    x=0.5, y=0.5,
    w=0.2, h=0.3,
    confidence=0.95,
    model_source="yolov8n"
)

print(det.bbox)   # [0.5, 0.5, 0.2, 0.3]
print(det.xyxy)   # [0.4, 0.35, 0.6, 0.65]
print(det.area)   # 0.06

# Immutable - creates new instance
high_conf = det.with_confidence(0.99)
```

---

### `DetectionSet`

**Location**: `detection_fusion.core.detection_set`

Rich aggregate class for working with collections of detections.

#### Constructor
```python
DetectionSet(detections: Dict[str, List[Detection]])
```

#### Properties
```python
ds.model_names    # List of model names
ds.image_names    # List of unique image names
ds.total_count    # Total number of detections
ds.raw_data       # Original dict
```

#### Methods

##### Filtering
```python
# Filter by confidence threshold
high_conf = ds.filter_by_confidence(min_conf=0.8)

# Filter by class IDs
persons = ds.filter_by_class([0])  # class_id 0
vehicles = ds.filter_by_class([1, 2, 3])

# Filter by model
model1_dets = ds.by_model("yolov8n")

# Filter by image
img1_dets = ds.by_image("image_001")
```

##### Grouping
```python
# Group by image
by_image = ds.group_by_image()  # Dict[str, DetectionSet]

# Group by class
by_class = ds.group_by_class()  # Dict[int, DetectionSet]
```

##### Statistics
```python
# Confidence statistics
stats = ds.confidence_stats()
# Returns: {"mean": 0.85, "std": 0.1, "min": 0.5, "max": 0.99}

# Class distribution
dist = ds.class_distribution()
# Returns: {0: 150, 1: 80, 2: 45}
```

##### Iteration
```python
# Get all detections as flat list
all_dets = ds.all_detections()

# Iterate directly
for detection in ds:
    print(detection.class_id)

# Length
print(len(ds))  # Total count
```

#### Example
```python
from detection_fusion import DetectionSet

detections = {
    "yolov8n": [det1, det2, det3],
    "yolov8s": [det4, det5],
    "yolov8m": [det6, det7, det8],
}

ds = DetectionSet(detections)
print(f"Models: {ds.model_names}")       # ['yolov8m', 'yolov8n', 'yolov8s']
print(f"Total: {ds.total_count}")        # 8

# Chain filtering
result = (
    ds.filter_by_confidence(0.8)
      .filter_by_class([0, 1])
)
```

---

## Strategy Classes

### `BaseStrategy`

**Location**: `detection_fusion.strategies.base`

Abstract base class for all ensemble strategies.

#### Class Attributes
```python
class MyStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="my_strategy",
        category="custom",
        description="My custom strategy",
        params_schema=MY_PARAM_SCHEMA,  # Optional
    )
```

#### Constructor
```python
BaseStrategy(
    iou_threshold: float = 0.5,
    config: Optional[StrategyConfig] = None
)
```

#### Abstract Methods
```python
@abstractmethod
def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
    """Merge detections from multiple models."""
    pass

@property
@abstractmethod
def name(self) -> str:
    """Return strategy name."""
    pass
```

#### Methods
```python
# Validate parameters against schema
validated = strategy.validate_params(iou_threshold=0.5)

# Get schema (class method)
schema = MyStrategy.get_params_schema()

# Get defaults (class method)
defaults = MyStrategy.get_param_defaults()
```

---

### `StrategyRegistry`

**Location**: `detection_fusion.strategies.registry`

Central registry for strategy discovery and instantiation.

#### Methods
```python
from detection_fusion.strategies import StrategyRegistry

# List all strategies
names = StrategyRegistry.list_all()
# ['adaptive_threshold', 'affirmative_nms', 'bayesian', ...]

# List by category
voting = StrategyRegistry.list_by_category("voting")
# ['majority_vote', 'weighted_vote']

# Create strategy instance
strategy = StrategyRegistry.create("weighted_vote", iou_threshold=0.6)

# Get strategy class
cls = StrategyRegistry.get_class("weighted_vote")

# Check if registered
exists = StrategyRegistry.is_registered("weighted_vote")  # True

# Get count
count = StrategyRegistry.count()  # 16

# Get metadata
metadata = StrategyRegistry.get_metadata("weighted_vote")
# StrategyMetadata(name='weighted_vote', category='voting', ...)

# Get full info
info = StrategyRegistry.get_info("weighted_vote")
# StrategyInfo(name, category, description, parameters, ...)

# List with metadata
all_info = StrategyRegistry.list_with_metadata()
# [{"name": "...", "category": "...", "description": "..."}, ...]
```

#### Convenience Functions
```python
from detection_fusion.strategies import create_strategy, list_strategies

strategy = create_strategy("nms", iou_threshold=0.5)
names = list_strategies()
```

---

### Available Strategies

| Name | Category | Description |
|------|----------|-------------|
| `majority_vote` | voting | Keep detections where multiple models agree |
| `weighted_vote` | voting | Weight by model and detection confidence |
| `nms` | nms | Standard Non-Maximum Suppression |
| `affirmative_nms` | nms | NMS requiring multi-model agreement |
| `dbscan` | clustering | Density-based spatial clustering |
| `soft_voting` | probabilistic | Probabilistic voting with temperature |
| `bayesian` | probabilistic | Bayesian fusion with class priors |
| `distance_weighted` | distance_based | Weight by distance to centroid |
| `centroid_clustering` | distance_based | Cluster by detection centers |
| `confidence_threshold` | confidence_based | Adaptive thresholds per model |
| `confidence_weighted_nms` | confidence_based | NMS with confidence-weighted boxes |
| `high_confidence_first` | confidence_based | Hierarchical confidence processing |
| `adaptive_threshold` | adaptive | Different IoU for object sizes |
| `density_adaptive` | adaptive | Context-aware for density |
| `multi_scale` | adaptive | Scale-specific processing |
| `consensus_ranking` | adaptive | Model ranking with confidence |

---

## Configuration

### `StrategyConfig`

**Location**: `detection_fusion.config.models`

Pydantic model for strategy configuration with builder pattern.

```python
from detection_fusion.config import StrategyConfig

# Default config
config = StrategyConfig()

# Builder pattern
config = (
    StrategyConfig()
    .with_overlap(threshold=0.6, method="iou")
    .with_voting(min_votes=3, use_weights=True)
    .with_confidence(min_threshold=0.3, temperature=1.0)
    .with_extra(custom_param=42)
)

# Access nested configs
print(config.overlap.threshold)   # 0.6
print(config.voting.min_votes)    # 3
```

### Nested Config Classes

```python
from detection_fusion.config import (
    OverlapConfig,
    VotingConfig,
    ConfidenceConfig,
)

overlap = OverlapConfig(threshold=0.5, method="iou")
voting = VotingConfig(min_votes=2, use_weights=True)
confidence = ConfidenceConfig(min_threshold=0.1, temperature=1.0)
```

---

### `RectificationConfig`

**Location**: `detection_fusion.config.models`

Pydantic model for GT rectification configuration with builder pattern.

```python
from detection_fusion.config import RectificationConfig

# Default config
config = RectificationConfig()

# Builder pattern
config = (
    RectificationConfig()
    .with_paths(labels_dir="my_labels", output_dir="my_output")
    .with_thresholds(iou=0.6, min_agreement=4)
    .with_output(most_correct=100, copy_images=False)
)

# Access nested configs
print(config.mode)                    # "minimize_error"
print(config.paths.labels_dir)        # "my_labels"
print(config.thresholds.iou)          # 0.6
print(config.output.most_correct)     # 100
```

### Rectification Nested Config Classes

```python
from detection_fusion.config import (
    RectificationPathsConfig,
    RectificationThresholdsConfig,
    RectificationOutputConfig,
)

paths = RectificationPathsConfig(
    labels_dir="labels",
    gt_dir="GT",
    images_dir="images",
    output_dir="rectified_dataset"
)

thresholds = RectificationThresholdsConfig(
    iou=0.5,
    confidence=0.5,
    min_agreement=3
)

output = RectificationOutputConfig(
    most_correct=50,
    most_incorrect=50,
    copy_images=True
)
```

### `ConfigLoader`

**Location**: `detection_fusion.config.loader`

Load configuration from YAML files.

```python
from detection_fusion.config import ConfigLoader
from pathlib import Path

# Load StrategyConfig from YAML
config = ConfigLoader.from_yaml(Path("configs/ensemble/default.yaml"))

# Load RectificationConfig from YAML
rect_config = ConfigLoader.load_rectification(Path("configs/gt_rectification/balanced.yaml"))

# Load from dict
config = ConfigLoader.from_dict({"overlap": {"threshold": 0.6}})
rect_config = ConfigLoader.rectification_from_dict({"mode": "maximize_error"})
```

---

## Parameter Validation

### `ParamSchema`

**Location**: `detection_fusion.strategies.params`

Schema for validating strategy parameters at runtime.

```python
from detection_fusion.strategies.params import ParamSchema, ParamSpec

schema = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for overlap",
        ),
        ParamSpec(
            name="min_votes",
            param_type="int",
            default=2,
            min_value=1,
            description="Minimum votes required",
        ),
    ]
)

# Validate parameters
validated = schema.validate({"iou_threshold": 0.6, "min_votes": 3})

# Get defaults
defaults = schema.get_defaults()
# {"iou_threshold": 0.5, "min_votes": 2}
```

### Predefined Schemas
```python
from detection_fusion.strategies.params import (
    VOTING_SCHEMA,
    NMS_SCHEMA,
    CLUSTERING_SCHEMA,
    PROBABILISTIC_SCHEMA,
    ADAPTIVE_SCHEMA,
    get_schema_for_category,
)

schema = get_schema_for_category("voting")
```

---

## Pipeline API

### `DetectionPipeline`

**Location**: `detection_fusion.pipeline`

Fluent interface for chaining detection operations.

```python
from detection_fusion.pipeline import DetectionPipeline

# Build and run pipeline
ctx = (
    DetectionPipeline()
    .with_config(config)
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .evaluate("GT/")
    .run()
)

# Access results
print(ctx.detections)           # DetectionSet
print(ctx.ensemble_result)      # List[Detection]
print(ctx.evaluation_result)    # EvaluationResult
print(ctx.metadata)             # Dict
```

### `PipelineContext`

**Location**: `detection_fusion.pipeline.context`

Container for pipeline state and results.

```python
from detection_fusion.pipeline.context import PipelineContext, EvaluationResult

ctx = PipelineContext()
ctx.set_detections(detection_set)
ctx.set_ensemble_result(merged_detections)
ctx.set_evaluation_result(eval_result)
ctx.metadata["strategy"] = "weighted_vote"
```

---

## Format System

### `FormatRegistry`

**Location**: `detection_fusion.data.formats.registry`

Registry for annotation format readers and writers.

```python
from detection_fusion.data.formats import FormatRegistry

# List formats
formats = FormatRegistry.list_formats()
# {"readers": ["yolo", "voc_xml", "coco"], "writers": ["yolo", "voc_xml", "coco"]}

# Get reader/writer
reader = FormatRegistry.get_reader("yolo")
writer = FormatRegistry.get_writer("yolo")

# Auto-detect format
reader = FormatRegistry.auto_detect_reader(path)
```

### Reader/Writer Protocol

```python
from detection_fusion.data.formats.base import AnnotationReader, AnnotationWriter

class MyReader(AnnotationReader):
    format_name = "my_format"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check if this reader can handle the path."""
        pass

    def read_file(self, path: Path) -> List[Detection]:
        """Read detections from a single file."""
        pass

    def read_directory(self, path: Path) -> Dict[str, List[Detection]]:
        """Read all detections from a directory."""
        pass
```

### Register Custom Format

```python
@FormatRegistry.register_reader("my_format")
class MyFormatReader(AnnotationReader):
    ...

@FormatRegistry.register_writer("my_format")
class MyFormatWriter(AnnotationWriter):
    ...
```

---

## CLI Commands

### Available Commands

```bash
detection-fusion --help
detection-fusion --version

# Strategy commands
detection-fusion list-strategies
detection-fusion list-strategies --category voting
detection-fusion list-strategies -v  # verbose

# Format commands
detection-fusion list-formats

# Merge command
detection-fusion merge --input <path> --strategy <name> --output <path>
detection-fusion merge -i labels/ -s weighted_vote -o unified/ --iou 0.5

# Validate command
detection-fusion validate --input <path> --gt <path> --strategy <name>

# Convert command
detection-fusion convert --input <path> --output <path> \
    --input-format <format> --output-format <format>

# Rectify command
detection-fusion rectify --labels-dir <path> --gt-dir <path> \
    --images-dir <path> --output <path> \
    --mode minimize_error

# Rectify with config file (CLI options override config values)
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
detection-fusion rectify -c configs/gt_rectification/conservative.yaml \
    --output custom_output/
```

---

## Convenience Functions

**Location**: `detection_fusion`

```python
from detection_fusion import (
    merge_detections,
    evaluate_detections,
    convert_annotations,
    list_strategies,
    list_formats,
)

# Merge detections
results = merge_detections(
    path="labels/",
    strategy="weighted_vote",
    iou_threshold=0.5,
    format="auto",
)

# Evaluate against ground truth
evaluation = evaluate_detections(
    predictions=results,
    gt_path="GT/",
    iou_threshold=0.5,
)

# Convert formats
convert_annotations(
    input_path="annotations.xml",
    output_path="labels/",
    input_format="voc_xml",
    output_format="yolo",
)

# List available options
strategies = list_strategies()
formats = list_formats()
```

---

## Exception Handling

**Location**: `detection_fusion.exceptions`

```python
from detection_fusion.exceptions import (
    DetectionFusionError,  # Base exception
    ConfigurationError,    # Config issues
    FormatError,           # Format read/write issues
)

try:
    reader = FormatRegistry.get_reader("unknown")
except FormatError as e:
    print(f"Format error: {e}")
```

---

## Type Hints

The package uses comprehensive type hints:

```python
from typing import List, Dict, Optional
from detection_fusion import Detection, DetectionSet
from detection_fusion.strategies import BaseStrategy

def my_function(
    detections: Dict[str, List[Detection]],
    strategy: BaseStrategy,
) -> List[Detection]:
    return strategy.merge(detections)
```

---

## Performance Tips

### Memory
- Use `DetectionSet.filter_by_confidence()` early to reduce data
- Process images in batches for large datasets

### Speed
- NMS strategies are fastest
- Clustering strategies scale with O(n²)
- Use format auto-detection only when needed

### Thread Safety
The package is not thread-safe. Create separate instances per thread:

```python
import threading

def worker():
    strategy = create_strategy("weighted_vote")
    # ... use strategy
```
