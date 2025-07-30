# API Documentation

Complete API reference for the DetectionFusion package.

## Core Classes

### `Detection`

**Location**: `detection_fusion.core.detection`

Represents a single object detection with bounding box and metadata.

#### Constructor
```python
Detection(class_id: int, x: float, y: float, w: float, h: float, 
          confidence: float, model_source: str = "")
```

#### Parameters
- `class_id`: Object class identifier
- `x`, `y`: Center coordinates (normalized 0-1)
- `w`, `h`: Width and height (normalized 0-1)
- `confidence`: Detection confidence score (0-1)
- `model_source`: Name of the model that produced this detection
- `image_name`: Image filename this detection belongs to (NEW in v0.2.0)

#### Properties
```python
detection.bbox          # [x, y, w, h] format
detection.xyxy          # [x1, y1, x2, y2] format  
detection.center        # (x, y) center point
detection.area          # Bounding box area
```

#### Methods (Enhanced in v0.2.0)
```python
detection.to_dict()     # Convert to dictionary
Detection.from_dict(data)  # Create from dictionary

# NEW: Hashable support - can be used in sets and dictionaries
detection_set = {detection1, detection2}
detection_dict = {detection: metadata}
```

### `EnsembleVoting`

**Location**: `detection_fusion.core.ensemble`

Main class for basic ensemble voting operations.

#### Constructor
```python
EnsembleVoting(labels_dir: str = "labels", output_dir: str = "labels/unified")
```

#### Parameters
- `labels_dir`: Directory containing model subdirectories
- `output_dir`: Directory for saving ensemble results

#### Methods

##### `load_detections(filename: str = "detections.txt") -> Dict[str, List[Detection]]`
Load detections from all model directories.

**Returns**: Dictionary mapping model names to their detections

##### `run_strategy(strategy_name: str, **kwargs) -> List[Detection]`
Run a specific ensemble strategy.

**Parameters**: 
- `strategy_name`: Name of strategy to run
- `**kwargs`: Additional strategy parameters

**Returns**: List of merged detections

##### `run_all_strategies(filename: str = "detections.txt", save_results: bool = True) -> Dict[str, List[Detection]]`
Run all registered strategies.

**Returns**: Dictionary mapping strategy names to results

##### `add_strategy(name: str, strategy) -> None`
Add a custom strategy to the ensemble.

##### `save_statistics(results: Dict, filename: str = "ensemble_stats.json") -> None`
Save ensemble statistics to JSON file.

### `AdvancedEnsemble`

**Location**: `detection_fusion.core.ensemble`

Extended ensemble class with advanced strategies. Inherits from `EnsembleVoting`.

#### Additional Methods

##### `set_strategy_params(strategy_name: str, **params) -> None`
Update parameters for a specific strategy.

**Example**:
```python
ensemble.set_strategy_params("dbscan", eps=0.15, min_samples=3)
```

### `MultiModelAnalyzer`

**Location**: `detection_fusion.core.analyzer`

Comprehensive analysis toolkit for comparing model performance.

#### Constructor
```python
MultiModelAnalyzer(labels_dir: str = "labels", iou_threshold: float = 0.5)
```

#### Methods

##### `load_detections(filename: str = "detections.txt") -> Dict[str, List[Detection]]`
Load detections from all model directories.

##### `compare_models(model1: str, model2: str) -> Dict[str, Any]`
Compare detections between two specific models.

**Returns**: Dictionary with comparison metrics:
- `total_matches`: Number of matched detections
- `model1_unique`: Detections only in model1
- `model2_unique`: Detections only in model2
- `avg_iou`: Average IoU of matches
- `class_matches`: Per-class match counts

##### `compare_all_models() -> pd.DataFrame`
Generate comparison matrix for all model pairs.

**Returns**: DataFrame with pairwise comparison results

##### `get_class_statistics() -> pd.DataFrame`
Calculate detection statistics per class across all models.

**Returns**: DataFrame with columns:
- Model columns: Detection counts per model
- `class_name`: Human-readable class name
- `total`: Total detections across all models
- `mean`: Average detections per model
- `std`: Standard deviation
- `variance`: Variance across models

##### `get_confidence_statistics() -> pd.DataFrame`
Calculate confidence score statistics for each model.

**Returns**: DataFrame with statistical measures per model

##### `find_consensus_detections(min_models: int = 2) -> List[Detection]`
Find detections that multiple models agree on.

##### `generate_report(output_file: str = "analysis_report.txt") -> None`
Generate comprehensive analysis report.

## Strategy Classes

All strategy classes inherit from `BaseStrategy` and implement the `merge` method.

### `BaseStrategy`

**Location**: `detection_fusion.strategies.base`

Abstract base class for all ensemble strategies.

#### Abstract Methods
```python
def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]
    """Merge detections from multiple models."""
    pass

@property
def name(self) -> str:
    """Return strategy name."""
    pass
```

### `MajorityVoting`

**Location**: `detection_fusion.strategies.voting`

Requires minimum number of models to agree on a detection.

#### Constructor
```python
MajorityVoting(iou_threshold: float = 0.5, min_votes: int = 2)
```

#### Parameters
- `iou_threshold`: IoU threshold for matching detections
- `min_votes`: Minimum number of models required to agree

### `WeightedVoting`

**Location**: `detection_fusion.strategies.voting`

Weighted voting using confidence scores and optional model weights.

#### Constructor
```python
WeightedVoting(iou_threshold: float = 0.5, use_model_weights: bool = True)
```

#### Parameters
- `use_model_weights`: Whether to weight by model performance

### `NMSStrategy`

**Location**: `detection_fusion.strategies.nms`

Standard Non-Maximum Suppression across all models.

#### Constructor
```python
NMSStrategy(iou_threshold: float = 0.5, score_threshold: float = 0.1)
```

#### Parameters
- `score_threshold`: Minimum confidence score to consider

### `AffirmativeNMS`

**Location**: `detection_fusion.strategies.nms`

NMS requiring agreement from multiple models.

#### Constructor
```python
AffirmativeNMS(iou_threshold: float = 0.5, min_models: int = 2)
```

#### Parameters
- `min_models`: Minimum number of models that must detect an object

### `DBSCANClustering`

**Location**: `detection_fusion.strategies.clustering`

Density-based spatial clustering of detections.

#### Constructor
```python
DBSCANClustering(eps: float = 0.1, min_samples: int = 2)
```

#### Parameters
- `eps`: Maximum distance between points in same cluster
- `min_samples`: Minimum samples per cluster

### `SoftVoting`

**Location**: `detection_fusion.strategies.probabilistic`

Probabilistic voting with temperature scaling.

#### Constructor
```python
SoftVoting(iou_threshold: float = 0.5, temperature: float = 1.0)
```

#### Parameters
- `temperature`: Temperature for confidence scaling (lower = more confident)

### `BayesianFusion`

**Location**: `detection_fusion.strategies.probabilistic`

Bayesian fusion with learned class priors.

#### Constructor
```python
BayesianFusion(iou_threshold: float = 0.5, 
               class_priors: Optional[Dict[int, float]] = None)
```

#### Parameters
- `class_priors`: Prior probabilities for each class (auto-calculated if None)

## Utility Functions

### I/O Functions

**Location**: `detection_fusion.utils.io`

#### `read_detections(file_path: str, model_name: str = "") -> List[Detection]`
Read detections from a text file.

#### `save_detections(detections: List[Detection], output_path: str) -> None`
Save detections to a text file.

#### `load_class_names(class_names_file: Optional[str] = None) -> Dict[int, str]`
Load class names from file or generate defaults.

#### `save_json_results(data: dict, output_path: str) -> None`
Save results as JSON.

#### `load_yaml_config(config_path: str) -> dict`
Load configuration from YAML file.

#### `save_yaml_config(data: dict, output_path: str) -> None`
Save configuration as YAML file.

#### `load_json_config(config_path: str) -> dict`
Load configuration from JSON file (deprecated, use YAML).

### Metrics Functions

**Location**: `detection_fusion.utils.metrics`

#### `calculate_iou(box1: List[float], box2: List[float]) -> float`
Calculate Intersection over Union between two bounding boxes.

#### `calculate_giou(box1: List[float], box2: List[float]) -> float`
Calculate Generalized IoU (GIoU).

#### `calculate_diou(box1: List[float], box2: List[float]) -> float`
Calculate Distance IoU (DIoU).

#### `batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor`
Calculate IoU matrix between two sets of boxes.

## Visualization Functions

**Location**: `detection_fusion.visualization.plots`

#### `plot_class_distribution(analyzer, top_n: int = 20, save_path: Optional[str] = None)`
Plot detection count distribution across classes.

#### `plot_confidence_distribution(analyzer, save_path: Optional[str] = None)`
Plot confidence score distributions for each model.

#### `plot_model_comparison_heatmap(analyzer, save_path: Optional[str] = None)`
Plot heatmap showing similarity between models.

#### `generate_all_plots(analyzer, output_dir: str = "plots", top_n: int = 20)`
Generate all visualization plots.


## Exception Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when detection files are not found.

```python
try:
    ensemble.load_detections("missing_file.txt")
except FileNotFoundError as e:
    print(f"Detection file not found: {e}")
```

#### `ValueError`
Raised for invalid strategy names or parameters.

```python
try:
    ensemble.run_strategy("invalid_strategy")
except ValueError as e:
    print(f"Invalid strategy: {e}")
```

## Configuration Schema

### Default Configuration Structure
```yaml
ensemble:
  labels_dir: "labels"
  output_dir: "labels/unified"
  iou_threshold: 0.5
  
  strategies:
    strategy_name:
      param1: "value1"
      param2: "value2"

analysis:
  iou_threshold: 0.5
  top_classes: 20
  generate_plots: true
  plot_formats: 
    - "png"
    - "pdf"

visualization:
  figure_size: [14, 8]
  dpi: 300
  style: "seaborn"
  color_palette: "Set2"
```

## Type Hints

The package uses comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Optional, Tuple, Any
from detection_fusion.core.detection import Detection

def my_function(detections: Dict[str, List[Detection]]) -> List[Detection]:
    """Function with proper type hints."""
    pass
```

## Performance Considerations

### Memory Usage
- Large datasets: Use batch processing
- Multiple strategies: Run individually to reduce memory

### Speed Optimization
- Use `batch_iou` for large IoU computations
- Enable multiprocessing for independent operations
- Cache model weights for repeated runs

## Thread Safety

The package is **not thread-safe**. For concurrent usage:

```python
import threading

# Create separate instances per thread
def worker_function():
    ensemble = EnsembleVoting("labels", f"output_{threading.current_thread().name}")
    # ... rest of processing
```