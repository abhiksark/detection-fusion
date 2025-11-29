# DetectionFusion Examples

This directory contains examples demonstrating the DetectionFusion v1.0 API.

**Author:** Abhik Sarkar

## Available Examples

### 1. `basic_usage.py`
**Core v1.0 API patterns**
- `merge_detections()` convenience function
- `Detection` class with keyword arguments
- `StrategyRegistry` and `create_strategy()`
- `DetectionPipeline` fluent interface
- Format conversion

```bash
python examples/basic_usage.py
```

### 2. `rectification_example.py`
**Ground truth rectification system**
- GT error detection with conservative and aggressive modes
- Dataset organization for human review
- Mode comparison and analysis

```bash
python examples/rectification_example.py
```

### 3. `gt_rectify_config_example.py`
**GT rectification configuration guide**
- YAML configuration file usage
- Different rectification modes and their use cases
- Configuration override examples

```bash
python examples/gt_rectify_config_example.py
```

## Prerequisites

1. **DetectionFusion installed**:
   ```bash
   pip install -e ".[full]"
   ```

2. **Sample data structure** (for merge/validate commands):
   ```
   labels/
   ├── model1/
   │   └── image1.txt
   ├── model2/
   │   └── image1.txt
   └── GT/
       └── image1.txt
   ```

## Quick Start

```python
from detection_fusion import merge_detections, Detection, StrategyRegistry

# List all 16 ensemble strategies
print(StrategyRegistry.list_all())

# Merge detections from multiple models
results = merge_detections("labels/", strategy="weighted_vote", iou_threshold=0.5)

# Work with Detection objects (keyword args required)
det = Detection(
    class_id=0,
    x=0.5, y=0.5,
    w=0.2, h=0.3,
    confidence=0.95,
    model_source="yolov8"
)

# Detection is immutable - use with_* methods for copies
det_copy = det.with_confidence(0.99)
```

## CLI Quick Reference

```bash
# List available strategies
detection-fusion list-strategies
detection-fusion list-strategies --category voting -v

# List supported formats
detection-fusion list-formats

# Merge detections
detection-fusion merge --input labels/ --strategy weighted_vote --output unified/

# Validate against ground truth
detection-fusion validate --input labels/ --gt GT/ --strategy weighted_vote

# Convert formats
detection-fusion convert --input annotations.xml --output labels/ \
    --input-format voc_xml --output-format yolo

# Rectify ground truth
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
```

## v1.0 API Patterns

### Detection Class
```python
# v1.0: Keyword arguments required, immutable (frozen Pydantic model)
det = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.3, confidence=0.9, model_source="model1")

# Create modified copies with with_* methods
det2 = det.with_confidence(0.95)
det3 = det.with_source("model2")
```

### Strategy Registry
```python
from detection_fusion.strategies import create_strategy, StrategyRegistry

# Create strategy with parameters
strategy = create_strategy("weighted_vote", iou_threshold=0.5)

# Merge detections
merged = strategy.merge({"model1": dets1, "model2": dets2})

# List all strategies
StrategyRegistry.list_all()
```

### Pipeline API
```python
from detection_fusion import DetectionPipeline

result = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .evaluate("labels/GT/")
    .run()
)

print(f"mAP: {result.evaluation_result.map_50:.3f}")
```

## Additional Resources

- [API Documentation](../docs/API.md)
- [Strategy Guide](../docs/STRATEGY_GUIDE.md)
- [Configuration Examples](../configs/)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
