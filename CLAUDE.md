# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DetectionFusion is a Python toolkit for fusing multiple object detection results with ground truth validation and error analysis. It provides 16 ensemble strategies for combining predictions from multiple detection models.

**Version:** 1.0.0
**Author:** Abhik Sarkar

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"      # Development tools
pip install -e ".[full]"     # Everything (torch, viz, CLI)

# Run tests
pytest
pytest --cov=detection_fusion --cov-report=html
pytest tests/strategies/test_registry.py

# Format and lint
black detection_fusion tests
ruff check detection_fusion tests

# CLI commands (v1.0)
detection-fusion list-strategies
detection-fusion list-strategies --category voting -v
detection-fusion merge --input labels/ --strategy weighted_vote --output unified/
detection-fusion validate --input labels/ --gt GT/ --strategy weighted_vote
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
detection-fusion convert --input annotations.xml --output labels/ --input-format voc_xml --output-format yolo
```

## Architecture (v1.0)

### Core Module (`detection_fusion/core/`)
- **Detection**: Pydantic BaseModel (frozen/immutable) representing a single detection
  - Use keyword args: `Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.3, confidence=0.9)`
  - Use `with_*` methods for modified copies: `det.with_confidence(0.8)`
- **DetectionSet**: Rich aggregate class for filtering, grouping, and statistics

### Strategies Module (`detection_fusion/strategies/`)
Strategies inherit from `BaseStrategy` and are registered via `@StrategyRegistry.register` decorator:
- **voting.py**: MajorityVoting, WeightedVoting
- **nms.py**: NMSStrategy, AffirmativeNMS
- **clustering.py**: DBSCANClustering
- **probabilistic.py**: SoftVoting, BayesianFusion
- **distance_based.py**: DistanceWeightedVoting, CentroidClustering
- **confidence_based.py**: ConfidenceThresholdVoting, ConfidenceWeightedNMS, HighConfidenceFirst
- **adaptive.py**: AdaptiveThresholdStrategy, DensityAdaptiveStrategy, MultiScaleStrategy, ConsensusRankingStrategy

**Creating strategies:**
```python
from detection_fusion.strategies import create_strategy, StrategyRegistry

strategy = create_strategy("weighted_vote", iou_threshold=0.5)
print(StrategyRegistry.list_all())  # List all 16 strategies
```

### Config Module (`detection_fusion/config/`)
- **StrategyConfig**: Pydantic model for ensemble configuration
- **RectificationConfig**: Pydantic model for GT rectification
- **ConfigLoader**: Load configs from YAML files

### Pipeline Module (`detection_fusion/pipeline/`)
Fluent interface for chaining operations:
```python
from detection_fusion.pipeline import DetectionPipeline

ctx = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .evaluate("GT/")
    .run()
)
print(f"mAP: {ctx.evaluation_result.mAP:.3f}")
```

### CLI Module (`detection_fusion/cli/`)
Click-based CLI with Rich output. Entry points: `detection-fusion` / `dfusion`

### Data Formats (`detection_fusion/data/formats/`)
- **FormatRegistry**: Extensible format reader/writer system
- Supports: YOLO (.txt), VOC XML (.xml), COCO JSON (.json)

## Detection File Format

YOLO format with normalized coordinates:
```
class_id x_center y_center width height confidence
0 0.5 0.3 0.2 0.4 0.85
```

## Data Structure Convention

```
labels/
├── model1/
│   └── *.txt (one file per image)
├── model2/
│   └── *.txt
└── GT/  # Ground truth directory
    └── *.txt
```

## Adding New Strategies

1. Create strategy class with `@StrategyRegistry.register` decorator:
```python
from detection_fusion.strategies.base import BaseStrategy, StrategyMetadata
from detection_fusion.strategies.registry import StrategyRegistry

@StrategyRegistry.register("my_strategy")
class MyStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="my_strategy",
        category="custom",
        description="My custom strategy"
    )

    def merge(self, detections, **kwargs):
        # Implementation
        pass

    @property
    def name(self):
        return "my_strategy"
```

2. Import the module in `detection_fusion/strategies/__init__.py` to trigger registration
