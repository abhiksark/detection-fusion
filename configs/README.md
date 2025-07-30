# DetectionFusion Configuration Files

This directory contains organized configuration files for different aspects of the DetectionFusion toolkit.

## 📁 Directory Structure

```
configs/
├── README.md                          # This file
├── ensemble/                          # Ensemble and analysis configurations
│   ├── default_config.yaml           # Default ensemble configuration
│   ├── advanced_strategies_config.yaml # All 19 strategies showcase
│   ├── high_precision_config.yaml    # Conservative strategies for critical applications
│   ├── high_recall_config.yaml       # Permissive strategies for maximum detection
│   └── small_objects_config.yaml     # Optimized for small object detection
└── gt_rectification/                  # Ground truth rectification configurations
    ├── gt_rectify_conservative.yaml  # High-precision error detection
    ├── gt_rectify_aggressive.yaml    # Comprehensive error detection
    ├── gt_rectify_balanced.yaml      # Balanced precision/recall approach
    └── gt_rectify_custom.yaml        # Template for custom configurations
```

## 🎯 Ensemble Configurations

Located in `configs/ensemble/` - Used with `merge.py`, `val.py`, and Python API.

### `default_config.yaml`
- **Purpose**: Standard ensemble configuration with commonly used strategies
- **Best for**: General-purpose ensemble workflows
- **Usage**: 
  ```bash
  python merge.py --config configs/ensemble/default_config.yaml
  ```

### `advanced_strategies_config.yaml`
- **Purpose**: Demonstrates all 19 available ensemble strategies
- **Best for**: Strategy comparison and performance analysis
- **Usage**:
  ```bash
  python val.py --config configs/ensemble/advanced_strategies_config.yaml --benchmark-strategies
  ```

### `high_precision_config.yaml`
- **Purpose**: Conservative strategies optimized for precision
- **Best for**: Critical applications where false positives are costly
- **Strategies**: Unanimous voting, high-confidence strategies, strict thresholds
- **Usage**:
  ```bash
  python merge.py --config configs/ensemble/high_precision_config.yaml
  ```

### `high_recall_config.yaml`
- **Purpose**: Permissive strategies optimized for recall
- **Best for**: Applications where missing detections is costly
- **Strategies**: Low-threshold voting, inclusive strategies
- **Usage**:
  ```bash
  python merge.py --config configs/ensemble/high_recall_config.yaml
  ```

### `small_objects_config.yaml`
- **Purpose**: Optimized for small object detection scenarios
- **Best for**: Small object datasets (COCO, traffic signs, etc.)
- **Features**: Adaptive thresholds, multi-scale strategies
- **Usage**:
  ```bash
  python merge.py --config configs/ensemble/small_objects_config.yaml
  ```

## 🔍 GT Rectification Configurations

Located in `configs/gt_rectification/` - Used with `gt_rectify.py`.

### `gt_rectify_conservative.yaml`
- **Purpose**: High-precision ground truth error detection
- **Best for**: Critical datasets where annotation quality is paramount
- **Approach**: Conservative thresholds, requires high consensus
- **Usage**:
  ```bash
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml
  ```

### `gt_rectify_aggressive.yaml`
- **Purpose**: Comprehensive error detection with maximum coverage
- **Best for**: Initial dataset quality assessment and bulk error detection
- **Approach**: Permissive thresholds, uses all available strategies
- **Usage**:
  ```bash
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml
  ```

### `gt_rectify_balanced.yaml`
- **Purpose**: Balanced approach between precision and recall
- **Best for**: General-purpose GT quality assessment
- **Approach**: Moderate thresholds with proven strategies
- **Usage**:
  ```bash
  python gt_rectify.py --config configs/gt_rectification/gt_rectify_balanced.yaml
  ```

### `gt_rectify_custom.yaml`
- **Purpose**: Template for creating custom rectification configurations
- **Best for**: Specialized datasets or specific requirements
- **Features**: Detailed comments, parameter explanations, usage instructions
- **Usage**:
  ```bash
  # Copy and customize
  cp configs/gt_rectification/gt_rectify_custom.yaml my_custom_config.yaml
  # Edit my_custom_config.yaml with your parameters
  python gt_rectify.py --config my_custom_config.yaml
  ```

## 🚀 Quick Start Examples

### Ensemble Workflows
```bash
# Basic ensemble with default settings
python merge.py --config configs/ensemble/default_config.yaml

# High-precision ensemble for critical applications
python merge.py --config configs/ensemble/high_precision_config.yaml

# Comprehensive strategy benchmarking
python val.py --config configs/ensemble/advanced_strategies_config.yaml --benchmark-strategies
```

### GT Rectification Workflows
```bash
# Conservative error detection (recommended for first-time use)
python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml

# Comprehensive error detection for dataset audit
python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml

# Balanced approach for routine quality checks
python gt_rectify.py --config configs/gt_rectification/gt_rectify_balanced.yaml
```

## 🛠️ Customization Tips

### 1. Configuration Override
All configurations support command-line overrides:
```bash
# Use conservative GT config but switch to aggressive mode
python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml --mode maximize_error

# Use high-precision ensemble but lower IoU threshold
python merge.py --config configs/ensemble/high_precision_config.yaml --iou-threshold 0.4
```

### 2. Creating Custom Configurations
1. **Copy an existing config** that's closest to your needs
2. **Modify parameters** based on your dataset and requirements
3. **Test with a small subset** of your data first
4. **Iterate and refine** based on results

### 3. Configuration Validation
- **Ensemble configs**: Test with `val.py --config your_config.yaml --analyze agreement`
- **GT rectification configs**: Test with a small dataset first using `--most-correct 10 --most-incorrect 10`

## 📊 Configuration Selection Guide

| Use Case | Recommended Ensemble Config | Recommended GT Rectification Config |
|----------|----------------------------|-------------------------------------|
| **Critical Applications** | `high_precision_config.yaml` | `gt_rectify_conservative.yaml` |
| **Research/Exploration** | `advanced_strategies_config.yaml` | `gt_rectify_aggressive.yaml` |
| **Production Workflows** | `default_config.yaml` | `gt_rectify_balanced.yaml` |
| **Small Objects** | `small_objects_config.yaml` | `gt_rectify_balanced.yaml` |
| **Maximum Coverage** | `high_recall_config.yaml` | `gt_rectify_aggressive.yaml` |
| **Custom Requirements** | Copy and modify existing | `gt_rectify_custom.yaml` template |

## 🔗 Related Documentation

- [Strategy Guide](../docs/STRATEGY_GUIDE.md) - Detailed explanation of all ensemble strategies
- [API Documentation](../docs/API.md) - Python API usage with configurations
- [Examples](../examples/) - Complete usage examples with configurations
- [Troubleshooting](../docs/TROUBLESHOOTING.md) - Common configuration issues and solutions

---

💡 **Tip**: Start with the recommended configurations for your use case, then customize as needed based on your specific requirements and dataset characteristics.