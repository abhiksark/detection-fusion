# DetectionFusion Examples

This directory contains comprehensive examples demonstrating various use cases of the DetectionFusion package.

## 🚀 Available Examples

### 1. `basic_usage.py`
**Basic ensemble voting and analysis**
- Simple ensemble voting workflows
- Multi-model analysis basics
- Perfect for getting started

```bash
python examples/basic_usage.py
```

### 2. `advanced_usage.py`
**Advanced ensemble techniques**
- Custom strategy parameters
- Advanced analysis features
- Visualization generation

```bash
python examples/advanced_usage.py
```

### 3. `advanced_strategies_demo.py`
**Complete strategy demonstration**
- All 17+ ensemble strategies
- Distance-based, confidence-based, and adaptive strategies
- Performance comparisons

```bash
python examples/advanced_strategies_demo.py
```

### 4. `config_usage.py`
**YAML configuration workflows**
- Configuration file management
- Batch processing setups
- Reproducible experiments

```bash
python examples/config_usage.py
```

### 5. `rectification_example.py`
**Ground truth rectification system**
- GT error detection with both conservative and aggressive modes
- Dataset organization for human review
- Mode comparison and analysis

```bash
python examples/rectification_example.py
```

### 6. `gt_rectify_config_example.py`
**GT rectification configuration guide**
- Demonstrates usage of YAML configuration files
- Shows different rectification modes and their use cases
- Configuration override examples

```bash
python examples/gt_rectify_config_example.py
```

## 📁 Prerequisites

Before running examples, ensure you have:

1. **Sample data structure**:
   ```
   labels/
   ├── model1/
   │   └── detections.txt
   ├── model2/
   │   └── detections.txt
   └── model3/
       └── detections.txt
   ```

2. **DetectionFusion installed**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 🎯 Example Usage Patterns

### Quick Start
```bash
# Run basic example
python examples/basic_usage.py

# Try advanced strategies
python examples/advanced_strategies_demo.py
```

### GT Rectification
```bash
# Conservative error detection
python examples/rectification_example.py

# Or use CLI directly
python gt_rectify.py --labels-dir labels --gt-dir gt --images-dir images --output-dir rectified
```

### Configuration-Based Workflows
```bash
# Use predefined configs
python examples/config_usage.py

# Or with CLI tools
python merge.py --config configs/high_precision_config.yaml
```

## 📊 Expected Outputs

Each example generates different outputs:

- **basic_usage.py**: Console output with detection counts
- **advanced_usage.py**: Plots and analysis files
- **advanced_strategies_demo.py**: Comprehensive strategy comparison
- **config_usage.py**: Configuration templates and results
- **rectification_example.py**: GT error analysis and organized datasets

## 🔧 Customization

All examples can be customized by:

1. **Modifying data paths** in the example files
2. **Adjusting parameters** (IoU thresholds, confidence levels)
3. **Adding your own strategies** or analysis steps
4. **Extending visualization** options

## 📚 Additional Resources

- [API Documentation](../docs/API.md)
- [Strategy Guide](../docs/STRATEGY_GUIDE.md)
- [Configuration Examples](../configs/)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)

## 💡 Tips

- Start with `basic_usage.py` to understand core concepts
- Use `rectification_example.py` for GT quality assessment
- Refer to `advanced_strategies_demo.py` for complete strategy overview
- Check `config_usage.py` for production workflows

Happy experimenting! 🎉