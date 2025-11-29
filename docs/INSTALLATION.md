# Installation Guide

Complete installation instructions for the DetectionFusion package v1.0.

**Author:** Abhik Sarkar

## Quick Installation

### Requirements
- Python 3.8 or higher
- pip (Python package installer)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/abhiksark/detection-fusion.git
cd detection-fusion

# Install in development mode
pip install -e .

# Or with CLI support
pip install -e ".[cli]"

# Or with everything
pip install -e ".[full]"
```

### Verify Installation
```bash
# Test Python import
python -c "from detection_fusion import Detection, merge_detections; print('Installation successful!')"

# Test CLI (requires [cli] extra)
detection-fusion --help
detection-fusion list-strategies
```

## Installation Options

### 1. Minimal Installation (Core only)
```bash
pip install -e .
```

**Includes**: Core functionality (Detection, strategies, pipeline, config)

### 2. CLI Installation
```bash
pip install -e ".[cli]"
```

**Includes**: Core + Click CLI with Rich output

### 3. Visualization Installation
```bash
pip install -e ".[viz]"
```

**Includes**: Core + matplotlib, seaborn for plots

### 4. Full Installation
```bash
pip install -e ".[full]"
```

**Includes**: Everything (torch, viz, CLI)

### 5. Development Installation
```bash
pip install -e ".[dev]"
```

**Includes**: Full + pytest, black, ruff, mypy

## Environment Setup

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install package
pip install -e ".[full]"
```

### Using Conda
```bash
# Create new environment
conda create -n detection-fusion python=3.10
conda activate detection-fusion

# Install PyTorch (optional, for torch-based operations)
conda install pytorch torchvision -c pytorch

# Install detection-fusion
pip install -e ".[cli,viz]"
```

## Dependencies

### Core Dependencies (Always Installed)
```
numpy>=1.19.0        # Numerical computations
pandas>=1.1.0        # Data manipulation
pydantic>=2.0.0      # Data validation (v1.0 requirement)
scikit-learn>=0.23.0 # Machine learning algorithms
PyYAML>=5.4.0        # Configuration loading
tqdm>=4.60.0         # Progress bars
```

### Optional Dependencies
```
# [torch] extra
torch>=1.7.0         # Deep learning framework
torchvision>=0.8.0   # Computer vision utilities

# [viz] extra
matplotlib>=3.3.0    # Basic plotting
seaborn>=0.11.0      # Statistical visualization

# [cli] extra
click>=8.0.0         # CLI framework
rich>=12.0.0         # Rich terminal output
```

## CLI Commands

After installation with `[cli]` extra:

```bash
# List available strategies
detection-fusion list-strategies
detection-fusion list-strategies --category voting -v

# List supported formats
detection-fusion list-formats

# Merge detections from multiple models
detection-fusion merge --input labels/ --strategy weighted_vote --output unified/

# Validate against ground truth
detection-fusion validate --input labels/ --gt GT/ --strategy weighted_vote

# Convert annotation formats
detection-fusion convert --input annotations.xml --output labels/ \
    --input-format voc_xml --output-format yolo

# Rectify ground truth annotations
detection-fusion rectify --labels-dir labels/ --gt-dir GT/ \
    --images-dir images/ --output rectified/

# Rectify with config file
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
```

## Verification Tests

### Basic Functionality Test
```python
from detection_fusion import Detection, merge_detections
from detection_fusion.strategies import create_strategy

# Create test detections (v1.0 uses keyword arguments)
det1 = Detection(
    class_id=0, x=0.5, y=0.5, w=0.2, h=0.3,
    confidence=0.8, model_source="model1"
)
det2 = Detection(
    class_id=0, x=0.52, y=0.48, w=0.18, h=0.32,
    confidence=0.85, model_source="model2"
)

# Test strategy creation
strategy = create_strategy("majority_vote", iou_threshold=0.5)
print(f"Strategy: {strategy.name}")

# Test merge
detections = {"model1": [det1], "model2": [det2]}
results = strategy.merge(detections)
print(f"Merged {len(results)} detections")
```

### Pipeline API Test
```python
from detection_fusion.pipeline import DetectionPipeline

# Build and run pipeline (if you have labels directory)
ctx = (
    DetectionPipeline()
    .load("labels/", format="yolo")
    .ensemble("weighted_vote", iou_threshold=0.5)
    .run()
)
print(f"Merged: {len(ctx.ensemble_result)} detections")
```

## Troubleshooting

### Common Issues

#### 1. Pydantic Import Error
```bash
# Solution: Ensure pydantic v2 is installed
pip install "pydantic>=2.0.0"
```

#### 2. CLI Command Not Found
```bash
# Solution: Install with CLI extra
pip install -e ".[cli]"

# Or reinstall
pip uninstall detection-fusion
pip install -e ".[cli]"
```

#### 3. Import Error: torch not found
```bash
# Solution: torch is optional in v1.0
# Install if needed:
pip install torch torchvision

# Or install full extras
pip install -e ".[full]"
```

#### 4. Permission Error during installation
```bash
# Solution: Use user installation
pip install --user -e .
```

### Debug Installation
```python
def debug_installation():
    """Run diagnostic checks."""
    import sys
    print(f"Python version: {sys.version}")

    try:
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
    except ImportError:
        print("Pydantic not installed (required)")

    try:
        import detection_fusion
        print(f"DetectionFusion version: {detection_fusion.__version__}")
        print("Installation successful")
    except ImportError as e:
        print(f"Installation failed: {e}")

debug_installation()
```

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2GHz
- **RAM**: 4GB
- **Storage**: 500MB free space
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 4+ cores, 3GHz+
- **RAM**: 8GB+
- **Storage**: 2GB+ free space

## Updating

### Update Package
```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e ".[full]"
```

### Migration from v0.2.x
When updating from v0.2.x to v1.0:

1. **Detection constructor changed** - Use keyword arguments:
   ```python
   # Old (v0.2.x)
   det = Detection(0, 0.5, 0.5, 0.2, 0.3, 0.8, "model1")

   # New (v1.0)
   det = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.3,
                   confidence=0.8, model_source="model1")
   ```

2. **Strategy creation changed** - Use registry:
   ```python
   # Old (v0.2.x)
   from detection_fusion.strategies import MajorityVoting
   strategy = MajorityVoting(iou_threshold=0.5)

   # New (v1.0)
   from detection_fusion.strategies import create_strategy
   strategy = create_strategy("majority_vote", iou_threshold=0.5)
   ```

3. **CLI changed** - Use `detection-fusion` command:
   ```bash
   # Old (v0.2.x)
   python merge.py --labels-dir labels/

   # New (v1.0)
   detection-fusion merge --input labels/ --strategy weighted_vote
   ```

See [CHANGELOG.md](../CHANGELOG.md) for full migration details.

## Getting Help

If you encounter issues:

1. **Check documentation**: Review this guide and [API.md](API.md)
2. **Search issues**: Look through [GitHub Issues](https://github.com/abhiksark/detection-fusion/issues)
3. **Ask questions**: Start a [GitHub Discussion](https://github.com/abhiksark/detection-fusion/discussions)
4. **Report bugs**: Create a new [GitHub Issue](https://github.com/abhiksark/detection-fusion/issues/new)
