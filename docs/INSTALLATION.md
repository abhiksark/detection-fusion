# Installation Guide

Complete installation instructions for the DetectionFusion package with requirements files and CLI tools.

## 🚀 Quick Installation

### Requirements
- Python 3.7 or higher
- pip (Python package installer)

### Basic Installation (Production)
```bash
# Clone the repository
git clone https://github.com/yourusername/detection-fusion.git
cd detection-fusion

# Install production dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/detection-fusion.git
cd detection-fusion

# Install development dependencies (includes all production deps)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Installation
```bash
# Test Python import
python -c "from detection_fusion import AdvancedEnsemble; print('✅ Installation successful!')"

# Test CLI tools
python merge.py --help
python val.py --help

# List available strategies
python merge.py --list-strategies
```

## 🔧 Installation Options

### 1. Development Installation
For contributors and developers:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pre-commit install
```

**Includes**:
- pytest for testing
- black for code formatting
- flake8 for linting
- pytest-cov for coverage

### 2. Visualization Installation
For advanced plotting features:

```bash
# Install with visualization extras
pip install -e ".[viz]"
```

**Includes**:
- plotly for interactive plots
- bokeh for web-based visualizations

### 3. Full Installation
Everything included:

```bash
# Install all extras
pip install -e ".[dev,viz]"
```

## 🐍 Environment Setup

### Using Conda (Recommended)
```bash
# Create new environment
conda create -n detection-fusion python=3.9
conda activate detection-fusion

# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install detection-fusion
cd detection-fusion
pip install -e .
```

### Using Virtual Environment
```bash
# Create virtual environment
python -m venv detection-fusion-env

# Activate environment
# On Linux/Mac:
source detection-fusion-env/bin/activate
# On Windows:
detection-fusion-env\Scripts\activate

# Install package
pip install -e .
```

### Using Poetry
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## 📦 Dependencies

### Core Dependencies
```
numpy>=1.19.0        # Numerical computations
pandas>=1.1.0        # Data manipulation
torch>=1.7.0         # Deep learning framework
torchvision>=0.8.0   # Computer vision utilities
scikit-learn>=0.23.0 # Machine learning algorithms
matplotlib>=3.3.0    # Basic plotting
seaborn>=0.11.0      # Statistical visualization
```

### Optional Dependencies
```
pycocotools>=2.0.2   # COCO format support
plotly>=5.0          # Interactive plots
bokeh>=2.3           # Web visualizations
pytest>=6.0          # Testing framework
black>=21.0          # Code formatting
flake8>=3.9          # Code linting
```

## 🖥️ Platform-Specific Instructions

### Ubuntu/Debian
```bash
# System dependencies
sudo apt update
sudo apt install python3-dev python3-pip

# For OpenCV (if needed)
sudo apt install libopencv-dev python3-opencv

# Install package
pip install -e .
```

### CentOS/RHEL
```bash
# System dependencies
sudo yum install python3-devel python3-pip

# Install package
pip install -e .
```

### macOS
```bash
# Using Homebrew
brew install python

# Install package
pip install -e .
```

### Windows
```cmd
# Using conda (recommended)
conda create -n detection-fusion python=3.9
conda activate detection-fusion

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install package
pip install -e .
```

## 🐳 Docker Installation

### Using Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  detection-fusion:
    build: .
    volumes:
      - ./labels:/app/labels
      - ./output:/app/output
    command: detection-fusion --labels-dir labels --output-dir output
```

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["detection-fusion"]
```

### Build and Run
```bash
# Build image
docker build -t detection-fusion .

# Run container
docker run -v $(pwd)/labels:/app/labels -v $(pwd)/output:/app/output detection-fusion
```

## ☁️ Cloud Platform Setup

### Google Colab
```python
# Install in Colab notebook
!git clone https://github.com/yourusername/detection-fusion.git
%cd detection-fusion
!pip install -e .

# Import and use
from detection_fusion import EnsembleVoting
```

### Kaggle Kernels
```python
# In Kaggle kernel
import subprocess
import sys

# Clone and install
subprocess.check_call([sys.executable, "-m", "pip", "install", 
                      "git+https://github.com/yourusername/detection-fusion.git"])
```

### AWS SageMaker
```python
# In SageMaker notebook
import subprocess
import sys

def install_detection_fusion():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/yourusername/detection-fusion.git"
    ])

install_detection_fusion()
```

## 🔍 Verification Tests

### Basic Functionality Test
```python
import numpy as np
from detection_fusion import EnsembleVoting, Detection

# Create test data
detections = {
    "model1": [
        Detection(0, 0.5, 0.5, 0.2, 0.3, 0.8, "model1"),
        Detection(1, 0.7, 0.3, 0.1, 0.2, 0.9, "model1")
    ],
    "model2": [
        Detection(0, 0.52, 0.48, 0.18, 0.32, 0.85, "model2"),
        Detection(2, 0.3, 0.7, 0.15, 0.25, 0.7, "model2")
    ]
}

# Test ensemble
ensemble = EnsembleVoting()
ensemble.detections = detections
ensemble.models = ["model1", "model2"]

# Run strategy
results = ensemble.run_strategy("majority_vote_2")
print(f"Test passed! Generated {len(results)} ensemble detections.")
```

### CLI Test
```bash
# Create test data structure
mkdir -p test_labels/model1 test_labels/model2

# Create test detection files
echo "0 0.5 0.5 0.2 0.3 0.8" > test_labels/model1/detections.txt
echo "1 0.7 0.3 0.1 0.2 0.9" >> test_labels/model1/detections.txt

echo "0 0.52 0.48 0.18 0.32 0.85" > test_labels/model2/detections.txt
echo "2 0.3 0.7 0.15 0.25 0.7" >> test_labels/model2/detections.txt

# Test CLI
detection-fusion --labels-dir test_labels --output-dir test_output

# Check output
ls test_output/
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Error: torch not found
```bash
# Solution: Install PyTorch
pip install torch torchvision

# Or with conda
conda install pytorch torchvision -c pytorch
```

#### 2. Permission Error during installation
```bash
# Solution: Use user installation
pip install --user -e .

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### 3. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4. scikit-learn compatibility
```bash
# Update scikit-learn
pip install --upgrade scikit-learn

# Or install specific version
pip install scikit-learn==1.3.0
```

#### 5. Matplotlib backend issues
```python
# Add to your script
import matplotlib
matplotlib.use('Agg')  # For headless environments
```

### Debug Installation
```python
def debug_installation():
    """Run diagnostic checks."""
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")
    
    try:
        import detection_fusion
        print(f"DetectionFusion version: {detection_fusion.__version__}")
        print("✅ Installation successful")
    except ImportError as e:
        print(f"❌ Installation failed: {e}")

debug_installation()
```

### Performance Issues

#### Memory Usage
```python
# Monitor memory usage
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Use throughout your code
print_memory_usage()
```

#### Speed Optimization
```bash
# Use multiple cores
export OMP_NUM_THREADS=4

# Or set in Python
import os
os.environ['OMP_NUM_THREADS'] = '4'
```

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2GHz
- **RAM**: 4GB
- **Storage**: 1GB free space
- **OS**: Linux, macOS, Windows 10+

### Recommended Requirements  
- **CPU**: 4+ cores, 3GHz+
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **GPU**: Optional, CUDA-compatible

### Large Dataset Requirements
- **RAM**: 16GB+
- **Storage**: SSD recommended
- **CPU**: 8+ cores for parallel processing

## 🔄 Updating

### Update Package
```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e .
```

### Update Dependencies
```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Or specific packages
pip install --upgrade torch torchvision
```

### Migration Guide
When updating between major versions, check the [CHANGELOG.md](CHANGELOG.md) for breaking changes and migration instructions.

## 🆘 Getting Help

If you encounter issues:

1. **Check documentation**: Review this guide and [API.md](API.md)
2. **Search issues**: Look through [GitHub Issues](https://github.com/yourusername/detection-fusion/issues)
3. **Ask questions**: Start a [GitHub Discussion](https://github.com/yourusername/detection-fusion/discussions)
4. **Report bugs**: Create a new [GitHub Issue](https://github.com/yourusername/detection-fusion/issues/new)

Include this information in bug reports:
- Operating system and version
- Python version
- Package version
- Error messages and traceback
- Minimal code to reproduce the issue