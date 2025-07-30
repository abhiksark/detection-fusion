# Troubleshooting Guide

Common issues and solutions for the DetectionFusion package.

## 🚨 Quick Diagnostics

### System Check Script
```python
def system_check():
    """Run comprehensive system diagnostics."""
    import sys
    import os
    import platform
    
    print("=== DetectionFusion System Check ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check core dependencies
    dependencies = [
        'numpy', 'pandas', 'torch', 'torchvision', 
        'sklearn', 'matplotlib', 'seaborn'
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: Not installed")
    
    # Check DetectionFusion
    try:
        import detection_fusion
        print(f"✅ detection_fusion: {detection_fusion.__version__}")
        
        # Test basic functionality
        from detection_fusion import EnsembleVoting
        ensemble = EnsembleVoting()
        print("✅ Basic import test passed")
        
    except Exception as e:
        print(f"❌ detection_fusion: {e}")

system_check()
```

## 🔧 Installation Issues

### Issue: Package Not Found
```
ModuleNotFoundError: No module named 'detection_fusion'
```

**Solutions**:
```bash
# 1. Verify installation
pip list | grep detection-fusion

# 2. Reinstall in development mode
pip install -e .

# 3. Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# 4. Install in correct environment
which python
pip show detection-fusion
```

### Issue: Permission Denied
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
```bash
# 1. Use user installation
pip install --user -e .

# 2. Fix directory permissions
sudo chown -R $USER:$USER ~/.local/

# 3. Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: PyTorch Installation Problems
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solutions**:
```bash
# 1. Install from PyTorch website
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Use conda
conda install pytorch torchvision -c pytorch

# 3. For specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📁 File and Data Issues

### Issue: Detection Files Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'labels/model1/detections.txt'
```

**Diagnosis**:
```bash
# Check directory structure
ls -la labels/
find labels/ -name "*.txt"
```

**Solutions**:
```bash
# 1. Create proper structure
mkdir -p labels/model1 labels/model2 labels/model3

# 2. Check file permissions
ls -la labels/model1/detections.txt

# 3. Use absolute paths
detection-fusion --labels-dir /full/path/to/labels
```

### Issue: Empty or Malformed Detection Files
```
ValueError: could not convert string to float: 'invalid_data'
```

**Diagnosis**:
```python
def validate_detection_file(file_path):
    """Validate detection file format."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 6:
                print(f"Line {i}: Too few columns ({len(parts)})")
            try:
                class_id = int(parts[0])
                x, y, w, h, conf = map(float, parts[1:6])
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    print(f"Line {i}: Coordinates out of range")
                if not (0 <= conf <= 1):
                    print(f"Line {i}: Confidence out of range")
            except ValueError as e:
                print(f"Line {i}: Parsing error - {e}")

# Usage
validate_detection_file("labels/model1/detections.txt")
```

**Solutions**:
```python
# 1. Fix format
# Correct format: class_id x_center y_center width height confidence
# Example: 0 0.5 0.3 0.2 0.4 0.85

# 2. Clean data
def clean_detection_file(input_path, output_path):
    """Clean and fix detection file."""
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    class_id = int(float(parts[0]))  # Handle float class IDs
                    x, y, w, h, conf = map(float, parts[1:6])
                    
                    # Clamp values to valid ranges
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    conf = max(0, min(1, conf))
                    
                    f_out.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                except ValueError:
                    continue  # Skip invalid lines
```

## 🧮 Memory and Performance Issues

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# 1. Process in batches
def process_large_dataset(detections, batch_size=1000):
    results = []
    for i in range(0, len(detections), batch_size):
        batch = dict(list(detections.items())[i:i+batch_size])
        batch_results = ensemble.run_strategy("majority_vote_2", batch)
        results.extend(batch_results)
    return results

# 2. Use CPU-only mode
import torch
torch.cuda.is_available = lambda: False

# 3. Reduce data size
filtered_detections = {
    model: [d for d in dets if d.confidence > 0.5]
    for model, dets in detections.items()
}
```

### Issue: Slow Performance
```python
# 1. Profile code
import time
import cProfile

def profile_ensemble():
    start = time.time()
    results = ensemble.run_all_strategies()
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    return results

# 2. Use faster strategies first
fast_strategies = ["nms", "majority_vote_2"]
slow_strategies = ["bayesian", "dbscan"]

# 3. Parallel processing
import multiprocessing as mp

def parallel_strategy(strategy_name):
    return ensemble.run_strategy(strategy_name)

with mp.Pool(4) as pool:
    results = pool.map(parallel_strategy, strategy_names)
```

## 🐛 Recently Fixed Issues (v0.2.0)

### Issue: "unhashable type: 'Detection'" Error
**Fixed in v0.2.0**: Detection objects can now be used in sets and dictionaries.

**Problem**: When running gt_rectify.py or certain ensemble operations, you might encounter:
```
TypeError: unhashable type: 'Detection'
```

**Solution**: This has been resolved by adding `__hash__` and `__eq__` methods to the Detection class. Update to v0.2.0 or later.

```python
# Now works correctly:
detection_set = set(detections)
detection_dict = {detection: confidence for detection, confidence in zip(detections, confidences)}
```

### Issue: GT Rectifier Initialization Errors
**Fixed in v0.2.0**: Proper parameter handling in GTRectifier initialization.

**Problem**: GT rectification failing with parameter errors like:
```
expected str, bytes or os.PathLike object, not NoneType
```

**Solution**: Fixed AdvancedEnsemble parameter passing and config value extraction:

```python
# Old (problematic) - Fixed in v0.2.0
ensemble = AdvancedEnsemble(labels_dir)  # Missing output_dir parameter

# New (correct) - Works in v0.2.0
ensemble = AdvancedEnsemble(labels_dir, output_dir, gt_dir)
```

### Issue: Configuration Loading Problems
**Fixed in v0.2.0**: Robust configuration value extraction with proper defaults.

**Problem**: Configuration files not being read correctly, resulting in None values.

**Solution**: Enhanced `get_config_value()` function with proper fallback mechanisms:

```python
# Now includes proper defaults
labels_dir = get_config_value('labels-dir', 'gt_rectification.labels_dir', 'labels')
```

### Issue: Progress Tracking Missing
**Fixed in v0.2.0**: Comprehensive tqdm integration throughout the codebase.

**Problem**: Long-running operations without progress feedback.

**Solution**: All CLI tools now include real-time progress bars:
- Model loading: `Loading models: 100%|██████████| 4/4`
- GT processing: `Loading GT: 100%|██████████| 2010/2010`
- Strategy execution: `Running strategies: 100%|██████████| 17/17`
- Image analysis: `Analyzing images: 100%|██████████| 150/150`

## 🎯 Strategy-Specific Issues

### Issue: No Detections from Ensemble
```python
# Diagnosis
def diagnose_empty_results():
    """Diagnose why ensemble produces no results."""
    
    # Check input detections
    total_detections = sum(len(dets) for dets in ensemble.detections.values())
    print(f"Total input detections: {total_detections}")
    
    if total_detections == 0:
        print("❌ No input detections found")
        return
    
    # Check per-model counts
    for model, dets in ensemble.detections.items():
        print(f"{model}: {len(dets)} detections")
    
    # Test with permissive parameters
    permissive_results = ensemble.run_strategy("majority_vote_2")
    print(f"Permissive results: {len(permissive_results)}")
    
    # Check IoU threshold
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        ensemble.strategies["majority_vote_2"].iou_threshold = threshold
        results = ensemble.run_strategy("majority_vote_2")
        print(f"IoU {threshold}: {len(results)} detections")

diagnose_empty_results()
```

**Solutions**:
```python
# 1. Lower IoU threshold
strategy = MajorityVoting(iou_threshold=0.3, min_votes=2)

# 2. Reduce minimum votes
strategy = MajorityVoting(iou_threshold=0.5, min_votes=1)

# 3. Use more permissive strategy
results = ensemble.run_strategy("nms")  # Most permissive
```

### Issue: Too Many Detections
```python
# Solutions
# 1. Increase IoU threshold
strategy = MajorityVoting(iou_threshold=0.7, min_votes=3)

# 2. Use conservative strategies
results = ensemble.run_strategy("unanimous")

# 3. Apply confidence filtering
high_conf_results = [d for d in results if d.confidence > 0.7]

# 4. Apply NMS post-processing
from detection_fusion.strategies import NMSStrategy
nms = NMSStrategy(iou_threshold=0.5, score_threshold=0.5)
final_results = nms.merge({"ensemble": results})
```

## 📊 Visualization Issues

### Issue: Plots Not Displaying
```python
# Solutions
# 1. Set backend explicitly
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'
import matplotlib.pyplot as plt

# 2. For headless environments
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode

# 3. Save instead of show
analyzer.plot_class_distribution(save_path="plot.png")
```

### Issue: Font/Display Problems
```python
# Solutions
# 1. Set font
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

# 2. Increase figure size
plt.rcParams['figure.figsize'] = [12, 8]

# 3. Set DPI
plt.rcParams['figure.dpi'] = 100
```

## 🔍 Analysis Issues

### Issue: Inconsistent Model Comparisons
```python
def debug_model_comparison():
    """Debug model comparison issues."""
    
    analyzer = MultiModelAnalyzer("labels")
    analyzer.load_detections("detections.txt")
    
    # Check model loading
    print("Loaded models:", analyzer.models)
    for model, dets in analyzer.detections.items():
        print(f"{model}: {len(dets)} detections")
        if dets:
            conf_stats = [d.confidence for d in dets]
            print(f"  Confidence range: {min(conf_stats):.3f} - {max(conf_stats):.3f}")
    
    # Check comparison parameters
    print(f"IoU threshold: {analyzer.iou_threshold}")
    
    # Test pairwise comparison
    if len(analyzer.models) >= 2:
        result = analyzer.compare_models(analyzer.models[0], analyzer.models[1])
        print("Sample comparison:", result)

debug_model_comparison()
```

### Issue: Class Name Mapping Problems
```python
# Solutions
# 1. Create class names file
class_names = ["person", "bicycle", "car", "motorcycle", "airplane"]
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

# 2. Load custom class names
analyzer.load_class_names("class_names.txt")

# 3. Set class names programmatically
analyzer.class_names = {0: "person", 1: "car", 2: "bike"}
```


## 🔧 Advanced Troubleshooting

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('detection_fusion')

# Add to your code
logger.debug("Loading detections...")
logger.info(f"Loaded {len(detections)} detections")
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_test():
    ensemble = EnsembleVoting("labels")
    results = ensemble.run_all_strategies()
    return results

# Run with: python -m memory_profiler script.py
```

### Performance Profiling
```python
import cProfile
import pstats

def profile_ensemble():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your code here
    results = ensemble.run_all_strategies()
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

profile_ensemble()
```

## 📞 Getting Help

### Information to Include in Bug Reports

```python
def collect_debug_info():
    """Collect system information for bug reports."""
    import sys
    import platform
    import detection_fusion
    
    info = {
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": sys.version,
        "detection_fusion_version": detection_fusion.__version__,
        "working_directory": os.getcwd()
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = "Not installed"
    
    return info

# Usage
debug_info = collect_debug_info()
print("Please include this information in your bug report:")
for key, value in debug_info.items():
    print(f"{key}: {value}")
```

### Minimal Reproducible Example Template
```python
"""
Minimal example to reproduce the issue.
Replace with your specific case.
"""
from detection_fusion import EnsembleVoting, Detection

# Create minimal test data
detections = {
    "model1": [Detection(0, 0.5, 0.5, 0.2, 0.3, 0.8, "model1")],
    "model2": [Detection(0, 0.52, 0.48, 0.18, 0.32, 0.85, "model2")]
}

# Reproduce the issue
ensemble = EnsembleVoting()
ensemble.detections = detections
ensemble.models = ["model1", "model2"]

try:
    results = ensemble.run_strategy("majority_vote_2")
    print(f"Expected: 1 detection, Got: {len(results)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### Where to Get Help

1. **Documentation**: Check [API.md](API.md) and [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)
2. **GitHub Issues**: Search existing issues or create new one
3. **GitHub Discussions**: Ask questions and get community help
4. **Stack Overflow**: Tag questions with `detection-fusion`

Remember to always include:
- Complete error messages and tracebacks
- System information (OS, Python version, etc.)
- Minimal code to reproduce the issue
- What you expected vs. what actually happened