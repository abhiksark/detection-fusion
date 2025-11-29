# Contributing to DetectionFusion

Thank you for your interest in contributing to the DetectionFusion package! This guide will help you get started with contributing to our project.

## 🤝 Ways to Contribute

### 1. Bug Reports
- Report bugs through [GitHub Issues](https://github.com/abhiksark/detection-fusion/issues)
- Use the bug report template
- Include system information and minimal reproduction code

### 2. Feature Requests
- Suggest new features through [GitHub Discussions](https://github.com/abhiksark/detection-fusion/discussions)
- Explain the use case and expected behavior
- Consider implementation complexity

### 3. Documentation
- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify explanations

### 4. Code Contributions
- Bug fixes
- New ensemble strategies
- Performance improvements
- Test coverage improvements

## 📋 v1.0 Architecture Notes

Before contributing, familiarize yourself with the v1.0 architecture:

### Detection Model (Pydantic)
Detection is now an **immutable Pydantic model**. Use keyword arguments:
```python
from detection_fusion import Detection

# Create detection with keyword args (required)
det = Detection(
    class_id=0, x=0.5, y=0.5, w=0.2, h=0.3,
    confidence=0.95, model_source="yolov8n"
)

# Detection is frozen - use with_* methods to create modified copies
det2 = det.with_confidence(0.8)
det3 = det.with_source("yolov8s")
```

### Strategy Registry
Use `create_strategy()` instead of instantiating strategy classes directly:
```python
from detection_fusion.strategies import create_strategy, StrategyRegistry

# Create strategy by name
strategy = create_strategy("weighted_vote", iou_threshold=0.5)

# List available strategies
print(StrategyRegistry.list_all())
```

### Pipeline API
Use the fluent Pipeline API for chaining operations:
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

### Configuration (Pydantic)
Use Pydantic config models with builder pattern:
```python
from detection_fusion.config import StrategyConfig

config = (
    StrategyConfig()
    .with_overlap(threshold=0.6)
    .with_voting(min_votes=3)
)
```

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
```bash
# Fork the repository on GitHub
git clone https://github.com/abhiksark/detection-fusion.git
cd detection-fusion
```

2. **Set Up Environment**
```bash
# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,viz]"
```

3. **Install Development Tools**
```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pytest --version
black --version
flake8 --version
```

### Development Workflow

1. **Create Feature Branch**
```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/issue-description
```

2. **Make Changes**
- Write code following our style guidelines
- Add tests for new functionality
- Update documentation if needed

3. **Test Your Changes**
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=detection_fusion

# Run specific test file
pytest tests/strategies/test_registry.py
```

4. **Format and Lint**
```bash
# Format code
black detection_fusion tests

# Lint code
flake8 detection_fusion tests

# Check imports
isort detection_fusion tests --check-only
```

5. **Commit Changes**
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new consensus strategy

- Implement consensus-based voting strategy
- Add tests for edge cases
- Update documentation with usage examples"
```

6. **Push and Create PR**
```bash
git push origin feat/your-feature-name
# Create Pull Request on GitHub
```

## 📝 Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes.
    
    Args:
        box1: First bounding box in [x, y, w, h] format
        box2: Second bounding box in [x, y, w, h] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Implementation here
    pass

# Bad
def calc_iou(b1, b2):
    # No docstring, unclear parameter names
    pass
```

### Documentation Style

Use Google-style docstrings:

```python
def merge_detections(detections: Dict[str, List[Detection]], 
                    iou_threshold: float = 0.5) -> List[Detection]:
    """Merge detections from multiple models.
    
    Args:
        detections: Dictionary mapping model names to their detections
        iou_threshold: IoU threshold for matching detections
        
    Returns:
        List of merged detections
        
    Raises:
        ValueError: If detections dictionary is empty
        
    Examples:
        >>> detections = {"model1": [det1, det2], "model2": [det3]}
        >>> merged = merge_detections(detections, iou_threshold=0.5)
        >>> len(merged)
        2
    """
```

### Type Hints

Use comprehensive type hints:

```python
from typing import List, Dict, Optional, Union
from detection_fusion.core.detection import Detection

def process_results(results: Optional[Dict[str, List[Detection]]], 
                   threshold: Union[int, float] = 0.5) -> List[Detection]:
    """Process ensemble results with proper type hints."""
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

1. **Test Structure**
```python
import pytest
from detection_fusion.strategies import create_strategy
from detection_fusion import Detection

class TestMajorityVoting:
    """Test cases for MajorityVoting strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = create_strategy("majority_vote", iou_threshold=0.5)
        self.sample_detections = {
            "model1": [Detection(
                class_id=0, x=0.5, y=0.5, w=0.2, h=0.3,
                confidence=0.8, model_source="model1"
            )],
            "model2": [Detection(
                class_id=0, x=0.52, y=0.48, w=0.18, h=0.32,
                confidence=0.85, model_source="model2"
            )]
        }

    def test_merge_basic(self):
        """Test basic merging functionality."""
        results = self.strategy.merge(self.sample_detections)
        assert len(results) == 1
        assert results[0].class_id == 0

    def test_merge_empty_input(self):
        """Test handling of empty input."""
        results = self.strategy.merge({})
        assert len(results) == 0

    @pytest.mark.parametrize("iou_threshold,expected_count", [
        (0.3, 1),
        (0.7, 0),
        (0.5, 1)
    ])
    def test_iou_threshold_sensitivity(self, iou_threshold, expected_count):
        """Test sensitivity to IoU threshold."""
        strategy = create_strategy("majority_vote", iou_threshold=iou_threshold)
        results = strategy.merge(self.sample_detections)
        assert len(results) == expected_count
```

2. **Test Categories**
- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

3. **Test Data**
```python
# Use realistic test data
@pytest.fixture
def sample_detections():
    """Provide realistic detection data for testing."""
    return {
        "yolov8n": [
            Detection(class_id=0, x=0.5, y=0.3, w=0.2, h=0.4,
                      confidence=0.85, model_source="yolov8n"),
            Detection(class_id=1, x=0.7, y=0.6, w=0.1, h=0.3,
                      confidence=0.92, model_source="yolov8n")
        ],
        "yolov8s": [
            Detection(class_id=0, x=0.52, y=0.28, w=0.18, h=0.42,
                      confidence=0.88, model_source="yolov8s"),
            Detection(class_id=2, x=0.3, y=0.7, w=0.15, h=0.25,
                      confidence=0.78, model_source="yolov8s")
        ]
    }
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=detection_fusion --cov-report=html

# Run specific test file
pytest tests/strategies/test_registry.py

# Run specific test
pytest tests/strategies/test_voting.py::TestMajorityVoting::test_merge_basic

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## 🎯 Adding New Ensemble Strategies

### Strategy Implementation (v1.0 Pattern)

1. **Create Strategy Class with Registry Decorator**
```python
from detection_fusion.strategies.base import BaseStrategy, StrategyMetadata
from detection_fusion.strategies.registry import StrategyRegistry
from detection_fusion import Detection
from typing import List, Dict

@StrategyRegistry.register("my_custom_strategy")
class MyCustomStrategy(BaseStrategy):
    """Custom ensemble strategy implementation."""

    # Required: Strategy metadata for registry
    metadata = StrategyMetadata(
        name="my_custom_strategy",
        category="custom",  # voting, nms, clustering, etc.
        description="My custom ensemble strategy"
    )

    def __init__(self, iou_threshold: float = 0.5, custom_param: float = 1.0, **kwargs):
        super().__init__(iou_threshold=iou_threshold, **kwargs)
        self.custom_param = custom_param

    @property
    def name(self) -> str:
        return "my_custom_strategy"

    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Implement your custom merging logic."""
        merged_detections = []

        # Example: Simple implementation
        all_detections = []
        for model_detections in detections.values():
            all_detections.extend(model_detections)

        # Apply your custom logic
        # ...

        return merged_detections
```

2. **Add Tests**
```python
from detection_fusion.strategies import create_strategy

class TestMyCustomStrategy:
    """Test cases for MyCustomStrategy."""

    def setup_method(self):
        self.strategy = create_strategy("my_custom_strategy", custom_param=2.0)

    def test_custom_logic(self):
        """Test custom strategy logic."""
        # Your tests here
        pass
```

3. **Update Module Imports**
```python
# In detection_fusion/strategies/__init__.py
# The @StrategyRegistry.register decorator auto-registers the strategy,
# but you still need to import the module to trigger registration:
from . import my_module  # This triggers the decorator

# Export for direct class access if needed
from .my_module import MyCustomStrategy
```

4. **Add Documentation**
```python
def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
    """Merge detections using custom algorithm.

    This strategy implements [describe your approach].

    Args:
        detections: Dictionary mapping model names to their detections
        **kwargs: Additional parameters

    Returns:
        List of merged detections

    Note:
        This strategy works best when [describe use case].
    """
```

## 📚 Documentation Contributions

### Types of Documentation

1. **API Documentation**: Function/class docstrings
2. **User Guides**: How-to guides and tutorials  
3. **Examples**: Code examples and use cases
4. **Reference**: Complete API reference

### Documentation Style

- Use clear, concise language
- Include code examples
- Provide practical use cases
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## 🐛 Bug Report Guidelines

### Before Reporting

1. **Search existing issues**: Check if the bug has already been reported
2. **Update to latest version**: Ensure you're using the latest version
3. **Minimal reproduction**: Create minimal code to reproduce the issue

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load detections from '...'
2. Run strategy '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Code to Reproduce**
```python
# Minimal code example
from detection_fusion import merge_detections, Detection
# ... rest of code
```

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package version: [e.g. 1.0.0]
- PyTorch version: [e.g. 1.12.0] (optional)

**Additional Context**
Any other context about the problem.
```

## 🎉 Recognition

### Contributors

Contributors are recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes
- Documentation credits

### Contribution Types

We recognize all types of contributions:
- 🐛 Bug fixes
- ✨ New features  
- 📝 Documentation
- 🧪 Tests
- 🎨 Code style improvements
- 🚀 Performance improvements
- 💡 Ideas and discussions

## 📞 Getting Help

### Community Support

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: Get feedback on your contributions

### Development Questions

If you have questions about development:

1. Check existing documentation
2. Search closed issues and discussions
3. Ask in GitHub Discussions
4. Tag maintainers if needed

## 📋 Release Process

### For Maintainers

1. **Version Bump**
```bash
# Update version in detection_fusion/_version.py
# Update CHANGELOG.md with new version
```

2. **Testing**
```bash
# Run full test suite
pytest

# Test installation
pip install -e .
```

3. **Release**
```bash
# Create release tag
git tag v1.0.0
git push origin v1.0.0

# Create GitHub release with changelog
```

## 🙏 Thank You

Thank you for contributing to DetectionFusion! Your contributions help make this project better for everyone in the object detection community.

Questions? Feel free to reach out through GitHub Discussions or Issues!