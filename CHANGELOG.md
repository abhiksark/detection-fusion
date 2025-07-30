# Changelog

All notable changes to the DetectionFusion package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- COCO format input/output support
- Model confidence calibration
- Uncertainty quantification
- Web-based visualization dashboard

## [0.2.0] - 2025-07-30

### Added

#### New Advanced Strategies (10 Added!)
- **Distance-Based Strategies**:
  - **DistanceWeightedVoting**: Weights detections by spatial distance to cluster centroid
  - **CentroidClustering**: Agglomerative clustering based on detection centers
- **Confidence-Based Strategies**:
  - **ConfidenceThresholdVoting**: Adaptive confidence thresholds per model
  - **ConfidenceWeightedNMS**: NMS with confidence-weighted box regression
  - **HighConfidenceFirst**: Prioritizes high-confidence detections hierarchically
- **Adaptive Strategies**:
  - **AdaptiveThresholdStrategy**: Different IoU thresholds for small vs large objects
  - **DensityAdaptiveStrategy**: Context-aware processing for high/low density regions
  - **MultiScaleStrategy**: Scale-specific processing (tiny/small/medium/large objects)
  - **ConsensusRankingStrategy**: Combines model ranking with confidence scores

#### Configuration System Overhaul
- **External YAML Configuration**: Moved configs outside package to `configs/` directory
- **Organized Configuration Structure**: Separated configs into logical subdirectories:
  - `configs/ensemble/`: Ensemble and analysis configurations
  - `configs/gt_rectification/`: GT rectification configurations
- **Specialized Config Files**:
  - `ensemble/default_config.yaml`: Standard ensemble configuration  
  - `ensemble/high_precision_config.yaml`: Conservative strategies for critical applications
  - `ensemble/high_recall_config.yaml`: Permissive strategies for maximum detection
  - `ensemble/small_objects_config.yaml`: Optimized for small object detection
  - `ensemble/advanced_strategies_config.yaml`: Showcases all 17+ strategies
- **YAML Configuration Functions**: `load_yaml_config()` and `save_yaml_config()`
- **Configuration Documentation**: Comprehensive `configs/README.md` with usage guide

#### Professional CLI Tools
- **merge.py**: Ensemble merging tool (intuitive train-like interface)
  - Multiple input modes: specific models, model directory, or config file
  - 17+ ensemble strategies available
  - Strategy comparison mode with individual result saving
  - Comprehensive parameter control (IoU, confidence, min votes)
  - Multiple output formats (YOLO, JSON)
  - Verbose/quiet modes with detailed progress reporting
- **val.py**: Model assessment tool (comprehensive validation interface)
  - Model agreement analysis with Jaccard index
  - Confidence distribution analysis with histogram bins
  - Class-wise performance breakdown
  - Strategy comparison with effectiveness metrics
  - Comprehensive visualization (plots, heatmaps, charts)
  - Multiple report formats (summary, full, JSON)
  - Pairwise model comparison matrix

#### Requirements Management
- **requirements.txt**: Production dependencies for end users
- **requirements-dev.txt**: Comprehensive development environment
  - Testing framework (pytest, pytest-cov, pytest-mock)
  - Code quality tools (black, flake8, isort, mypy)
  - Documentation tools (sphinx, sphinx-rtd-theme)
  - Development utilities (pre-commit, jupyter, profiling tools)

#### Ground Truth Evaluation Framework (NEW!)
- **Complete ground truth evaluation system** with support for standard object detection metrics (AP, mAP, precision, recall, F1-score)
- **Error analysis framework** with detailed error classification (False Positives, False Negatives, Localization Errors, Classification Errors, Duplicate Detections)
- **Strategy optimization** using ground truth feedback for automatic best strategy selection
- **COCO-style evaluation** with multiple AP calculation methods (11-point, all-points, COCO-style interpolation)

#### New Evaluation Module (`detection_fusion.evaluation`)
- `EvaluationMetrics` class with comprehensive metric calculations
- `APCalculator` with multiple interpolation methods
- `ErrorAnalyzer` for detailed error classification and analysis
- `Evaluator` as main orchestrator for evaluation workflows
- `StrategyOptimizer` for GT-based strategy selection and parameter tuning

#### GT Rectification System (NEW!)
- **Ground Truth Rectification System** (`gt_rectify.py`) for identifying potential annotation errors
- **Two-mode analysis**: Conservative (minimize_error) and Aggressive (maximize_error) approaches
- **Consensus-based error detection** across all 17+ ensemble strategies  
- **Organized dataset creation** with most correct/incorrect images for human review
- **Comprehensive error analysis** with confidence scoring and strategy diversity assessment
- **Professional CLI interface** with extensive help, validation, and progress reporting
- **Human-readable reports** with specific recommendations for GT improvements
- **YAML configuration support** with 4 pre-configured rectification profiles:
  - `gt_rectify_conservative.yaml`: High-precision error detection
  - `gt_rectify_aggressive.yaml`: Comprehensive error detection
  - `gt_rectify_balanced.yaml`: Balanced precision/recall approach
  - `gt_rectify_custom.yaml`: Template for custom configurations

#### Enhanced Core Components
- **Ground truth support** added to `Detection`, `EnsembleVoting`, and `AdvancedEnsemble` classes
- **GT directory structure validation** with automatic discovery of ground truth files
- **Caching system** for efficient ground truth data loading
- **Enhanced data structures** with GT comparison methods across all core classes

#### CLI Tool Enhancements

##### val.py (Assessment Tool)
- `--gt` parameter for ground truth evaluation mode
- `--gt-dir` and `--gt-file` for GT path specification
- `--error-analysis` for detailed error breakdown
- `--optimize-strategy` for finding optimal ensemble strategies
- `--benchmark-strategies` for comparing all strategies against GT
- `--metrics` for selecting specific evaluation metrics
- New analysis types: "evaluation" and "error-analysis"
- Comprehensive GT evaluation reporting in multiple formats

##### merge.py (Ensemble Tool)
- `--gt` parameter for GT-guided strategy selection
- `--auto-strategy` for automatic optimal strategy selection
- `--optimize-strategy` for GT-based strategy optimization
- `--evaluation-metric` for choosing optimization criteria (map_50, f1_score, precision, recall)
- **Ground truth-guided workflow** with intelligent fallback mechanisms
- **Strategy performance comparison** with detailed GT metrics display
- **Enhanced reporting** showing strategy performance vs ground truth

##### gt_rectify.py (Rectification Tool) 
- Comprehensive CLI for GT error detection and dataset organization
- Support for both conservative and aggressive error detection modes
- Extensive parameter customization (IoU thresholds, confidence levels, strategy agreement)
- Directory validation and progress reporting
- Mode-aware recommendations and analysis interpretation

#### Enhanced Examples and Documentation
- **Advanced Strategy Demo**: Comprehensive demonstration of all new strategies
- **Configuration Usage Examples**: YAML-based configuration workflows
- **Strategy Performance Analysis**: Comparative analysis tools
- **GT Rectification Examples**: Complete demonstrations of both rectification modes
- **Examples moved to root level** for better accessibility and organization
- **Configuration Guide Example**: `gt_rectify_config_example.py` with comprehensive usage demonstrations

#### Project Infrastructure Improvements
- **Comprehensive .gitignore**: Professional .gitignore covering:
  - Python development (IDEs, virtual environments, build artifacts)
  - Machine learning workflows (models, datasets, experiment outputs)
  - Object detection specific files (detection results, label files, annotations)
  - Data privacy protection (user configs, API keys, large data files)
  - Development tools (profiling, CI/CD, documentation builds)
- **Organized Project Structure**: Clean separation of concerns with logical directory organization
- **Codebase Cleanup**: Removed obsolete files (cococompare.py, compare.py, ensemble_voting.py, ensemble_advanced.py, multi_model_analysis.py)
- **Code Quality Assurance**: 
  - Ruff linting: 0 errors across 34 Python files
  - MyPy type checking: Configured with gradual typing support
  - 8,612 lines of clean, professional Python code

### Changed

#### Configuration Format Migration
- **JSON → YAML**: All configuration files now use YAML format for better readability
- **External Configuration**: Configs moved from `detection_fusion/configs/` to `configs/`
- **Enhanced Configuration Structure**: Added visualization and analysis sections

#### CLI Removal
- **Removed CLI Tools**: Eliminated `detection-fusion` and `od-analyze` command-line interfaces
- **Pure Python API**: Package now focuses entirely on programmatic usage
- **Simplified Installation**: No console script entry points

#### Documentation Updates
- **Complete Strategy Guide Rewrite**: Updated with all 17+ strategies and examples
- **YAML Configuration Examples**: All docs updated to use YAML instead of JSON
- **Performance Optimization Guide**: Added strategy performance characteristics

### Dependencies Added
- **PyYAML>=5.4.0**: For YAML configuration support
- Enhanced scikit-learn usage for distance calculations in clustering strategies

### Total Package Stats (v0.2.0)
- **17+ Ensemble Strategies**: Comprehensive coverage of all ensemble approaches
- **6 Strategy Categories**: From basic voting to advanced adaptive methods
- **9 Specialized Configs**: Ready-to-use configurations organized in logical directories:
  - 5 ensemble configurations (default, high-precision, high-recall, small-objects, advanced-strategies)
  - 4 GT rectification configurations (conservative, aggressive, balanced, custom template)
- **Professional Project Structure**: Organized configs, comprehensive .gitignore, clean documentation
- **Zero CLI Dependencies**: Pure Python API for maximum flexibility

## [0.1.0] - 2025-07-30

### Added

#### Core Features
- **EnsembleVoting** class for basic ensemble operations
- **AdvancedEnsemble** class with advanced strategies
- **MultiModelAnalyzer** for comprehensive model analysis
- **Detection** data class for representing object detections

#### Basic Ensemble Strategies (7 strategies)
- **Majority Voting**: Requires minimum number of models to agree
- **Weighted Voting**: Weights by confidence scores and model performance
- **NMS Strategy**: Standard Non-Maximum Suppression
- **Affirmative NMS**: NMS requiring multi-model agreement
- **DBSCAN Clustering**: Density-based spatial clustering
- **Soft Voting**: Probabilistic voting with temperature scaling
- **Bayesian Fusion**: Bayesian inference with class priors

#### Analysis Capabilities
- Pairwise model comparison with IoU-based matching
- Class-wise detection statistics and variance analysis
- Confidence score distribution analysis
- Consensus detection identification
- Model similarity assessment

#### Visualization
- Class distribution bar charts
- Confidence histogram plots
- Model similarity heatmaps
- Variance analysis plots
- Automated plot generation pipeline

#### Utilities
- File I/O operations for detection formats
- IoU, GIoU, and DIoU metric calculations
- Class name loading and management
- JSON configuration support

#### Command Line Interface
- `detection-fusion`: Main ensemble voting tool
- `od-analyze`: Model analysis and comparison tool
- Comprehensive CLI argument support
- Batch processing capabilities

#### Documentation
- Complete API reference
- Strategy selection guide
- Installation instructions
- Troubleshooting guide
- Usage examples and tutorials

### Technical Details

#### Supported Formats
- YOLO format detection files (normalized coordinates)
- Custom class name files
- JSON configuration files
- Multiple output formats (text, JSON, plots)

#### Performance Features
- Efficient IoU calculations using PyTorch
- Memory-optimized batch processing
- Configurable strategy parameters
- Parallel processing support

#### Quality Assurance
- Comprehensive test suite
- Type hints throughout codebase
- Code formatting with Black
- Linting with Flake8
- Documentation coverage

### Dependencies
- Python 3.7+
- NumPy >= 1.19.0
- Pandas >= 1.1.0
- PyTorch >= 1.7.0
- scikit-learn >= 0.23.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0

### Installation
```bash
pip install -e .
```

### Breaking Changes
- N/A (Initial release)

### Migration Guide
- N/A (Initial release)

---

## Version History

### Pre-release Development

#### [0.0.3] - 2025-07-27
- Added advanced ensemble strategies
- Implemented visualization system
- Created CLI tools

#### [0.0.2] - 2025-07-22
- Refactored code into modular structure
- Added strategy base classes
- Implemented basic voting strategies

#### [0.0.1] - 2025-07-15
- Initial prototype
- Basic majority voting implementation
- Simple model comparison tools

---

## Future Roadmap

### Version 0.3.0
- **New Features**:
  - COCO format input/output support
  - Real-time ensemble processing
  - Model confidence calibration
  - Uncertainty quantification
  - Docker containerization
  - Comprehensive Testing

- **Improvements**:
  - GPU acceleration for large datasets
  - Enhanced visualization options
  - Performance optimizations
  - Extended test coverage

### Version 0.4.0
- **New Features**:
  - Web-based dashboard
  - REST API interface
  - Database integration
  - Automated hyperparameter tuning
  - Model drift detection

- **Improvements**:
  - Streaming data support
  - Advanced metrics calculation
  - Multi-format input support
  - Cloud deployment options

### Version 1.0.0
- **Stable Release**:
  - Production-ready features
  - Complete documentation
  - Enterprise support
  - Performance benchmarks
  - Security audit

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- How to report bugs
- How to suggest new features  
- Development workflow
- Code style guidelines
- Testing requirements

### Contributor Recognition

Special thanks to all contributors:

- **Core Development Team**: Initial package development
- **Community Contributors**: Bug reports, feature requests, and improvements
- **Beta Testers**: Early testing and feedback

---

## Support and Maintenance

### Support Policy
- **Bug Fixes**: Provided for current and previous major version
- **Security Updates**: Provided for all supported versions
- **Feature Updates**: Added to latest version only


### Release Schedule
- **Patch releases**: Monthly (bug fixes, minor improvements)
- **Minor releases**: Quarterly (new features, non-breaking changes)
- **Major releases**: Annually (breaking changes, major features)

---

For more information, see:
- [Installation Guide](INSTALLATION.md)
- [API Documentation](API.md)
- [Strategy Guide](STRATEGY_GUIDE.md)
- [Examples](EXAMPLES.md)
- [Troubleshooting](TROUBLESHOOTING.md)