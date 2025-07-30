# Ensemble Strategy Guide

Comprehensive guide to choosing and configuring ensemble strategies for your object detection models. Now featuring **19 advanced strategies** for every use case.

## 🎯 Strategy Selection Guide

### Strategy Categories Overview

| Category | Best For | Strategies | Key Benefits |
|----------|----------|------------|--------------|
| **Basic Voting** | General purpose, balanced results | Majority, Weighted, Unanimous | Simple, robust, interpretable |
| **NMS-Based** | Fast processing, duplicate removal | NMS, Affirmative NMS, Confidence NMS | Speed, established methodology |
| **Clustering** | Variable object sizes, spatial relationships | DBSCAN, Centroid, Distance-weighted | Handles overlaps naturally |
| **Probabilistic** | Uncertainty quantification, class imbalance | Soft Voting, Bayesian Fusion | Principled uncertainty handling |
| **Confidence-Based** | Quality filtering, adaptive thresholds | Confidence Threshold, High Conf First | Leverages model confidence |
| **Adaptive** | Multi-scale objects, varying conditions | Adaptive Threshold, Multi-scale, Density | Context-aware processing |

### When to Use Each Strategy

| Strategy | Best For | Pros | Cons | Typical Use Case |
|----------|----------|------|------|------------------|
| **Majority Voting** | General purpose | Simple, robust | May miss rare objects | Balanced precision/recall |
| **Weighted Voting** | Different model quality | Leverages model strength | Needs calibration | Heterogeneous model ensemble |
| **Unanimous** | High precision needs | Very conservative | Low recall | Critical applications |
| **NMS** | High recall scenarios | Fast, preserves best | No consensus | Speed-critical applications |
| **Affirmative NMS** | Quality + speed balance | Multi-model validation | Moderate complexity | Production systems |
| **DBSCAN** | Varying object sizes | Handles spatial patterns | Parameter sensitive | Dense object scenes |
| **Soft Voting** | Probabilistic fusion | Smooth boundaries | Computationally complex | Uncertainty quantification |
| **Bayesian Fusion** | Class imbalance | Principled approach | Requires priors | Scientific applications |
| **Distance Weighted** | Spatial consistency | Location-aware | Distance metric dependent | Spatially coherent objects |
| **Centroid Clustering** | Irregular shapes | Shape-agnostic | Clustering overhead | Non-rectangular objects |
| **Confidence Threshold** | Quality filtering | Adaptive filtering | Threshold selection | Mixed-quality models |
| **Confidence NMS** | Box refinement | Refined localization | Parameter tuning | Precise localization |
| **High Conf First** | Hierarchical quality | Clear prioritization | Binary decision | Tiered confidence levels |
| **Adaptive Threshold** | Multi-scale objects | Scale-aware | Size estimation needed | Small + large objects |
| **Density Adaptive** | Crowded scenes | Context-aware | Grid parameter tuning | Surveillance, crowds |
| **Multi-Scale** | Size variation | Optimized per scale | Complexity | Multi-resolution datasets |
| **Consensus Ranking** | Model reliability | Rank-aware | Ranking computation | Ranked model outputs |

## 📊 Strategy Deep Dive

### Command Line Usage

Before diving into Python API details, here are command-line examples for quick usage:

```bash
# Try different strategies via CLI
python merge.py --models model1 model2 model3 --strategy majority_vote_2
python merge.py --models model1 model2 model3 --strategy weighted_vote  
python merge.py --models model1 model2 model3 --strategy multi_scale

# Compare multiple strategies
python merge.py --models-dir labels/ --strategies majority_vote_2 weighted_vote affirmative_nms

# Use specialized configurations
python merge.py --config configs/small_objects_config.yaml
python merge.py --config configs/high_precision_config.yaml
```

### 1. Basic Voting Strategies

#### Majority Voting
**When to use**: Default choice for most scenarios.

```bash
# CLI usage
python merge.py --models model1 model2 model3 --strategy majority_vote_2 --iou 0.5
python merge.py --models model1 model2 model3 --strategy majority_vote_3 --iou 0.5
```

```python
# Python API
from detection_fusion.strategies import MajorityVoting

# Conservative: Require 3+ models
strict_voter = MajorityVoting(iou_threshold=0.5, min_votes=3)

# Permissive: Require 2+ models  
permissive_voter = MajorityVoting(iou_threshold=0.5, min_votes=2)
```

**Parameter tuning**:
- `min_votes=2`: More detections, some false positives
- `min_votes=majority`: Balanced precision/recall
- `min_votes=all`: High precision, low recall
- `iou_threshold`: 0.3 (small objects) to 0.7 (large objects)

#### Weighted Voting
**When to use**: When models have different quality levels.

```python
from detection_fusion.strategies import WeightedVoting

# Automatic model weighting based on average confidence
weighted_voter = WeightedVoting(iou_threshold=0.5, use_model_weights=True)

# Equal weighting (confidence only)
equal_voter = WeightedVoting(use_model_weights=False)
```

**Best practices**:
- Works well when model performance varies significantly
- Ensure confidence scores are well-calibrated
- Monitor for overconfidence bias

### 2. NMS-Based Strategies

#### Standard NMS
**When to use**: Fast processing, high recall scenarios.

```python
from detection_fusion.strategies import NMSStrategy

# Standard NMS
nms = NMSStrategy(iou_threshold=0.5, score_threshold=0.1)

# Aggressive suppression for crowded scenes
aggressive_nms = NMSStrategy(iou_threshold=0.3, score_threshold=0.3)
```

#### Confidence-Weighted NMS
**When to use**: When you need refined bounding box localization.

```python
from detection_fusion.strategies import ConfidenceWeightedNMS

# Higher confidence gets more influence in box regression
conf_nms = ConfidenceWeightedNMS(iou_threshold=0.5, confidence_power=2.0)

# More extreme weighting
extreme_conf_nms = ConfidenceWeightedNMS(confidence_power=3.0)
```

**Confidence power effects**:
- `power=1.0`: Linear weighting
- `power=2.0`: Quadratic weighting (recommended)
- `power>2.0`: Extreme weighting toward high confidence

#### Affirmative NMS
**When to use**: Balance between speed and multi-model validation.

```python
from detection_fusion.strategies import AffirmativeNMS

# Require 2+ models per detection
affirmative = AffirmativeNMS(iou_threshold=0.5, min_models=2)

# Very conservative: require 3+ models
strict_affirmative = AffirmativeNMS(iou_threshold=0.6, min_models=3)
```

### 3. Clustering Strategies

#### DBSCAN Clustering
**When to use**: Variable object sizes, spatial relationships matter.

```python
from detection_fusion.strategies import DBSCANClustering

# Tight clusters for small objects
tight_clustering = DBSCANClustering(eps=0.05, min_samples=2)

# Loose clusters for large objects
loose_clustering = DBSCANClustering(eps=0.15, min_samples=3)
```

**Parameter selection**:
- `eps`: Maximum distance between cluster points
  - Small objects: 0.05-0.1
  - Large objects: 0.1-0.2
- `min_samples`: Minimum detections per cluster
  - Conservative: 3+
  - Permissive: 2

#### Centroid Clustering
**When to use**: When IoU might not capture object relationships well.

```python
from detection_fusion.strategies import CentroidClustering

# Cluster based on center distance
centroid_cluster = CentroidClustering(
    distance_threshold=0.1,  # Maximum center distance
    min_cluster_size=2       # Minimum detections per cluster
)
```

**Benefits**:
- Works with irregular object shapes
- Fast clustering based on centers only
- Good for non-rectangular objects

#### Distance-Weighted Voting
**When to use**: When spatial consistency within clusters matters.

```python
from detection_fusion.strategies import DistanceWeightedVoting

# Weight by distance to cluster centroid
distance_voter = DistanceWeightedVoting(
    iou_threshold=0.5,
    distance_weight=1.0  # Balance between distance and confidence
)
```

### 4. Confidence-Based Strategies

#### Confidence Threshold Voting
**When to use**: When models have very different confidence calibration.

```python
from detection_fusion.strategies import ConfidenceThresholdVoting

# Adaptive thresholds per model
adaptive_conf = ConfidenceThresholdVoting(
    iou_threshold=0.5,
    base_confidence=0.5,
    adaptive_threshold=True  # Use median confidence per model
)

# Fixed threshold for all models
fixed_conf = ConfidenceThresholdVoting(
    base_confidence=0.7,
    adaptive_threshold=False
)
```

#### High Confidence First
**When to use**: When you have a clear confidence hierarchy.

```python
from detection_fusion.strategies import HighConfidenceFirst

# Hierarchical confidence processing
hierarchical = HighConfidenceFirst(
    iou_threshold=0.5,
    high_conf_threshold=0.8,  # High confidence cutoff
    low_conf_threshold=0.3    # Minimum confidence to consider
)
```

### 5. Adaptive Strategies

#### Adaptive Threshold Strategy
**When to use**: Datasets with both small and large objects.

```python
from detection_fusion.strategies import AdaptiveThresholdStrategy

# Different thresholds for different object sizes
adaptive = AdaptiveThresholdStrategy(
    small_threshold=0.3,  # Permissive for small objects
    large_threshold=0.7,  # Strict for large objects
    size_cutoff=0.05      # Area threshold (5% of image)
)
```

#### Multi-Scale Strategy
**When to use**: When objects span multiple scales (tiny to large).

```python
from detection_fusion.strategies import MultiScaleStrategy

# Automatic scale-specific processing
multi_scale = MultiScaleStrategy(iou_threshold=0.5)

# Scale categories and their thresholds:
# - Tiny (<1% area): Very permissive
# - Small (1-5% area): Moderate
# - Medium (5-15% area): Standard
# - Large (>15% area): Strict
```

#### Density Adaptive Strategy
**When to use**: Scenes with varying object density.

```python
from detection_fusion.strategies import DensityAdaptiveStrategy

# Adapt strategy based on spatial density
density_adaptive = DensityAdaptiveStrategy(
    iou_threshold=0.5,
    grid_size=5,                    # 5x5 spatial grid
    high_density_threshold=10       # 10+ detections = high density
)
```

**Processing logic**:
- High density regions → Aggressive NMS
- Low density regions → Conservative majority voting

#### Consensus Ranking Strategy
**When to use**: When model ranking information is available.

```python
from detection_fusion.strategies import ConsensusRankingStrategy

# Combine ranking with confidence
ranking_strategy = ConsensusRankingStrategy(
    iou_threshold=0.5,
    rank_weight=0.5  # Balance between rank and confidence
)
```

### 6. Probabilistic Strategies

#### Soft Voting
**When to use**: When you need smooth probability estimates.

```python
from detection_fusion.strategies import SoftVoting

# Standard temperature
soft_voter = SoftVoting(iou_threshold=0.5, temperature=1.0)

# Sharpened probabilities (more decisive)
sharp_voter = SoftVoting(temperature=0.5)

# Smoothed probabilities (less decisive)  
smooth_voter = SoftVoting(temperature=2.0)
```

#### Bayesian Fusion
**When to use**: Class imbalance, principled uncertainty quantification.

```python
from detection_fusion.strategies import BayesianFusion

# Automatic prior learning from data
bayesian = BayesianFusion(iou_threshold=0.5)

# Custom class priors based on dataset knowledge
custom_priors = {
    0: 0.4,  # person - most common
    1: 0.3,  # car - common
    2: 0.2,  # bike - less common
    3: 0.1   # other - rare
}
bayesian_custom = BayesianFusion(class_priors=custom_priors)
```

## 🔧 Configuration Best Practices

### IoU Threshold Selection by Object Type

| Object Type | Recommended IoU | Strategy Examples |
|-------------|-----------------|-------------------|
| Tiny objects (<1% area) | 0.2-0.3 | `adaptive_threshold`, `multi_scale` |
| Small objects (1-5% area) | 0.3-0.4 | `confidence_threshold`, `distance_weighted` |
| Medium objects (5-15% area) | 0.5 | `majority_vote`, `weighted_vote` |
| Large objects (>15% area) | 0.6-0.7 | `affirmative_nms`, `bayesian` |
| Mixed sizes | Adaptive | `adaptive_threshold`, `multi_scale` |

### YAML Configuration Examples

#### Small Objects Configuration
```yaml
# configs/small_objects_config.yaml
ensemble:
  iou_threshold: 0.3
  strategies:
    adaptive_threshold:
      small_threshold: 0.2
      large_threshold: 0.4
      size_cutoff: 0.02
    
    multi_scale:
      iou_threshold: 0.3
    
    distance_weighted:
      iou_threshold: 0.3
      distance_weight: 1.5
```

#### High Precision Configuration
```yaml
# configs/high_precision_config.yaml
ensemble:
  iou_threshold: 0.7
  strategies:
    affirmative_nms:
      min_models: 3
      iou_threshold: 0.7
    
    bayesian:
      iou_threshold: 0.7
      class_priors:
        0: 0.4
        1: 0.3
        2: 0.3
    
    high_confidence_first:
      high_conf_threshold: 0.8
      low_conf_threshold: 0.5
```

#### Crowded Scene Configuration
```yaml
# configs/crowded_scene_config.yaml
ensemble:
  strategies:
    density_adaptive:
      grid_size: 8
      high_density_threshold: 15
    
    confidence_weighted_nms:
      iou_threshold: 0.3
      confidence_power: 2.5
    
    dbscan:
      eps: 0.08
      min_samples: 3
```

## 🚀 Strategy Workflows

### Conservative Pipeline (High Precision)
```python
# Step 1: Unanimous agreement (highest precision)
unanimous_results = ensemble.run_strategy("unanimous")

# Step 2: High confidence detections
high_conf_results = ensemble.run_strategy("high_confidence_first")

# Step 3: Bayesian fusion for remaining candidates
bayesian_results = ensemble.run_strategy("bayesian")

# Combine with priority: unanimous > high_conf > bayesian
final_results = combine_with_priority([
    unanimous_results, high_conf_results, bayesian_results
])
```

### Balanced Pipeline (Production)
```python
# Step 1: Multi-scale processing
multi_scale_results = ensemble.run_strategy("multi_scale")

# Step 2: Affirmative NMS for validation
affirmative_results = ensemble.run_strategy("affirmative_nms")

# Step 3: Distance-weighted voting for refinement
final_results = ensemble.run_strategy("distance_weighted")
```

### High Recall Pipeline (Detection First)
```python
# Step 1: Permissive NMS to catch everything
nms_results = ensemble.run_strategy("nms")

# Step 2: Confidence filtering
conf_results = ensemble.run_strategy("confidence_threshold")

# Step 3: Density-aware processing
final_results = ensemble.run_strategy("density_adaptive")
```

### Adaptive Pipeline (Context-Aware)
```python
def adaptive_pipeline(detections):
    """Context-aware ensemble pipeline."""
    
    # Analyze scene characteristics
    total_detections = sum(len(dets) for dets in detections.values())
    avg_size = calculate_average_object_size(detections)
    density = calculate_spatial_density(detections)
    
    # Choose strategy based on scene
    if total_detections > 100 and density > 0.5:
        # Crowded scene
        return ensemble.run_strategy("density_adaptive")
    elif avg_size < 0.05:
        # Small objects
        return ensemble.run_strategy("multi_scale")
    elif len(detections) > 5:
        # Many models
        return ensemble.run_strategy("consensus_ranking")
    else:
        # Default
        return ensemble.run_strategy("weighted_vote")
```

## 📈 Performance Optimization

### Strategy Performance Characteristics

| Strategy | Speed | Memory | Precision | Recall | Use Case |
|----------|-------|--------|-----------|---------|----------|
| `nms` | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Speed critical |
| `majority_vote` | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Balanced |
| `affirmative_nms` | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Production |
| `dbscan` | ⚡ | ⚡ | ⭐⭐ | ⭐⭐⭐ | Complex scenes |
| `bayesian` | ⚡ | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Scientific |
| `multi_scale` | ⚡ | ⚡ | ⭐⭐⭐ | ⭐⭐⭐ | Multi-resolution |
| `consensus_ranking` | ⚡ | ⚡ | ⭐⭐⭐ | ⭐⭐ | Ranked models |

### Speed Optimization Tips

1. **Pre-filtering by confidence**:
```python
# Filter before ensemble
filtered_detections = {
    model: [d for d in dets if d.confidence > 0.1]
    for model, dets in detections.items()
}
```

2. **Strategy ordering** (fast → slow):
```python
# Run fast strategies first for quick feedback
strategies = [
    "nms",                    # Fastest
    "majority_vote_2", 
    "affirmative_nms",
    "distance_weighted",
    "multi_scale",           # Moderate
    "dbscan",
    "bayesian",              # Slowest
    "consensus_ranking"
]
```

3. **Batch processing for large datasets**:
```python
def process_large_dataset(detections, strategy, batch_size=1000):
    """Process large datasets in batches."""
    results = []
    models = list(detections.keys())
    
    # Calculate total detections per model
    model_lengths = {model: len(dets) for model, dets in detections.items()}
    max_length = max(model_lengths.values())
    
    for start_idx in range(0, max_length, batch_size):
        batch_detections = {}
        for model in models:
            end_idx = min(start_idx + batch_size, len(detections[model]))
            batch_detections[model] = detections[model][start_idx:end_idx]
        
        batch_results = strategy.merge(batch_detections)
        results.extend(batch_results)
    
    return results
```

## 🎛️ Parameter Tuning Guidelines

### Grid Search for Strategy Parameters
```python
import itertools
from detection_fusion.utils import load_yaml_config

def grid_search_strategy(strategy_class, param_grid, test_data):
    """Grid search for optimal strategy parameters."""
    
    best_score = 0
    best_params = {}
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_combo in itertools.product(*param_values):
        params = dict(zip(param_names, param_combo))
        
        # Test strategy with these parameters
        strategy = strategy_class(**params)
        results = strategy.merge(test_data)
        
        # Evaluate (replace with your metric)
        score = evaluate_results(results)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score

# Example usage
param_grid = {
    'iou_threshold': [0.3, 0.5, 0.7],
    'min_votes': [2, 3, 4],
    'distance_weight': [0.5, 1.0, 1.5]
}

best_params, score = grid_search_strategy(
    DistanceWeightedVoting, param_grid, test_detections
)
```

### Bayesian Optimization for Complex Strategies
```python
# For strategies with many parameters, use Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer

def objective_function(params):
    """Objective function for optimization."""
    strategy = MultiScaleStrategy(
        iou_threshold=params[0],
        size_cutoffs=params[1:4]
    )
    results = strategy.merge(validation_data)
    return -evaluate_results(results)  # Minimize negative score

# Define search space
search_space = [
    Real(0.3, 0.7, name='iou_threshold'),
    Real(0.005, 0.02, name='tiny_threshold'),
    Real(0.02, 0.08, name='small_threshold'),
    Real(0.08, 0.2, name='medium_threshold')
]

# Run optimization
result = gp_minimize(objective_function, search_space, n_calls=50)
optimal_params = result.x
```

## 🔬 Strategy Debugging and Analysis

### Debugging Strategy Behavior
```python
def debug_strategy_behavior(strategy, detections, debug_level=1):
    """Debug strategy behavior with detailed logging."""
    
    print(f"=== Debugging {strategy.name} ===")
    
    # Input analysis
    total_input = sum(len(dets) for dets in detections.values())
    print(f"Input: {total_input} detections from {len(detections)} models")
    
    if debug_level >= 2:
        for model, dets in detections.items():
            avg_conf = np.mean([d.confidence for d in dets]) if dets else 0
            print(f"  {model}: {len(dets)} detections, avg conf: {avg_conf:.3f}")
    
    # Run strategy with timing
    import time
    start_time = time.time()
    results = strategy.merge(detections)
    end_time = time.time()
    
    # Output analysis
    print(f"Output: {len(results)} detections")
    print(f"Processing time: {end_time - start_time:.3f}s")
    print(f"Reduction ratio: {len(results)/total_input:.2f}")
    
    if debug_level >= 2 and results:
        avg_conf_out = np.mean([d.confidence for d in results])
        print(f"Average output confidence: {avg_conf_out:.3f}")
        
        # Class distribution
        from collections import Counter
        class_dist = Counter([d.class_id for d in results])
        print(f"Class distribution: {dict(class_dist)}")
    
    return results

# Usage
debug_results = debug_strategy_behavior(
    ensemble.strategies["multi_scale"], 
    detections, 
    debug_level=2
)
```

### Comparative Strategy Analysis
```python
def compare_strategies_detailed(strategies, detections):
    """Detailed comparison of multiple strategies."""
    
    results = {}
    metrics = {}
    
    print("Strategy Comparison Analysis")
    print("=" * 60)
    
    for strategy_name in strategies:
        if strategy_name in ensemble.strategies:
            strategy = ensemble.strategies[strategy_name]
            
            # Run strategy
            strategy_results = strategy.merge(detections)
            results[strategy_name] = strategy_results
            
            # Calculate metrics
            if strategy_results:
                metrics[strategy_name] = {
                    'count': len(strategy_results),
                    'avg_confidence': np.mean([d.confidence for d in strategy_results]),
                    'std_confidence': np.std([d.confidence for d in strategy_results]),
                    'coverage': calculate_spatial_coverage(strategy_results),
                    'avg_size': np.mean([d.w * d.h for d in strategy_results])
                }
            else:
                metrics[strategy_name] = {
                    'count': 0, 'avg_confidence': 0, 'std_confidence': 0,
                    'coverage': 0, 'avg_size': 0
                }
    
    # Print comparison table
    print(f"{'Strategy':<20} {'Count':<8} {'Avg Conf':<10} {'Coverage':<10} {'Avg Size':<10}")
    print("-" * 60)
    
    for strategy_name, metric in metrics.items():
        print(f"{strategy_name:<20} {metric['count']:<8} "
              f"{metric['avg_confidence']:<10.3f} {metric['coverage']:<10.3f} "
              f"{metric['avg_size']:<10.4f}")
    
    return results, metrics

def calculate_spatial_coverage(detections):
    """Calculate spatial coverage of detections."""
    if not detections:
        return 0.0
    
    # Calculate bounding box of all detections
    x_coords = [d.x for d in detections]
    y_coords = [d.y for d in detections]
    
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    return x_range * y_range
```

## 🏆 Strategy Combinations and Pipelines

### Ensemble of Ensembles
```python
def hierarchical_ensemble(detections):
    """Multi-level ensemble processing."""
    
    # Level 1: Quick filtering strategies
    level1_strategies = ["nms", "confidence_threshold", "high_confidence_first"]
    level1_results = {}
    
    for strategy_name in level1_strategies:
        results = ensemble.run_strategy(strategy_name)
        level1_results[strategy_name] = results
    
    # Level 2: Advanced processing on filtered results
    level2_input = {}
    for strategy_name, results in level1_results.items():
        level2_input[f"filtered_{strategy_name}"] = results
    
    # Apply advanced strategies to filtered results
    level2_strategies = ["bayesian", "consensus_ranking"]
    final_results = []
    
    for strategy_name in level2_strategies:
        strategy = ensemble.strategies[strategy_name]
        results = strategy.merge(level2_input)
        final_results.extend(results)
    
    # Final deduplication
    dedup_strategy = NMSStrategy(iou_threshold=0.5)
    final_detections = dedup_strategy.merge({"ensemble": final_results})
    
    return final_detections
```

### Context-Aware Strategy Selection
```python
def smart_strategy_selection(detections):
    """Automatically select best strategy based on data characteristics."""
    
    # Analyze input characteristics
    total_detections = sum(len(dets) for dets in detections.values())
    num_models = len(detections)
    
    # Calculate scene characteristics
    all_dets = [d for dets in detections.values() for d in dets]
    if not all_dets:
        return []
    
    avg_confidence = np.mean([d.confidence for d in all_dets])
    avg_size = np.mean([d.w * d.h for d in all_dets])
    size_variance = np.var([d.w * d.h for d in all_dets])
    
    # Decision logic
    if size_variance > 0.01:  # High size variation
        selected_strategy = "multi_scale"
    elif avg_size < 0.05:  # Small objects
        selected_strategy = "adaptive_threshold"
    elif total_detections > 200:  # Crowded scene
        selected_strategy = "density_adaptive"
    elif avg_confidence < 0.5:  # Low confidence models
        selected_strategy = "confidence_threshold"
    elif num_models > 5:  # Many models
        selected_strategy = "consensus_ranking"
    else:  # Default case
        selected_strategy = "weighted_vote"
    
    print(f"Auto-selected strategy: {selected_strategy}")
    print(f"  Reason: avg_size={avg_size:.3f}, size_var={size_variance:.3f}, "
          f"total_dets={total_detections}, models={num_models}")
    
    return ensemble.run_strategy(selected_strategy)
```

This comprehensive strategy guide provides everything needed to select, configure, and optimize ensemble strategies for any object detection scenario. The 17+ available strategies cover every use case from speed-critical applications to high-precision scientific work.