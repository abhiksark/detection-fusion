"""
Ensemble strategies for object detection fusion.

This module provides 19 strategies for combining detections from multiple models:

Voting strategies:
- MajorityVoting: Keep detections where multiple models agree
- WeightedVoting: Weight by model and detection confidence

NMS strategies:
- NMSStrategy: Standard Non-Maximum Suppression
- AffirmativeNMS: NMS requiring agreement from multiple models

Clustering strategies:
- DBSCANClustering: Density-based spatial clustering

Probabilistic strategies:
- SoftVoting: Probabilistic voting with temperature scaling
- BayesianFusion: Bayesian inference with learned class priors

Distance-based strategies:
- DistanceWeightedVoting: Weights by spatial distance to cluster centroid
- CentroidClustering: Agglomerative clustering based on detection centers

Confidence-based strategies:
- ConfidenceThresholdVoting: Adaptive confidence thresholds per model
- ConfidenceWeightedNMS: NMS with confidence-weighted box regression
- HighConfidenceFirst: Prioritizes high-confidence detections

Adaptive strategies:
- AdaptiveThresholdStrategy: Different IoU thresholds for object sizes
- DensityAdaptiveStrategy: Context-aware processing for detection density
- MultiScaleStrategy: Scale-specific processing
- ConsensusRankingStrategy: Combines model ranking with confidence

Usage:
    # Direct instantiation
    from detection_fusion.strategies import MajorityVoting
    strategy = MajorityVoting(iou_threshold=0.5, min_votes=2)

    # Via registry
    from detection_fusion.strategies import StrategyRegistry
    strategy = StrategyRegistry.create("majority_vote", iou_threshold=0.5)
    available = StrategyRegistry.list_all()
"""

# Base classes and infrastructure
from .adaptive import (
    AdaptiveThresholdStrategy,
    ConsensusRankingStrategy,
    DensityAdaptiveStrategy,
    MultiScaleStrategy,
)
from .base import BaseStrategy, StrategyMetadata
from .clustering import DBSCANClustering
from .confidence_based import ConfidenceThresholdVoting, ConfidenceWeightedNMS, HighConfidenceFirst
from .distance_based import CentroidClustering, DistanceWeightedVoting
from .mixins import (
    BoxMergingMixin,
    ClassVotingMixin,
    ClusteringMixin,
    DetectionUtilsMixin,
    FullStrategyMixin,
    ModelWeightsMixin,
)
from .nms import AffirmativeNMS, NMSStrategy
from .params import (
    DEFAULT_CONFIG,
    HIGH_PRECISION_CONFIG,
    HIGH_RECALL_CONFIG,
    ClusteringParams,
    ConfidenceParams,
    OverlapParams,
    StrategyConfig,
    VotingParams,
)
from .probabilistic import BayesianFusion, SoftVoting
from .registry import StrategyRegistry, create_strategy, list_strategies

# Strategy implementations
from .voting import MajorityVoting, WeightedVoting

__all__ = [
    # Base classes
    "BaseStrategy",
    "StrategyMetadata",
    # Configuration
    "StrategyConfig",
    "OverlapParams",
    "VotingParams",
    "ConfidenceParams",
    "ClusteringParams",
    "DEFAULT_CONFIG",
    "HIGH_PRECISION_CONFIG",
    "HIGH_RECALL_CONFIG",
    # Mixins
    "ClusteringMixin",
    "ModelWeightsMixin",
    "BoxMergingMixin",
    "ClassVotingMixin",
    "DetectionUtilsMixin",
    "FullStrategyMixin",
    # Registry
    "StrategyRegistry",
    "create_strategy",
    "list_strategies",
    # Strategy implementations
    "MajorityVoting",
    "WeightedVoting",
    "NMSStrategy",
    "AffirmativeNMS",
    "DBSCANClustering",
    "SoftVoting",
    "BayesianFusion",
    "DistanceWeightedVoting",
    "CentroidClustering",
    "ConfidenceThresholdVoting",
    "ConfidenceWeightedNMS",
    "HighConfidenceFirst",
    "AdaptiveThresholdStrategy",
    "DensityAdaptiveStrategy",
    "MultiScaleStrategy",
    "ConsensusRankingStrategy",
]
