"""
Strategy registry for discovery and instantiation.

Provides a central registry for all ensemble strategies, enabling:
- Strategy discovery (list_all, list_by_category)
- Factory-based instantiation (create by name)
- Automatic registration via decorator
- Strategy metadata retrieval (get_metadata, get_info)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from .base import BaseStrategy, StrategyMetadata
    from .params import StrategyConfig


@dataclass
class StrategyInfo:
    """Complete information about a registered strategy."""

    name: str
    category: str
    description: str
    supports_ground_truth: bool
    class_name: str
    module: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "supports_ground_truth": self.supports_ground_truth,
            "class_name": self.class_name,
            "module": self.module,
            "parameters": self.parameters,
        }


class StrategyRegistry:
    """Central registry for all ensemble strategies.

    Provides strategy discovery, instantiation, and categorization.
    Strategies register themselves using the @register decorator.

    Example:
        # Register a strategy (decorator)
        @StrategyRegistry.register
        class MyStrategy(BaseStrategy):
            ...

        # Create a strategy instance
        strategy = StrategyRegistry.create("my_strategy")
        strategy = StrategyRegistry.create("my_strategy", iou_threshold=0.6)

        # List available strategies
        all_names = StrategyRegistry.list_all()
        voting_strategies = StrategyRegistry.list_by_category("voting")

        # Get metadata
        meta = StrategyRegistry.get_metadata("bayesian")
    """

    _strategies: Dict[str, Type["BaseStrategy"]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, strategy_class: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
        """Decorator to register a strategy class.

        The strategy class must have a 'name' property that returns
        the strategy's identifier string.

        Args:
            strategy_class: Strategy class to register

        Returns:
            The same class (for decorator chaining)
        """
        # Create a temporary instance to get the name
        # This works because BaseStrategy.__init__ only requires iou_threshold
        try:
            temp_instance = strategy_class.__new__(strategy_class)
            # Call parent init if needed
            if hasattr(strategy_class, "__init__"):
                # Try to get name from class or instance
                if hasattr(strategy_class, "strategy_name"):
                    name = strategy_class.strategy_name
                elif hasattr(temp_instance, "name"):
                    # Some strategies define name as property
                    name = strategy_class.__name__.lower()
                    # Convert CamelCase to snake_case
                    import re

                    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", strategy_class.__name__)
                    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
                else:
                    name = strategy_class.__name__.lower()
            else:
                name = strategy_class.__name__.lower()
        except Exception:
            # Fallback: use class name
            import re

            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", strategy_class.__name__)
            name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        cls._strategies[name] = strategy_class
        return strategy_class

    @classmethod
    def register_with_name(cls, name: str):
        """Decorator to register a strategy with a specific name.

        Args:
            name: Name to register the strategy under

        Returns:
            Decorator function
        """

        def decorator(strategy_class: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def create(
        cls, name: str, config: Optional["StrategyConfig"] = None, **kwargs
    ) -> "BaseStrategy":
        """Create a strategy instance by name.

        Args:
            name: Strategy name (e.g., "weighted_vote", "bayesian")
            config: Optional StrategyConfig for configuration
            **kwargs: Additional arguments passed to strategy constructor

        Returns:
            Configured strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        cls._ensure_initialized()

        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")

        strategy_class = cls._strategies[name]

        # Build kwargs from config if provided
        if config is not None:
            kwargs.setdefault("iou_threshold", config.overlap.threshold)

        return strategy_class(**kwargs)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy names
        """
        cls._ensure_initialized()
        return sorted(cls._strategies.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List strategies in a specific category.

        Categories are derived from strategy metadata:
        - voting: MajorityVoting, WeightedVoting
        - nms: NMSStrategy, AffirmativeNMS
        - clustering: DBSCANClustering
        - probabilistic: SoftVoting, BayesianFusion
        - distance_based: DistanceWeightedVoting, CentroidClustering
        - confidence_based: ConfidenceThresholdVoting, etc.
        - adaptive: AdaptiveThresholdStrategy, etc.

        Args:
            category: Category name

        Returns:
            List of strategy names in that category
        """
        cls._ensure_initialized()

        result = []
        for name, strategy_class in cls._strategies.items():
            # Check metadata first
            if hasattr(strategy_class, "metadata") and strategy_class.metadata:
                if strategy_class.metadata.category == category:
                    result.append(name)
                    continue

            # Fallback: try to match by module name
            module = strategy_class.__module__
            if category in module:
                result.append(name)

        return sorted(result)

    @classmethod
    def get_class(cls, name: str) -> Type["BaseStrategy"]:
        """Get the strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy name is unknown
        """
        cls._ensure_initialized()

        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: '{name}'")
        return cls._strategies[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy name

        Returns:
            True if registered, False otherwise
        """
        cls._ensure_initialized()
        return name in cls._strategies

    @classmethod
    def count(cls) -> int:
        """Get the number of registered strategies.

        Returns:
            Number of registered strategies
        """
        cls._ensure_initialized()
        return len(cls._strategies)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies (useful for testing)."""
        cls._strategies.clear()
        cls._initialized = False

    @classmethod
    def get_metadata(cls, name: str) -> Optional["StrategyMetadata"]:
        """Get metadata for a strategy.

        Args:
            name: Strategy name

        Returns:
            StrategyMetadata if available, None otherwise
        """
        cls._ensure_initialized()
        strategy_cls = cls._strategies.get(name)
        if strategy_cls and hasattr(strategy_cls, "metadata"):
            return strategy_cls.metadata
        return None

    @classmethod
    def get_info(cls, name: str) -> StrategyInfo:
        """Get complete information about a strategy.

        Args:
            name: Strategy name

        Returns:
            StrategyInfo with complete strategy details

        Raises:
            ValueError: If strategy name is unknown
        """
        cls._ensure_initialized()

        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: '{name}'")

        strategy_cls = cls._strategies[name]
        metadata = getattr(strategy_cls, "metadata", None)

        # Extract parameter info from __init__
        import inspect

        params = {}
        if hasattr(strategy_cls, "__init__"):
            sig = inspect.signature(strategy_cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs"):
                    continue
                param_info = {"default": None, "required": True}
                if param.default is not inspect.Parameter.empty:
                    param_info["default"] = param.default
                    param_info["required"] = False
                if param.annotation is not inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                params[param_name] = param_info

        return StrategyInfo(
            name=name,
            category=metadata.category if metadata else "unknown",
            description=metadata.description if metadata else (strategy_cls.__doc__ or ""),
            supports_ground_truth=metadata.supports_ground_truth if metadata else True,
            class_name=strategy_cls.__name__,
            module=strategy_cls.__module__,
            parameters=params,
        )

    @classmethod
    def list_with_metadata(cls) -> List[Dict[str, Any]]:
        """List all strategies with their metadata.

        Returns:
            List of dicts with name, category, description
        """
        cls._ensure_initialized()
        result = []
        for name in sorted(cls._strategies.keys()):
            metadata = cls.get_metadata(name)
            result.append(
                {
                    "name": name,
                    "category": metadata.category if metadata else "unknown",
                    "description": metadata.description if metadata else "",
                }
            )
        return result

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure strategies are loaded into the registry."""
        if cls._initialized:
            return

        # Import strategy modules to trigger registration
        # This is done lazily to avoid circular imports
        cls._load_builtin_strategies()
        cls._initialized = True

    @classmethod
    def _load_builtin_strategies(cls) -> None:
        """Load and register all built-in strategies."""
        # Import all strategy modules - this triggers their registration
        # if they use the @register decorator

        # Manual registration for strategies that don't use the decorator yet
        # This ensures backwards compatibility during the migration
        from .adaptive import (
            AdaptiveThresholdStrategy,
            ConsensusRankingStrategy,
            DensityAdaptiveStrategy,
            MultiScaleStrategy,
        )
        from .clustering import DBSCANClustering
        from .confidence_based import (
            ConfidenceThresholdVoting,
            ConfidenceWeightedNMS,
            HighConfidenceFirst,
        )
        from .distance_based import CentroidClustering, DistanceWeightedVoting
        from .nms import AffirmativeNMS, NMSStrategy
        from .probabilistic import BayesianFusion, SoftVoting
        from .voting import MajorityVoting, WeightedVoting

        # Register with canonical names
        strategy_mapping = {
            # Voting strategies
            "majority_vote": MajorityVoting,
            "majority_vote_2": lambda **kw: MajorityVoting(min_votes=2, **kw),
            "majority_vote_3": lambda **kw: MajorityVoting(min_votes=3, **kw),
            "weighted_vote": WeightedVoting,
            "unanimous": lambda **kw: MajorityVoting(min_votes=999, **kw),
            # NMS strategies
            "nms": NMSStrategy,
            "affirmative_nms": AffirmativeNMS,
            # Clustering strategies
            "dbscan": DBSCANClustering,
            # Probabilistic strategies
            "soft_voting": SoftVoting,
            "bayesian": BayesianFusion,
            # Distance-based strategies
            "distance_weighted": DistanceWeightedVoting,
            "centroid_clustering": CentroidClustering,
            # Confidence-based strategies
            "confidence_threshold": ConfidenceThresholdVoting,
            "confidence_weighted_nms": ConfidenceWeightedNMS,
            "high_confidence_first": HighConfidenceFirst,
            # Adaptive strategies
            "adaptive_threshold": AdaptiveThresholdStrategy,
            "density_adaptive": DensityAdaptiveStrategy,
            "multi_scale": MultiScaleStrategy,
            "consensus_ranking": ConsensusRankingStrategy,
        }

        for name, strategy in strategy_mapping.items():
            if callable(strategy) and not isinstance(strategy, type):
                # It's a factory function, skip for now
                continue
            cls._strategies[name] = strategy


# Convenience function for creating strategies
def create_strategy(name: str, **kwargs) -> "BaseStrategy":
    """Create a strategy instance by name.

    This is a convenience function wrapping StrategyRegistry.create().

    Args:
        name: Strategy name
        **kwargs: Arguments passed to strategy constructor

    Returns:
        Strategy instance
    """
    return StrategyRegistry.create(name, **kwargs)


def list_strategies() -> List[str]:
    """List all available strategy names.

    Returns:
        Sorted list of strategy names
    """
    return StrategyRegistry.list_all()


def get_strategy_info(name: str) -> StrategyInfo:
    """Get complete information about a strategy.

    Args:
        name: Strategy name

    Returns:
        StrategyInfo with complete strategy details
    """
    return StrategyRegistry.get_info(name)


def list_strategies_with_metadata() -> List[Dict[str, Any]]:
    """List all strategies with their metadata.

    Returns:
        List of dicts with name, category, description
    """
    return StrategyRegistry.list_with_metadata()


__all__ = [
    "StrategyRegistry",
    "StrategyInfo",
    "create_strategy",
    "list_strategies",
    "get_strategy_info",
    "list_strategies_with_metadata",
]
