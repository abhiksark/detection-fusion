"""
Base strategy class for all ensemble strategies.

All strategies must inherit from BaseStrategy and implement the merge() method.
Strategies can optionally use mixins for shared functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from ..core.detection import Detection

if TYPE_CHECKING:
    from .params import ParamSchema, StrategyConfig


@dataclass
class StrategyMetadata:
    """Metadata describing a strategy's characteristics.

    Attributes:
        name: Canonical name for the strategy
        category: Category (voting, nms, clustering, probabilistic, adaptive)
        description: Human-readable description
        supports_ground_truth: Whether strategy can use GT for optimization
        params_schema: Optional ParamSchema for parameter validation
    """

    name: str
    category: str
    description: str
    supports_ground_truth: bool = True
    params_schema: Optional["ParamSchema"] = None


class BaseStrategy(ABC):
    """Base class for all ensemble strategies.

    All strategies must inherit from this class and implement the merge() method.
    Strategies can optionally use mixins for shared functionality:
    - ClusteringMixin: Detection clustering by IoU
    - ModelWeightsMixin: Model weight computation
    - BoxMergingMixin: Weighted box averaging
    - ClassVotingMixin: Class voting methods

    Attributes:
        iou_threshold: IoU threshold for overlap detection
        metadata: Optional class-level metadata about the strategy

    Example:
        class MyStrategy(BaseStrategy, ClusteringMixin, BoxMergingMixin):
            metadata = StrategyMetadata(
                name="my_strategy",
                category="voting",
                description="My custom voting strategy"
            )

            def merge(self, detections, **kwargs):
                all_dets = self._flatten(detections)
                clusters = self.cluster_detections(all_dets)
                return [self.merge_cluster(c) for c in clusters]

    Note on kwargs:
        The kwargs parameter in merge() accepts runtime overrides:
        - Any strategy-specific parameter can be overridden
        - Common overrides: iou_threshold, min_votes, confidence_threshold
    """

    # Class-level metadata (optional, can be overridden by subclasses)
    metadata: ClassVar[Optional[StrategyMetadata]] = None

    def __init__(self, iou_threshold: float = 0.5, config: Optional["StrategyConfig"] = None):
        """Initialize the strategy.

        Args:
            iou_threshold: IoU threshold for overlap detection
            config: Optional StrategyConfig for full configuration
        """
        if config is not None:
            self.iou_threshold = config.overlap.threshold
            self._config = config
        else:
            self.iou_threshold = iou_threshold
            self._config = None

    @abstractmethod
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """Merge detections from multiple models.

        Args:
            detections: Dictionary mapping model names to their detections
            **kwargs: Additional strategy-specific parameters that override
                      the instance configuration

        Returns:
            List of merged detections
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name.

        Returns:
            Canonical name for this strategy (e.g., "weighted_vote")
        """
        pass

    @property
    def category(self) -> str:
        """Return strategy category.

        Returns:
            Category name (voting, nms, clustering, probabilistic, adaptive)
        """
        if self.metadata:
            return self.metadata.category
        return "unknown"

    @property
    def description(self) -> str:
        """Return strategy description.

        Returns:
            Human-readable description of what this strategy does
        """
        if self.metadata:
            return self.metadata.description
        return self.__doc__ or ""

    def _flatten(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Flatten detections from all models into a single list.

        Args:
            detections: Dict mapping model names to detection lists

        Returns:
            Single flat list of all detections
        """
        return [det for dets in detections.values() for det in dets]

    def _get_param(self, kwargs: dict, name: str, default=None):
        """Get a parameter from kwargs or instance config.

        Args:
            kwargs: Runtime kwargs dict
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value
        """
        if name in kwargs:
            return kwargs[name]
        if self._config and hasattr(self._config, name):
            return getattr(self._config, name)
        return default

    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """Validate parameters against the strategy's schema.

        If the strategy has a params_schema defined in its metadata,
        validate the provided kwargs against it. Otherwise, return
        kwargs unchanged.

        Args:
            **kwargs: Parameters to validate

        Returns:
            Dict with validated parameters (including defaults)

        Raises:
            ValueError: If validation fails
        """
        if self.metadata and self.metadata.params_schema:
            return self.metadata.params_schema.validate(kwargs)
        return kwargs

    @classmethod
    def get_params_schema(cls) -> Optional["ParamSchema"]:
        """Get the parameter schema for this strategy.

        Returns:
            ParamSchema if defined, None otherwise
        """
        if hasattr(cls, "metadata") and cls.metadata:
            return cls.metadata.params_schema
        return None

    @classmethod
    def get_param_defaults(cls) -> Dict[str, Any]:
        """Get default parameter values for this strategy.

        Returns:
            Dict of parameter name to default value
        """
        schema = cls.get_params_schema()
        if schema:
            return schema.get_defaults()
        return {}

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(iou_threshold={self.iou_threshold})"
