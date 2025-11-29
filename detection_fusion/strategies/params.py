"""
Strategy parameter configuration classes.

Provides standardized configuration for all ensemble strategies,
replacing inconsistent parameter naming across strategy files.

Includes:
- Parameter dataclasses (OverlapParams, VotingParams, etc.)
- StrategyConfig for full configuration
- ParamSchema for runtime parameter validation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class OverlapParams:
    """Parameters for detection overlap/clustering.

    Attributes:
        threshold: IoU or distance threshold for overlap detection
        method: Overlap method ("iou", "center_distance", "dbscan")
    """

    threshold: float = 0.5
    method: str = "iou"


@dataclass(frozen=True)
class VotingParams:
    """Parameters for voting-based strategies.

    Attributes:
        min_votes: Minimum number of models that must agree
        use_weights: Whether to use model weights in voting
    """

    min_votes: int = 2
    use_weights: bool = True


@dataclass(frozen=True)
class ConfidenceParams:
    """Parameters for confidence handling.

    Attributes:
        min_threshold: Minimum confidence to consider a detection
        temperature: Temperature for softmax-based methods
    """

    min_threshold: float = 0.1
    temperature: float = 1.0


@dataclass(frozen=True)
class ClusteringParams:
    """Parameters specific to clustering strategies.

    Attributes:
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples for DBSCAN
    """

    eps: float = 0.1
    min_samples: int = 2


@dataclass
class StrategyConfig:
    """Complete strategy configuration.

    Combines all parameter groups into a single configuration object.
    Strategies can access relevant parameters through this unified interface.

    Example:
        config = StrategyConfig(
            overlap=OverlapParams(threshold=0.6),
            voting=VotingParams(min_votes=3)
        )
        strategy = MajorityVoting(config=config)
    """

    overlap: OverlapParams = field(default_factory=OverlapParams)
    voting: VotingParams = field(default_factory=VotingParams)
    confidence: ConfidenceParams = field(default_factory=ConfidenceParams)
    clustering: ClusteringParams = field(default_factory=ClusteringParams)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StrategyConfig":
        """Create StrategyConfig from a dictionary.

        Args:
            config_dict: Dictionary with parameter values

        Returns:
            StrategyConfig instance
        """
        overlap_dict = config_dict.get("overlap", {})
        voting_dict = config_dict.get("voting", {})
        confidence_dict = config_dict.get("confidence", {})
        clustering_dict = config_dict.get("clustering", {})
        extra = config_dict.get("extra", {})

        return cls(
            overlap=OverlapParams(**overlap_dict) if overlap_dict else OverlapParams(),
            voting=VotingParams(**voting_dict) if voting_dict else VotingParams(),
            confidence=ConfidenceParams(**confidence_dict)
            if confidence_dict
            else ConfidenceParams(),
            clustering=ClusteringParams(**clustering_dict)
            if clustering_dict
            else ClusteringParams(),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overlap": {
                "threshold": self.overlap.threshold,
                "method": self.overlap.method,
            },
            "voting": {
                "min_votes": self.voting.min_votes,
                "use_weights": self.voting.use_weights,
            },
            "confidence": {
                "min_threshold": self.confidence.min_threshold,
                "temperature": self.confidence.temperature,
            },
            "clustering": {
                "eps": self.clustering.eps,
                "min_samples": self.clustering.min_samples,
            },
            "extra": self.extra,
        }

    def with_overlap(self, **kwargs) -> "StrategyConfig":
        """Return new config with modified overlap params."""
        new_overlap = OverlapParams(
            threshold=kwargs.get("threshold", self.overlap.threshold),
            method=kwargs.get("method", self.overlap.method),
        )
        return StrategyConfig(
            overlap=new_overlap,
            voting=self.voting,
            confidence=self.confidence,
            clustering=self.clustering,
            extra=self.extra,
        )

    def with_voting(self, **kwargs) -> "StrategyConfig":
        """Return new config with modified voting params."""
        new_voting = VotingParams(
            min_votes=kwargs.get("min_votes", self.voting.min_votes),
            use_weights=kwargs.get("use_weights", self.voting.use_weights),
        )
        return StrategyConfig(
            overlap=self.overlap,
            voting=new_voting,
            confidence=self.confidence,
            clustering=self.clustering,
            extra=self.extra,
        )


# Default configurations for common use cases
DEFAULT_CONFIG = StrategyConfig()

HIGH_PRECISION_CONFIG = StrategyConfig(
    overlap=OverlapParams(threshold=0.6),
    voting=VotingParams(min_votes=3),
    confidence=ConfidenceParams(min_threshold=0.5),
)

HIGH_RECALL_CONFIG = StrategyConfig(
    overlap=OverlapParams(threshold=0.3),
    voting=VotingParams(min_votes=1),
    confidence=ConfidenceParams(min_threshold=0.1),
)


class ParamSpec(BaseModel):
    """Specification for a single parameter."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Parameter name")
    param_type: str = Field(description="Type (float, int, bool, str)")
    default: Any = Field(default=None, description="Default value")
    required: bool = Field(default=False, description="Whether required")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    choices: Optional[List[Any]] = Field(default=None, description="Valid choices")
    description: str = Field(default="", description="Parameter description")

    def validate_value(self, value: Any) -> Any:
        """Validate a value against this spec.

        Args:
            value: Value to validate

        Returns:
            Validated/coerced value

        Raises:
            ValueError: If validation fails
        """
        if value is None:
            if self.required:
                raise ValueError(f"Parameter '{self.name}' is required")
            return self.default

        # Type coercion
        type_map = {"float": float, "int": int, "bool": bool, "str": str}
        if self.param_type in type_map:
            try:
                value = type_map[self.param_type](value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Parameter '{self.name}' must be {self.param_type}: {e}")

        # Range validation
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Parameter '{self.name}' must be >= {self.min_value}, got {value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Parameter '{self.name}' must be <= {self.max_value}, got {value}")

        # Choice validation
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"Parameter '{self.name}' must be one of {self.choices}, got {value}")

        return value


class ParamSchema(BaseModel):
    """Schema for validating strategy parameters.

    Defines what parameters a strategy accepts and validates them.

    Example:
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float",
                         default=0.5, min_value=0, max_value=1),
                ParamSpec(name="min_votes", param_type="int",
                         default=2, min_value=1),
            ]
        )
        validated = schema.validate({"iou_threshold": 0.6, "min_votes": 3})
    """

    model_config = ConfigDict(frozen=True)

    params: List[ParamSpec] = Field(default_factory=list)

    def validate(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against schema.

        Args:
            kwargs: Parameters to validate

        Returns:
            Dict with validated parameters (including defaults)

        Raises:
            ValueError: If validation fails
        """
        result = {}
        param_names = {p.name for p in self.params}

        # Validate known parameters
        for spec in self.params:
            value = kwargs.get(spec.name)
            result[spec.name] = spec.validate_value(value)

        # Check for unknown parameters
        unknown = set(kwargs.keys()) - param_names
        if unknown:
            # Allow 'config' as it's a standard parameter
            unknown = unknown - {"config"}
            if unknown:
                raise ValueError(f"Unknown parameters: {unknown}")

        return result

    def get_defaults(self) -> Dict[str, Any]:
        """Get all default values.

        Returns:
            Dict of parameter name to default value
        """
        return {spec.name: spec.default for spec in self.params}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "params": [
                {
                    "name": p.name,
                    "type": p.param_type,
                    "default": p.default,
                    "required": p.required,
                    "min": p.min_value,
                    "max": p.max_value,
                    "choices": p.choices,
                    "description": p.description,
                }
                for p in self.params
            ]
        }


# Common parameter schemas for strategy categories
VOTING_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for detection overlap",
        ),
        ParamSpec(
            name="min_votes",
            param_type="int",
            default=2,
            min_value=1,
            description="Minimum models that must agree",
        ),
        ParamSpec(
            name="use_weights",
            param_type="bool",
            default=True,
            description="Whether to use model weights",
        ),
    ]
)

NMS_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for suppression",
        ),
        ParamSpec(
            name="confidence_threshold",
            param_type="float",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence to keep",
        ),
    ]
)

CLUSTERING_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for clustering",
        ),
        ParamSpec(
            name="eps",
            param_type="float",
            default=0.1,
            min_value=0.0,
            description="DBSCAN epsilon parameter",
        ),
        ParamSpec(
            name="min_samples",
            param_type="int",
            default=2,
            min_value=1,
            description="Minimum samples per cluster",
        ),
    ]
)

PROBABILISTIC_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for overlap",
        ),
        ParamSpec(
            name="temperature",
            param_type="float",
            default=1.0,
            min_value=0.01,
            description="Temperature for probability scaling",
        ),
    ]
)

ADAPTIVE_SCHEMA = ParamSchema(
    params=[
        ParamSpec(
            name="iou_threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="Base IoU threshold",
        ),
        ParamSpec(
            name="small_threshold",
            param_type="float",
            default=0.3,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for small objects",
        ),
        ParamSpec(
            name="large_threshold",
            param_type="float",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            description="IoU threshold for large objects",
        ),
    ]
)


def get_schema_for_category(category: str) -> ParamSchema:
    """Get the parameter schema for a strategy category.

    Args:
        category: Category name (voting, nms, clustering, etc.)

    Returns:
        ParamSchema for the category
    """
    schemas = {
        "voting": VOTING_SCHEMA,
        "nms": NMS_SCHEMA,
        "clustering": CLUSTERING_SCHEMA,
        "probabilistic": PROBABILISTIC_SCHEMA,
        "adaptive": ADAPTIVE_SCHEMA,
    }
    return schemas.get(category, ParamSchema())


__all__ = [
    "OverlapParams",
    "VotingParams",
    "ConfidenceParams",
    "ClusteringParams",
    "StrategyConfig",
    "DEFAULT_CONFIG",
    "HIGH_PRECISION_CONFIG",
    "HIGH_RECALL_CONFIG",
    "ParamSpec",
    "ParamSchema",
    "VOTING_SCHEMA",
    "NMS_SCHEMA",
    "CLUSTERING_SCHEMA",
    "PROBABILISTIC_SCHEMA",
    "ADAPTIVE_SCHEMA",
    "get_schema_for_category",
]
