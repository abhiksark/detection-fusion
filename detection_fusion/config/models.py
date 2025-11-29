from typing import Any, Dict

from pydantic import BaseModel, Field

from .defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MIN_VOTES,
    DEFAULT_TEMPERATURE,
)


class OverlapConfig(BaseModel):
    threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)
    method: str = Field(default="iou")


class VotingConfig(BaseModel):
    min_votes: int = Field(default=DEFAULT_MIN_VOTES, ge=1)
    use_weights: bool = Field(default=True)


class ConfidenceConfig(BaseModel):
    min_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0.0)


class StrategyConfig(BaseModel):
    overlap: OverlapConfig = Field(default_factory=OverlapConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def with_overlap(self, **kwargs) -> "StrategyConfig":
        new_overlap = self.overlap.model_copy(update=kwargs)
        return self.model_copy(update={"overlap": new_overlap})

    def with_voting(self, **kwargs) -> "StrategyConfig":
        new_voting = self.voting.model_copy(update=kwargs)
        return self.model_copy(update={"voting": new_voting})

    def with_confidence(self, **kwargs) -> "StrategyConfig":
        new_confidence = self.confidence.model_copy(update=kwargs)
        return self.model_copy(update={"confidence": new_confidence})

    def with_extra(self, **kwargs) -> "StrategyConfig":
        new_extra = {**self.extra, **kwargs}
        return self.model_copy(update={"extra": new_extra})


# Rectification Configuration Models


class RectificationPathsConfig(BaseModel):
    """Paths configuration for GT rectification."""

    labels_dir: str = Field(default="labels")
    gt_dir: str = Field(default="GT")
    images_dir: str = Field(default="images")
    output_dir: str = Field(default="rectified_dataset")


class RectificationThresholdsConfig(BaseModel):
    """Threshold configuration for GT rectification."""

    iou: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_agreement: int = Field(default=3, ge=1)


class RectificationOutputConfig(BaseModel):
    """Output configuration for GT rectification."""

    most_correct: int = Field(default=50, ge=0)
    most_incorrect: int = Field(default=50, ge=0)
    copy_images: bool = Field(default=True)


class RectificationConfig(BaseModel):
    """Configuration for GT rectification operations.

    Supports both 'minimize_error' (conservative) and 'maximize_error'
    (aggressive) modes for ground truth correction.
    """

    mode: str = Field(default="minimize_error")
    paths: RectificationPathsConfig = Field(default_factory=RectificationPathsConfig)
    thresholds: RectificationThresholdsConfig = Field(default_factory=RectificationThresholdsConfig)
    output: RectificationOutputConfig = Field(default_factory=RectificationOutputConfig)

    def with_paths(self, **kwargs) -> "RectificationConfig":
        """Return a copy with updated paths configuration."""
        new_paths = self.paths.model_copy(update=kwargs)
        return self.model_copy(update={"paths": new_paths})

    def with_thresholds(self, **kwargs) -> "RectificationConfig":
        """Return a copy with updated thresholds configuration."""
        new_thresholds = self.thresholds.model_copy(update=kwargs)
        return self.model_copy(update={"thresholds": new_thresholds})

    def with_output(self, **kwargs) -> "RectificationConfig":
        """Return a copy with updated output configuration."""
        new_output = self.output.model_copy(update=kwargs)
        return self.model_copy(update={"output": new_output})
