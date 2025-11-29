"""Tests for config models."""

import pytest
from pydantic import ValidationError

from detection_fusion.config import (
    OverlapConfig,
    VotingConfig,
    ConfidenceConfig,
    StrategyConfig,
    RectificationConfig,
    RectificationPathsConfig,
    RectificationThresholdsConfig,
    RectificationOutputConfig,
)


class TestOverlapConfig:
    """Tests for OverlapConfig."""

    def test_defaults(self):
        """Test default values."""
        config = OverlapConfig()
        assert config.threshold == 0.5
        assert config.method == "iou"

    def test_custom_values(self):
        """Test custom values."""
        config = OverlapConfig(threshold=0.7, method="giou")
        assert config.threshold == 0.7
        assert config.method == "giou"

    def test_validation_threshold_range(self):
        """Test threshold must be in [0, 1]."""
        with pytest.raises(ValidationError):
            OverlapConfig(threshold=1.5)

        with pytest.raises(ValidationError):
            OverlapConfig(threshold=-0.1)


class TestVotingConfig:
    """Tests for VotingConfig."""

    def test_defaults(self):
        """Test default values."""
        config = VotingConfig()
        assert config.min_votes == 2
        assert config.use_weights is True

    def test_custom_values(self):
        """Test custom values."""
        config = VotingConfig(min_votes=3, use_weights=False)
        assert config.min_votes == 3
        assert config.use_weights is False

    def test_validation_min_votes(self):
        """Test min_votes must be >= 1."""
        with pytest.raises(ValidationError):
            VotingConfig(min_votes=0)


class TestConfidenceConfig:
    """Tests for ConfidenceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ConfidenceConfig()
        assert config.min_threshold == 0.1
        assert config.temperature == 1.0

    def test_validation_temperature(self):
        """Test temperature must be > 0."""
        with pytest.raises(ValidationError):
            ConfidenceConfig(temperature=0)

        with pytest.raises(ValidationError):
            ConfidenceConfig(temperature=-1)


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_defaults(self):
        """Test default nested configs."""
        config = StrategyConfig()
        assert config.overlap.threshold == 0.5
        assert config.voting.min_votes == 2
        assert config.confidence.min_threshold == 0.1
        assert config.extra == {}

    def test_with_overlap(self):
        """Test with_overlap builder method."""
        config = StrategyConfig()
        new_config = config.with_overlap(threshold=0.7)

        assert new_config.overlap.threshold == 0.7
        assert config.overlap.threshold == 0.5  # Original unchanged

    def test_with_voting(self):
        """Test with_voting builder method."""
        config = StrategyConfig()
        new_config = config.with_voting(min_votes=3, use_weights=False)

        assert new_config.voting.min_votes == 3
        assert new_config.voting.use_weights is False

    def test_with_confidence(self):
        """Test with_confidence builder method."""
        config = StrategyConfig()
        new_config = config.with_confidence(min_threshold=0.2)

        assert new_config.confidence.min_threshold == 0.2

    def test_with_extra(self):
        """Test with_extra builder method."""
        config = StrategyConfig()
        new_config = config.with_extra(custom_param=42)

        assert new_config.extra["custom_param"] == 42
        assert config.extra == {}  # Original unchanged

    def test_chained_builders(self):
        """Test chaining builder methods."""
        config = (
            StrategyConfig()
            .with_overlap(threshold=0.6)
            .with_voting(min_votes=3)
            .with_extra(param1="value1")
        )

        assert config.overlap.threshold == 0.6
        assert config.voting.min_votes == 3
        assert config.extra["param1"] == "value1"


class TestRectificationPathsConfig:
    """Tests for RectificationPathsConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RectificationPathsConfig()
        assert config.labels_dir == "labels"
        assert config.gt_dir == "GT"
        assert config.images_dir == "images"
        assert config.output_dir == "rectified_dataset"

    def test_custom_values(self):
        """Test custom values."""
        config = RectificationPathsConfig(
            labels_dir="my_labels",
            gt_dir="my_gt",
            images_dir="my_images",
            output_dir="my_output"
        )
        assert config.labels_dir == "my_labels"
        assert config.gt_dir == "my_gt"
        assert config.images_dir == "my_images"
        assert config.output_dir == "my_output"


class TestRectificationThresholdsConfig:
    """Tests for RectificationThresholdsConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RectificationThresholdsConfig()
        assert config.iou == 0.5
        assert config.confidence == 0.5
        assert config.min_agreement == 3

    def test_custom_values(self):
        """Test custom values."""
        config = RectificationThresholdsConfig(
            iou=0.6,
            confidence=0.7,
            min_agreement=4
        )
        assert config.iou == 0.6
        assert config.confidence == 0.7
        assert config.min_agreement == 4

    def test_validation_iou_range(self):
        """Test iou must be in [0, 1]."""
        with pytest.raises(ValidationError):
            RectificationThresholdsConfig(iou=1.5)
        with pytest.raises(ValidationError):
            RectificationThresholdsConfig(iou=-0.1)

    def test_validation_confidence_range(self):
        """Test confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            RectificationThresholdsConfig(confidence=1.5)
        with pytest.raises(ValidationError):
            RectificationThresholdsConfig(confidence=-0.1)

    def test_validation_min_agreement(self):
        """Test min_agreement must be >= 1."""
        with pytest.raises(ValidationError):
            RectificationThresholdsConfig(min_agreement=0)


class TestRectificationOutputConfig:
    """Tests for RectificationOutputConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RectificationOutputConfig()
        assert config.most_correct == 50
        assert config.most_incorrect == 50
        assert config.copy_images is True

    def test_custom_values(self):
        """Test custom values."""
        config = RectificationOutputConfig(
            most_correct=100,
            most_incorrect=200,
            copy_images=False
        )
        assert config.most_correct == 100
        assert config.most_incorrect == 200
        assert config.copy_images is False

    def test_validation_most_correct(self):
        """Test most_correct must be >= 0."""
        with pytest.raises(ValidationError):
            RectificationOutputConfig(most_correct=-1)

    def test_validation_most_incorrect(self):
        """Test most_incorrect must be >= 0."""
        with pytest.raises(ValidationError):
            RectificationOutputConfig(most_incorrect=-1)


class TestRectificationConfig:
    """Tests for RectificationConfig."""

    def test_defaults(self):
        """Test default nested configs."""
        config = RectificationConfig()
        assert config.mode == "minimize_error"
        assert config.paths.labels_dir == "labels"
        assert config.thresholds.iou == 0.5
        assert config.output.most_correct == 50

    def test_with_paths(self):
        """Test with_paths builder method."""
        config = RectificationConfig()
        new_config = config.with_paths(labels_dir="my_labels", output_dir="my_output")

        assert new_config.paths.labels_dir == "my_labels"
        assert new_config.paths.output_dir == "my_output"
        assert config.paths.labels_dir == "labels"  # Original unchanged

    def test_with_thresholds(self):
        """Test with_thresholds builder method."""
        config = RectificationConfig()
        new_config = config.with_thresholds(iou=0.6, min_agreement=4)

        assert new_config.thresholds.iou == 0.6
        assert new_config.thresholds.min_agreement == 4

    def test_with_output(self):
        """Test with_output builder method."""
        config = RectificationConfig()
        new_config = config.with_output(most_correct=100, copy_images=False)

        assert new_config.output.most_correct == 100
        assert new_config.output.copy_images is False

    def test_chained_builders(self):
        """Test chaining builder methods."""
        config = (
            RectificationConfig()
            .with_paths(labels_dir="my_labels")
            .with_thresholds(iou=0.7)
            .with_output(most_correct=200)
        )

        assert config.paths.labels_dir == "my_labels"
        assert config.thresholds.iou == 0.7
        assert config.output.most_correct == 200

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            "mode": "maximize_error",
            "paths": {
                "labels_dir": "custom_labels",
                "gt_dir": "custom_gt"
            },
            "thresholds": {
                "iou": 0.4,
                "confidence": 0.3
            }
        }
        config = RectificationConfig.model_validate(data)

        assert config.mode == "maximize_error"
        assert config.paths.labels_dir == "custom_labels"
        assert config.paths.gt_dir == "custom_gt"
        assert config.thresholds.iou == 0.4
        assert config.thresholds.confidence == 0.3
        # Defaults for unspecified values
        assert config.paths.images_dir == "images"
        assert config.thresholds.min_agreement == 3
