"""Tests for ConfigLoader."""

import pytest
from pathlib import Path

from detection_fusion.config import (
    ConfigLoader,
    StrategyConfig,
    RectificationConfig,
)


class TestConfigLoaderStrategyConfig:
    """Tests for loading StrategyConfig."""

    def test_from_yaml(self, tmp_path):
        """Test loading StrategyConfig from YAML file."""
        yaml_content = """
overlap:
  threshold: 0.6
  method: "giou"
voting:
  min_votes: 3
  use_weights: false
confidence:
  min_threshold: 0.2
  temperature: 0.8
extra:
  custom_param: 42
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = ConfigLoader.from_yaml(config_file)

        assert isinstance(config, StrategyConfig)
        assert config.overlap.threshold == 0.6
        assert config.overlap.method == "giou"
        assert config.voting.min_votes == 3
        assert config.voting.use_weights is False
        assert config.confidence.min_threshold == 0.2
        assert config.confidence.temperature == 0.8
        assert config.extra["custom_param"] == 42

    def test_from_yaml_partial(self, tmp_path):
        """Test loading partial config uses defaults."""
        yaml_content = """
overlap:
  threshold: 0.7
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = ConfigLoader.from_yaml(config_file)

        assert config.overlap.threshold == 0.7
        assert config.overlap.method == "iou"  # Default
        assert config.voting.min_votes == 2  # Default
        assert config.confidence.min_threshold == 0.1  # Default

    def test_from_yaml_empty(self, tmp_path):
        """Test loading empty YAML returns defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = ConfigLoader.from_yaml(config_file)

        assert isinstance(config, StrategyConfig)
        assert config.overlap.threshold == 0.5

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            "overlap": {"threshold": 0.6},
            "voting": {"min_votes": 4}
        }

        config = ConfigLoader.from_dict(data)

        assert config.overlap.threshold == 0.6
        assert config.voting.min_votes == 4


class TestConfigLoaderRectificationConfig:
    """Tests for loading RectificationConfig."""

    def test_load_rectification(self, tmp_path):
        """Test loading RectificationConfig from YAML file."""
        yaml_content = """
mode: "maximize_error"
paths:
  labels_dir: "my_labels"
  gt_dir: "my_gt"
  images_dir: "my_images"
  output_dir: "my_output"
thresholds:
  iou: 0.4
  confidence: 0.3
  min_agreement: 2
output:
  most_correct: 100
  most_incorrect: 200
  copy_images: false
"""
        config_file = tmp_path / "rectify.yaml"
        config_file.write_text(yaml_content)

        config = ConfigLoader.load_rectification(config_file)

        assert isinstance(config, RectificationConfig)
        assert config.mode == "maximize_error"
        assert config.paths.labels_dir == "my_labels"
        assert config.paths.gt_dir == "my_gt"
        assert config.paths.images_dir == "my_images"
        assert config.paths.output_dir == "my_output"
        assert config.thresholds.iou == 0.4
        assert config.thresholds.confidence == 0.3
        assert config.thresholds.min_agreement == 2
        assert config.output.most_correct == 100
        assert config.output.most_incorrect == 200
        assert config.output.copy_images is False

    def test_load_rectification_partial(self, tmp_path):
        """Test loading partial rectification config uses defaults."""
        yaml_content = """
mode: "minimize_error"
thresholds:
  iou: 0.6
"""
        config_file = tmp_path / "rectify.yaml"
        config_file.write_text(yaml_content)

        config = ConfigLoader.load_rectification(config_file)

        assert config.mode == "minimize_error"
        assert config.thresholds.iou == 0.6
        assert config.thresholds.confidence == 0.5  # Default
        assert config.thresholds.min_agreement == 3  # Default
        assert config.paths.labels_dir == "labels"  # Default
        assert config.output.most_correct == 50  # Default

    def test_load_rectification_empty(self, tmp_path):
        """Test loading empty YAML returns defaults."""
        config_file = tmp_path / "rectify.yaml"
        config_file.write_text("")

        config = ConfigLoader.load_rectification(config_file)

        assert isinstance(config, RectificationConfig)
        assert config.mode == "minimize_error"

    def test_rectification_from_dict(self):
        """Test creating RectificationConfig from dict."""
        data = {
            "mode": "maximize_error",
            "thresholds": {"iou": 0.4}
        }

        config = ConfigLoader.rectification_from_dict(data)

        assert config.mode == "maximize_error"
        assert config.thresholds.iou == 0.4


class TestConfigLoaderActualConfigs:
    """Tests that actual config files in configs/ are loadable."""

    @pytest.fixture
    def configs_dir(self):
        """Get the configs directory path."""
        return Path(__file__).parent.parent.parent / "configs"

    def test_load_ensemble_default(self, configs_dir):
        """Test loading ensemble/default.yaml."""
        config_file = configs_dir / "ensemble" / "default.yaml"
        if config_file.exists():
            config = ConfigLoader.from_yaml(config_file)
            assert isinstance(config, StrategyConfig)

    def test_load_ensemble_high_precision(self, configs_dir):
        """Test loading ensemble/high_precision.yaml."""
        config_file = configs_dir / "ensemble" / "high_precision.yaml"
        if config_file.exists():
            config = ConfigLoader.from_yaml(config_file)
            assert isinstance(config, StrategyConfig)

    def test_load_rectification_balanced(self, configs_dir):
        """Test loading gt_rectification/balanced.yaml."""
        config_file = configs_dir / "gt_rectification" / "balanced.yaml"
        if config_file.exists():
            config = ConfigLoader.load_rectification(config_file)
            assert isinstance(config, RectificationConfig)

    def test_load_rectification_conservative(self, configs_dir):
        """Test loading gt_rectification/conservative.yaml."""
        config_file = configs_dir / "gt_rectification" / "conservative.yaml"
        if config_file.exists():
            config = ConfigLoader.load_rectification(config_file)
            assert isinstance(config, RectificationConfig)

    def test_load_rectification_aggressive(self, configs_dir):
        """Test loading gt_rectification/aggressive.yaml."""
        config_file = configs_dir / "gt_rectification" / "aggressive.yaml"
        if config_file.exists():
            config = ConfigLoader.load_rectification(config_file)
            assert isinstance(config, RectificationConfig)
