"""Tests for data loader module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from detection_fusion import Detection, DetectionSet
from detection_fusion.data.loader import FileDetectionLoader, load_detections


class TestFileDetectionLoader:
    """Tests for FileDetectionLoader class."""

    def test_init_default_excluded(self, tmp_path):
        """Test default excluded directories."""
        loader = FileDetectionLoader(str(tmp_path))
        assert "unified" in loader._excluded
        assert "GT" in loader._excluded
        assert "__pycache__" in loader._excluded

    def test_init_custom_excluded(self, tmp_path):
        """Test custom excluded directories."""
        loader = FileDetectionLoader(str(tmp_path), excluded_dirs={"custom"})
        assert "custom" in loader._excluded
        assert "unified" not in loader._excluded

    def test_find_all_models(self, tmp_path):
        """Test discovering model directories."""
        (tmp_path / "model1").mkdir()
        (tmp_path / "model2").mkdir()
        (tmp_path / "unified").mkdir()  # Should be excluded
        (tmp_path / "some_file.txt").touch()  # Not a directory

        loader = FileDetectionLoader(str(tmp_path))
        models = loader.find_all_models()

        assert "model1" in models
        assert "model2" in models
        assert "unified" not in models
        assert len(models) == 2

    def test_find_all_models_sorted(self, tmp_path):
        """Test that models are returned sorted."""
        (tmp_path / "zebra").mkdir()
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()

        loader = FileDetectionLoader(str(tmp_path))
        models = loader.find_all_models()

        assert models == ["alpha", "beta", "zebra"]

    def test_load_nonexistent_model(self, tmp_path):
        """Test loading from non-existent model returns empty list."""
        loader = FileDetectionLoader(str(tmp_path))
        result = loader.load("nonexistent")
        assert result == []

    def test_load_as_set(self, yolo_labels_dir):
        """Test loading all models as DetectionSet."""
        # Force YOLO format since auto-detect checks root dir which has no .txt files
        loader = FileDetectionLoader(str(yolo_labels_dir), format="yolo")
        detection_set = loader.load_as_set()

        assert isinstance(detection_set, DetectionSet)
        assert detection_set.total_count > 0


class TestLoadDetections:
    """Tests for load_detections convenience function."""

    def test_load_with_model_name(self, yolo_labels_dir):
        """Test that model_name is set on loaded detections."""
        model_dir = list(yolo_labels_dir.iterdir())[0]
        if model_dir.is_dir():
            dets = load_detections(str(model_dir), model_name="test_model")
            if dets:
                assert all(d.model_source == "test_model" for d in dets)

    def test_load_single_file(self, tmp_path):
        """Test loading from a single YOLO file."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2 0.9\n1 0.3 0.3 0.1 0.1 0.8\n")

        dets = load_detections(str(label_file), format="yolo")
        assert len(dets) == 2
        assert dets[0].class_id == 0
        assert dets[1].class_id == 1
