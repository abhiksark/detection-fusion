"""Tests for YOLO format reader/writer."""

import pytest
from pathlib import Path

from detection_fusion import Detection
from detection_fusion.data.formats.yolo import YOLOReader, YOLOWriter


class TestYOLOReader:
    """Tests for YOLOReader class."""

    def test_can_read_txt_file(self, tmp_path):
        """Test detection of YOLO .txt files."""
        txt_file = tmp_path / "labels.txt"
        txt_file.write_text("0 0.5 0.5 0.2 0.2\n")
        assert YOLOReader.can_read(tmp_path)

    def test_can_read_empty_dir(self, tmp_path):
        """Test detection fails on empty directory."""
        assert not YOLOReader.can_read(tmp_path)

    def test_read_file_basic(self, tmp_path):
        """Test reading basic YOLO format."""
        label_file = tmp_path / "image1.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        reader = YOLOReader()
        detections = reader.read_file(label_file)

        assert len(detections) == 2
        assert detections[0].class_id == 0
        assert detections[0].x == pytest.approx(0.5)
        assert detections[0].y == pytest.approx(0.5)
        assert detections[0].w == pytest.approx(0.2)
        assert detections[0].h == pytest.approx(0.2)

    def test_read_file_with_confidence(self, tmp_path):
        """Test reading YOLO format with confidence scores."""
        label_file = tmp_path / "image1.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2 0.95\n")

        reader = YOLOReader()
        detections = reader.read_file(label_file)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.95)

    def test_read_file_image_name(self, tmp_path):
        """Test that image name is set from filename."""
        label_file = tmp_path / "my_image.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\n")

        reader = YOLOReader()
        detections = reader.read_file(label_file)

        assert detections[0].image_name == "my_image"

    def test_read_file_empty(self, tmp_path):
        """Test reading empty file returns empty list."""
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")

        reader = YOLOReader()
        detections = reader.read_file(label_file)

        assert detections == []

    def test_read_directory(self, tmp_path):
        """Test reading entire directory."""
        (tmp_path / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (tmp_path / "img2.txt").write_text("1 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.15 0.15\n")

        reader = YOLOReader()
        result = reader.read_directory(tmp_path)

        assert "img1" in result
        assert "img2" in result
        assert len(result["img1"]) == 1
        assert len(result["img2"]) == 2

    def test_read_invalid_line_skipped(self, tmp_path):
        """Test invalid lines are skipped."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\ninvalid line\n1 0.3 0.3 0.1 0.1\n")

        reader = YOLOReader()
        detections = reader.read_file(label_file)

        assert len(detections) == 2


class TestYOLOWriter:
    """Tests for YOLOWriter class."""

    def test_write_file_basic(self, tmp_path):
        """Test writing basic YOLO format."""
        detections = [
            Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.2, confidence=0.9),
            Detection(class_id=1, x=0.3, y=0.3, w=0.1, h=0.1, confidence=0.8),
        ]

        output_file = tmp_path / "output.txt"
        writer = YOLOWriter()
        writer.write_file(detections, output_file)

        content = output_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        parts = lines[0].split()
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.5)

    def test_write_directory(self, tmp_path):
        """Test writing to directory."""
        detections_by_image = {
            "img1": [Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.2)],
            "img2": [
                Detection(class_id=1, x=0.3, y=0.3, w=0.1, h=0.1),
                Detection(class_id=2, x=0.7, y=0.7, w=0.15, h=0.15),
            ],
        }

        output_dir = tmp_path / "output"
        writer = YOLOWriter()
        writer.write_directory(detections_by_image, output_dir)

        assert (output_dir / "img1.txt").exists()
        assert (output_dir / "img2.txt").exists()

        img2_content = (output_dir / "img2.txt").read_text()
        assert len(img2_content.strip().split("\n")) == 2

    def test_write_empty_detections(self, tmp_path):
        """Test writing empty detection list."""
        output_file = tmp_path / "empty.txt"
        writer = YOLOWriter()
        writer.write_file([], output_file)

        assert output_file.exists()
        assert output_file.read_text() == ""
