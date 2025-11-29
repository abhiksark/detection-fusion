"""Tests for format registry."""

import pytest

from detection_fusion.data.formats import FormatRegistry
from detection_fusion.data.formats.base import AnnotationReader, AnnotationWriter
from detection_fusion.exceptions import FormatError


class TestFormatRegistry:
    """Tests for FormatRegistry class."""

    def test_list_formats(self):
        """Test listing available formats."""
        formats = FormatRegistry.list_formats()
        assert "readers" in formats
        assert "writers" in formats
        assert "yolo" in formats["readers"]
        assert "yolo" in formats["writers"]

    def test_get_reader_yolo(self):
        """Test getting YOLO reader."""
        reader = FormatRegistry.get_reader("yolo")
        assert isinstance(reader, AnnotationReader)

    def test_get_reader_unknown(self):
        """Test getting unknown format raises error."""
        with pytest.raises(FormatError) as excinfo:
            FormatRegistry.get_reader("unknown_format")
        assert "Unknown format" in str(excinfo.value)

    def test_get_writer_yolo(self):
        """Test getting YOLO writer."""
        writer = FormatRegistry.get_writer("yolo")
        assert isinstance(writer, AnnotationWriter)

    def test_get_writer_unknown(self):
        """Test getting unknown writer raises error."""
        with pytest.raises(FormatError) as excinfo:
            FormatRegistry.get_writer("unknown_format")
        assert "Unknown format" in str(excinfo.value)

    def test_auto_detect_yolo(self, tmp_path):
        """Test auto-detecting YOLO format."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2 0.9\n")

        reader = FormatRegistry.auto_detect_reader(tmp_path)
        assert reader is not None

    def test_auto_detect_fails(self, tmp_path):
        """Test auto-detect fails on unknown format."""
        # Create an empty directory with no recognizable files
        (tmp_path / "unknown.xyz").write_text("random content")

        with pytest.raises(FormatError) as excinfo:
            FormatRegistry.auto_detect_reader(tmp_path)
        assert "Could not auto-detect" in str(excinfo.value)


class TestFormatRegistryDecorator:
    """Tests for registry decorators."""

    def test_register_reader_decorator(self):
        """Test reader registration via decorator."""

        @FormatRegistry.register_reader("test_format_reader")
        class TestReader(AnnotationReader):
            format_name = "test_format_reader"

            @classmethod
            def can_read(cls, path):
                return False

            def read_file(self, path):
                return []

            def read_directory(self, path):
                return {}

        assert "test_format_reader" in FormatRegistry._readers

        # Cleanup
        del FormatRegistry._readers["test_format_reader"]

    def test_register_writer_decorator(self):
        """Test writer registration via decorator."""

        @FormatRegistry.register_writer("test_format_writer")
        class TestWriter(AnnotationWriter):
            format_name = "test_format_writer"

            def write_file(self, detections, path):
                pass

            def write_directory(self, detections_by_image, path):
                pass

        assert "test_format_writer" in FormatRegistry._writers

        # Cleanup
        del FormatRegistry._writers["test_format_writer"]
