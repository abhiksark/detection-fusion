"""Tests for Detection class."""

import pytest
from pydantic import ValidationError

from detection_fusion import Detection


class TestDetection:
    """Tests for Detection class."""

    def test_create_detection(self):
        """Test basic detection creation."""
        det = Detection(
            class_id=0,
            x=0.5,
            y=0.5,
            w=0.1,
            h=0.1,
            confidence=0.9,
        )
        assert det.class_id == 0
        assert det.x == 0.5
        assert det.confidence == 0.9

    def test_detection_immutable(self):
        """Test that Detection is immutable (frozen)."""
        det = Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1)
        with pytest.raises(ValidationError):
            det.x = 0.6

    def test_detection_bbox(self, sample_detection):
        """Test bbox property."""
        bbox = sample_detection.bbox
        assert bbox == [0.5, 0.5, 0.1, 0.1]

    def test_detection_xyxy(self):
        """Test xyxy conversion."""
        det = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.2)
        xyxy = det.xyxy
        assert xyxy == pytest.approx([0.4, 0.4, 0.6, 0.6])

    def test_detection_center(self, sample_detection):
        """Test center property."""
        assert sample_detection.center == (0.5, 0.5)

    def test_detection_area(self):
        """Test area calculation."""
        det = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.3)
        assert det.area == pytest.approx(0.06)

    def test_detection_with_confidence(self, sample_detection):
        """Test with_confidence factory method."""
        new_det = sample_detection.with_confidence(0.5)
        assert new_det.confidence == 0.5
        assert sample_detection.confidence == 0.9  # Original unchanged

    def test_detection_with_source(self, sample_detection):
        """Test with_source factory method."""
        new_det = sample_detection.with_source("new_model")
        assert new_det.model_source == "new_model"

    def test_detection_with_image(self, sample_detection):
        """Test with_image factory method."""
        new_det = sample_detection.with_image("new_image")
        assert new_det.image_name == "new_image"

    def test_detection_iou_same_box(self):
        """Test IoU of identical boxes."""
        det = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.2)
        assert det.iou_with(det) == pytest.approx(1.0)

    def test_detection_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        det1 = Detection(class_id=0, x=0.1, y=0.1, w=0.1, h=0.1)
        det2 = Detection(class_id=0, x=0.9, y=0.9, w=0.1, h=0.1)
        assert det1.iou_with(det2) == 0.0

    def test_detection_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        det1 = Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.2)
        det2 = Detection(class_id=0, x=0.55, y=0.55, w=0.2, h=0.2)
        iou = det1.iou_with(det2)
        assert 0 < iou < 1

    def test_detection_to_dict(self, sample_detection):
        """Test to_dict serialization."""
        d = sample_detection.to_dict()
        assert d["class_id"] == 0
        assert d["bbox"] == [0.5, 0.5, 0.1, 0.1]
        assert d["confidence"] == 0.9

    def test_detection_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "class_id": 1,
            "bbox": [0.3, 0.4, 0.2, 0.15],
            "confidence": 0.75,
            "model_source": "test",
        }
        det = Detection.from_dict(data)
        assert det.class_id == 1
        assert det.x == 0.3
        assert det.y == 0.4
        assert det.w == 0.2
        assert det.h == 0.15
        assert det.confidence == 0.75

    def test_detection_hash(self):
        """Test that Detection is hashable."""
        det1 = Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1)
        det2 = Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1)
        det3 = Detection(class_id=0, x=0.51, y=0.5, w=0.1, h=0.1)

        # Same detections should have same hash
        assert hash(det1) == hash(det2)

        # Can be used in sets
        det_set = {det1, det2, det3}
        assert len(det_set) == 2

    def test_detection_validation_class_id(self):
        """Test that class_id must be non-negative."""
        with pytest.raises(ValidationError):
            Detection(class_id=-1, x=0.5, y=0.5, w=0.1, h=0.1)

    def test_detection_validation_coordinates(self):
        """Test that coordinates must be in [0, 1]."""
        with pytest.raises(ValidationError):
            Detection(class_id=0, x=1.5, y=0.5, w=0.1, h=0.1)

        with pytest.raises(ValidationError):
            Detection(class_id=0, x=-0.1, y=0.5, w=0.1, h=0.1)

    def test_detection_validation_confidence(self):
        """Test that confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1, confidence=1.5)

    def test_detection_default_confidence(self):
        """Test default confidence value."""
        det = Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1)
        assert det.confidence == 1.0
