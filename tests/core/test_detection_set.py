"""Tests for DetectionSet class."""

from detection_fusion import Detection, DetectionSet


class TestDetectionSet:
    """Tests for DetectionSet class."""

    def test_create_detection_set(self, multi_model_detections):
        """Test basic DetectionSet creation."""
        ds = DetectionSet(multi_model_detections)
        assert len(ds.model_names) == 3
        assert ds.total_count == 5

    def test_by_model(self, detection_set):
        """Test getting detections by model."""
        model1_dets = detection_set.by_model("model1")
        assert len(model1_dets) == 2

        # Non-existent model returns empty list
        unknown_dets = detection_set.by_model("unknown")
        assert len(unknown_dets) == 0

    def test_by_image(self, multi_model_detections):
        """Test filtering by image."""
        # Add detections for another image
        multi_model_detections["model1"].append(
            Detection(class_id=2, x=0.6, y=0.6, w=0.1, h=0.1, image_name="img2")
        )
        ds = DetectionSet(multi_model_detections)

        img1_set = ds.by_image("img1")
        assert img1_set.total_count == 5

        img2_set = ds.by_image("img2")
        assert img2_set.total_count == 1

    def test_all_detections(self, detection_set):
        """Test getting all detections as flat list."""
        all_dets = detection_set.all_detections()
        assert len(all_dets) == 5
        assert all(isinstance(d, Detection) for d in all_dets)

    def test_filter_by_confidence(self, detection_set):
        """Test confidence filtering."""
        high_conf = detection_set.filter_by_confidence(0.85)
        # Should keep: 0.9, 0.85, 0.88, 0.87 (4 detections)
        assert high_conf.total_count == 4

        very_high_conf = detection_set.filter_by_confidence(0.9)
        assert very_high_conf.total_count == 1

    def test_filter_by_class(self, detection_set):
        """Test class filtering."""
        class0 = detection_set.filter_by_class([0])
        assert class0.total_count == 3

        class1 = detection_set.filter_by_class([1])
        assert class1.total_count == 2

        both = detection_set.filter_by_class([0, 1])
        assert both.total_count == 5

    def test_group_by_image(self, multi_model_detections):
        """Test grouping by image."""
        multi_model_detections["model1"].append(
            Detection(class_id=2, x=0.6, y=0.6, w=0.1, h=0.1, image_name="img2")
        )
        ds = DetectionSet(multi_model_detections)

        by_image = ds.group_by_image()
        assert "img1" in by_image
        assert "img2" in by_image
        assert by_image["img1"].total_count == 5
        assert by_image["img2"].total_count == 1

    def test_group_by_class(self, detection_set):
        """Test grouping by class."""
        by_class = detection_set.group_by_class()
        assert 0 in by_class
        assert 1 in by_class
        assert by_class[0].total_count == 3
        assert by_class[1].total_count == 2

    def test_confidence_stats(self, detection_set):
        """Test confidence statistics."""
        stats = detection_set.confidence_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] == 0.82
        assert stats["max"] == 0.9

    def test_confidence_stats_empty(self):
        """Test confidence stats on empty set."""
        ds = DetectionSet({})
        stats = ds.confidence_stats()
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_class_distribution(self, detection_set):
        """Test class distribution."""
        dist = detection_set.class_distribution()
        assert dist[0] == 3
        assert dist[1] == 2

    def test_model_names(self, detection_set):
        """Test model_names property."""
        names = detection_set.model_names
        assert set(names) == {"model1", "model2", "model3"}

    def test_image_names(self, detection_set):
        """Test image_names property."""
        names = detection_set.image_names
        assert names == ["img1"]

    def test_total_count(self, detection_set):
        """Test total_count property."""
        assert detection_set.total_count == 5

    def test_iteration(self, detection_set):
        """Test iteration over detections."""
        count = 0
        for det in detection_set:
            assert isinstance(det, Detection)
            count += 1
        assert count == 5

    def test_len(self, detection_set):
        """Test __len__ method."""
        assert len(detection_set) == 5

    def test_repr(self, detection_set):
        """Test __repr__ method."""
        repr_str = repr(detection_set)
        assert "DetectionSet" in repr_str
        assert "models=3" in repr_str
        assert "detections=5" in repr_str

    def test_raw_data(self, detection_set, multi_model_detections):
        """Test raw_data property."""
        raw = detection_set.raw_data
        assert raw == multi_model_detections

    def test_chained_filtering(self, detection_set):
        """Test chaining filter operations."""
        result = detection_set.filter_by_class([0]).filter_by_confidence(0.88)
        assert result.total_count == 2
