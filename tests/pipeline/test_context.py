"""Tests for pipeline context."""

import pytest

from detection_fusion import Detection, DetectionSet
from detection_fusion.pipeline.context import PipelineContext, EvaluationResult


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_default_values(self):
        """Test default values are zeros."""
        result = EvaluationResult()
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.mAP == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.per_class_metrics == {}

    def test_custom_values(self):
        """Test setting custom values."""
        result = EvaluationResult(
            precision=0.85,
            recall=0.90,
            f1_score=0.87,
            mAP=0.82,
            true_positives=100,
            false_positives=20,
            false_negatives=10,
        )
        assert result.precision == 0.85
        assert result.recall == 0.90
        assert result.true_positives == 100


class TestPipelineContext:
    """Tests for PipelineContext class."""

    def test_initial_state(self):
        """Test initial context state."""
        ctx = PipelineContext()
        assert ctx.detections is None
        assert ctx.ground_truth is None
        assert ctx.ensemble_result is None
        assert ctx.evaluation_result is None
        assert ctx.config is None
        assert ctx.metadata == {}

    def test_set_detections(self, multi_model_detections):
        """Test setting detections."""
        ctx = PipelineContext()
        ds = DetectionSet(multi_model_detections)
        ctx.set_detections(ds)

        assert ctx.detections is ds
        assert ctx.detections.total_count == 5

    def test_set_ground_truth(self):
        """Test setting ground truth."""
        ctx = PipelineContext()
        gt = [
            Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1, image_name="img1"),
            Detection(class_id=1, x=0.3, y=0.3, w=0.2, h=0.2, image_name="img1"),
        ]
        ctx.set_ground_truth(gt)

        assert ctx.ground_truth is gt
        assert len(ctx.ground_truth) == 2

    def test_set_ensemble_result(self):
        """Test setting ensemble result."""
        ctx = PipelineContext()
        result = [
            Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1),
        ]
        ctx.set_ensemble_result(result)

        assert ctx.ensemble_result is result

    def test_set_evaluation_result(self):
        """Test setting evaluation result."""
        ctx = PipelineContext()
        eval_result = EvaluationResult(precision=0.9, recall=0.85)
        ctx.set_evaluation_result(eval_result)

        assert ctx.evaluation_result is eval_result
        assert ctx.evaluation_result.precision == 0.9

    def test_metadata_storage(self):
        """Test storing metadata."""
        ctx = PipelineContext()
        ctx.metadata["strategy"] = "weighted_vote"
        ctx.metadata["threshold"] = 0.5

        assert ctx.metadata["strategy"] == "weighted_vote"
        assert ctx.metadata["threshold"] == 0.5
