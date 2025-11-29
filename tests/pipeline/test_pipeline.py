"""Tests for detection pipeline."""

from detection_fusion import DetectionSet
from detection_fusion.config import StrategyConfig
from detection_fusion.pipeline import DetectionPipeline
from detection_fusion.pipeline.context import PipelineContext
from detection_fusion.pipeline.stages import PipelineStage


class MockStage(PipelineStage):
    """Mock stage for testing."""

    def __init__(self, name: str):
        self.name = name
        self.processed = False

    def process(self, context: PipelineContext):
        self.processed = True
        context.metadata[self.name] = True
        return None


class TestDetectionPipeline:
    """Tests for DetectionPipeline class."""

    def test_initial_state(self):
        """Test pipeline initial state."""
        pipeline = DetectionPipeline()
        assert len(pipeline._stages) == 0
        assert pipeline.context is not None

    def test_fluent_interface(self, yolo_labels_dir):
        """Test fluent interface returns self."""
        pipeline = DetectionPipeline()

        result = pipeline.load(str(yolo_labels_dir), format="yolo")
        assert result is pipeline

        result = pipeline.ensemble("weighted_vote")
        assert result is pipeline

    def test_with_config(self):
        """Test setting config."""
        config = StrategyConfig().with_overlap(threshold=0.7)
        pipeline = DetectionPipeline().with_config(config)

        assert pipeline.context.config is config
        assert pipeline.context.config.overlap.threshold == 0.7

    def test_add_stage(self):
        """Test adding custom stage."""
        pipeline = DetectionPipeline()
        stage = MockStage("test")

        result = pipeline.add_stage(stage)
        assert result is pipeline
        assert len(pipeline._stages) == 1

    def test_run_executes_stages(self):
        """Test run executes all stages in order."""
        pipeline = DetectionPipeline()
        stage1 = MockStage("stage1")
        stage2 = MockStage("stage2")

        pipeline.add_stage(stage1).add_stage(stage2)
        ctx = pipeline.run()

        assert stage1.processed
        assert stage2.processed
        assert ctx.metadata.get("stage1") is True
        assert ctx.metadata.get("stage2") is True

    def test_run_returns_context(self):
        """Test run returns context."""
        pipeline = DetectionPipeline()
        stage = MockStage("test")
        pipeline.add_stage(stage)

        result = pipeline.run()
        assert isinstance(result, PipelineContext)

    def test_context_property(self):
        """Test context property access."""
        pipeline = DetectionPipeline()
        ctx = pipeline.context

        assert isinstance(ctx, PipelineContext)
        assert ctx is pipeline._context


class TestPipelineIntegration:
    """Integration tests for pipeline with real stages."""

    def test_load_stage(self, yolo_labels_dir):
        """Test load stage populates detections."""
        pipeline = DetectionPipeline()
        ctx = pipeline.load(str(yolo_labels_dir), format="yolo").run()

        assert ctx.detections is not None
        assert isinstance(ctx.detections, DetectionSet)

    def test_chained_load_ensemble(self, yolo_labels_dir):
        """Test chained load and ensemble stages."""
        pipeline = DetectionPipeline()

        ctx = pipeline.load(str(yolo_labels_dir), format="yolo").ensemble("weighted_vote").run()

        assert ctx.detections is not None
        assert ctx.ensemble_result is not None
        assert isinstance(ctx.ensemble_result, list)
