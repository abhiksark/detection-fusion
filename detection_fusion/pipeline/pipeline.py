from typing import List

from detection_fusion.config import StrategyConfig

from .context import PipelineContext
from .stages import EnsembleStage, EvaluationStage, LoadStage, PipelineStage


class DetectionPipeline:
    """Fluent pipeline builder for chaining ensemble and evaluation."""

    def __init__(self):
        self._stages: List[PipelineStage] = []
        self._context = PipelineContext()

    def load(self, path: str, format: str = "auto") -> "DetectionPipeline":
        self._stages.append(LoadStage(path, format=format))
        return self

    def ensemble(self, strategy: str, **kwargs) -> "DetectionPipeline":
        self._stages.append(EnsembleStage(strategy, **kwargs))
        return self

    def evaluate(self, gt_path: str, iou_threshold: float = 0.5) -> "DetectionPipeline":
        self._stages.append(EvaluationStage(gt_path, iou_threshold=iou_threshold))
        return self

    def with_config(self, config: StrategyConfig) -> "DetectionPipeline":
        self._context.config = config
        return self

    def add_stage(self, stage: PipelineStage) -> "DetectionPipeline":
        self._stages.append(stage)
        return self

    def run(self) -> PipelineContext:
        for stage in self._stages:
            stage.process(self._context)
        return self._context

    @property
    def context(self) -> PipelineContext:
        return self._context
