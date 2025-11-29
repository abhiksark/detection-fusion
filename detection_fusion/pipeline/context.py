from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from detection_fusion.config import StrategyConfig
from detection_fusion.core.detection import Detection
from detection_fusion.core.detection_set import DetectionSet


@dataclass
class EvaluationResult:
    """Result of evaluation against ground truth."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mAP: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_class_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Shared state passed through pipeline stages."""

    detections: Optional[DetectionSet] = None
    ground_truth: Optional[List[Detection]] = None
    ensemble_result: Optional[List[Detection]] = None
    evaluation_result: Optional[EvaluationResult] = None
    config: Optional[StrategyConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_detections(self, detections: DetectionSet) -> None:
        self.detections = detections

    def set_ground_truth(self, ground_truth: List[Detection]) -> None:
        self.ground_truth = ground_truth

    def set_ensemble_result(self, result: List[Detection]) -> None:
        self.ensemble_result = result

    def set_evaluation_result(self, result: EvaluationResult) -> None:
        self.evaluation_result = result
