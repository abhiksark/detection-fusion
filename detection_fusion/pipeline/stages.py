from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar

from detection_fusion.core.detection import Detection
from detection_fusion.core.detection_set import DetectionSet
from detection_fusion.data import FileDetectionLoader, GroundTruthRepository

from .context import EvaluationResult, PipelineContext

T = TypeVar("T")


class PipelineStage(ABC, Generic[T]):
    """Base class for pipeline stages."""

    @abstractmethod
    def process(self, context: PipelineContext) -> T: ...


class LoadStage(PipelineStage[DetectionSet]):
    """Stage for loading detections from filesystem."""

    def __init__(
        self,
        path: str,
        format: str = "auto",
    ):
        self._path = path
        self._format = format

    def process(self, context: PipelineContext) -> DetectionSet:
        loader = FileDetectionLoader(self._path, format=self._format)
        detection_set = loader.load_as_set()
        context.set_detections(detection_set)
        return detection_set


class EnsembleStage(PipelineStage[List[Detection]]):
    """Stage for running ensemble strategy."""

    def __init__(self, strategy_name: str, **kwargs):
        self._strategy_name = strategy_name
        self._kwargs = kwargs

    def process(self, context: PipelineContext) -> List[Detection]:
        if context.detections is None:
            raise ValueError("No detections loaded. Run LoadStage first.")

        # Import here to avoid circular imports
        from detection_fusion.strategies import StrategyRegistry

        # Merge config from context with kwargs
        config = context.config
        strategy = StrategyRegistry.create(self._strategy_name, config=config, **self._kwargs)

        result = strategy.merge(context.detections.raw_data)
        context.set_ensemble_result(result)
        return result


class EvaluationStage(PipelineStage[EvaluationResult]):
    """Stage for evaluating against ground truth."""

    def __init__(
        self,
        gt_path: str,
        iou_threshold: float = 0.5,
    ):
        self._gt_path = gt_path
        self._iou_threshold = iou_threshold

    def process(self, context: PipelineContext) -> EvaluationResult:
        if context.ensemble_result is None:
            raise ValueError("No ensemble result. Run EnsembleStage first.")

        gt_repo = GroundTruthRepository(self._gt_path)
        ground_truth = gt_repo.load()
        context.set_ground_truth(ground_truth)

        result = self._evaluate(
            context.ensemble_result,
            ground_truth,
            self._iou_threshold,
        )
        context.set_evaluation_result(result)
        return result

    def _evaluate(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection],
        iou_threshold: float,
    ) -> EvaluationResult:
        # Group by image
        pred_by_image: Dict[str, List[Detection]] = {}
        for det in predictions:
            if det.image_name not in pred_by_image:
                pred_by_image[det.image_name] = []
            pred_by_image[det.image_name].append(det)

        gt_by_image: Dict[str, List[Detection]] = {}
        for det in ground_truth:
            if det.image_name not in gt_by_image:
                gt_by_image[det.image_name] = []
            gt_by_image[det.image_name].append(det)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        all_images = set(pred_by_image.keys()) | set(gt_by_image.keys())

        for image_name in all_images:
            preds = pred_by_image.get(image_name, [])
            gts = gt_by_image.get(image_name, [])

            matched_gt = set()

            # Sort predictions by confidence
            preds_sorted = sorted(preds, key=lambda d: d.confidence, reverse=True)

            for pred in preds_sorted:
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if idx in matched_gt:
                        continue
                    if pred.class_id != gt.class_id:
                        continue

                    iou = pred.iou_with(gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += len(gts) - len(matched_gt)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mAP=precision,  # Simplified - proper mAP requires AP curve
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )
