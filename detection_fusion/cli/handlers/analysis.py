"""
Concrete analysis handler implementations.
"""

from collections import Counter
from typing import Dict, List

import numpy as np

from ...core.detection import Detection
from ...utils.metrics import calculate_iou
from .base import AnalysisContext, AnalysisHandler, AnalysisResult
from .registry import AnalysisRegistry


@AnalysisRegistry.register("agreement")
class AgreementAnalysisHandler(AnalysisHandler):
    """Analyzes agreement between models."""

    description = "Analyze detection agreement between models"

    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze model agreement."""
        error = self.validate(context)
        if error:
            return AnalysisResult(name=self.name, success=False, error=error)

        detections = context.detections
        result_data = {
            "total_models": len(detections),
            "model_stats": {},
            "agreement_matrix": {},
            "consensus_detections": 0,
        }

        # Individual model stats
        for model_name, model_dets in detections.items():
            class_counts = Counter(d.class_id for d in model_dets)
            result_data["model_stats"][model_name] = {
                "total_detections": len(model_dets),
                "avg_confidence": float(np.mean([d.confidence for d in model_dets]))
                if model_dets
                else 0,
                "confidence_std": float(np.std([d.confidence for d in model_dets]))
                if model_dets
                else 0,
                "class_distribution": dict(class_counts),
            }

        # Pairwise agreement
        model_names = list(detections.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                agreement = self._calculate_pairwise_agreement(
                    detections[model1], detections[model2], context.iou_threshold
                )
                result_data["agreement_matrix"][f"{model1}_{model2}"] = agreement

        return AnalysisResult(name=self.name, data=result_data)

    def _calculate_pairwise_agreement(
        self, dets1: List[Detection], dets2: List[Detection], iou_threshold: float
    ) -> Dict:
        """Calculate agreement between two detection sets."""
        if not dets1 or not dets2:
            return {
                "jaccard": 0.0,
                "matches": 0,
                "model1_unique": len(dets1),
                "model2_unique": len(dets2),
            }

        matches = 0
        matched_j = set()

        for det1 in dets1:
            for j, det2 in enumerate(dets2):
                if j in matched_j:
                    continue
                if det1.class_id == det2.class_id:
                    iou = calculate_iou(det1.bbox, det2.bbox)
                    if iou >= iou_threshold:
                        matches += 1
                        matched_j.add(j)
                        break

        union = len(dets1) + len(dets2) - matches
        jaccard = matches / union if union > 0 else 0.0

        return {
            "jaccard": float(jaccard),
            "matches": matches,
            "model1_unique": len(dets1) - matches,
            "model2_unique": len(dets2) - len(matched_j),
        }

    def format_summary(self, result: AnalysisResult) -> str:
        """Format agreement summary."""
        lines = [f"Models: {result.get('total_models', 0)}"]
        agreement_data = result.get("agreement_matrix", {})
        if agreement_data:
            scores = [v["jaccard"] for v in agreement_data.values()]
            lines.append(f"Avg agreement: {np.mean(scores):.3f}")
        return "\n".join(lines)


@AnalysisRegistry.register("confidence")
class ConfidenceAnalysisHandler(AnalysisHandler):
    """Analyzes confidence score distributions."""

    description = "Analyze confidence score distributions"

    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze confidence distributions."""
        error = self.validate(context)
        if error:
            return AnalysisResult(name=self.name, success=False, error=error)

        detections = context.detections
        result_data = {
            "model_confidence_stats": {},
            "confidence_bins": {},
        }

        for model_name, model_dets in detections.items():
            if not model_dets:
                continue

            confidences = [d.confidence for d in model_dets]

            result_data["model_confidence_stats"][model_name] = {
                "mean": float(np.mean(confidences)),
                "median": float(np.median(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "percentiles": {
                    "25": float(np.percentile(confidences, 25)),
                    "75": float(np.percentile(confidences, 75)),
                    "90": float(np.percentile(confidences, 90)),
                    "95": float(np.percentile(confidences, 95)),
                },
            }

            hist, bin_edges = np.histogram(confidences, bins=context.confidence_bins)
            result_data["confidence_bins"][model_name] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        return AnalysisResult(name=self.name, data=result_data)

    def format_summary(self, result: AnalysisResult) -> str:
        """Format confidence summary."""
        stats = result.get("model_confidence_stats", {})
        lines = []
        for model, data in stats.items():
            lines.append(f"{model}: mean={data['mean']:.3f}, std={data['std']:.3f}")
        return "\n".join(lines)


@AnalysisRegistry.register("class-wise")
class ClassWiseAnalysisHandler(AnalysisHandler):
    """Analyzes performance by object class."""

    description = "Analyze performance by object class"

    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze class-wise performance."""
        error = self.validate(context)
        if error:
            return AnalysisResult(name=self.name, success=False, error=error)

        detections = context.detections
        result_data = {"class_statistics": {}, "model_class_performance": {}}

        # Collect all classes
        all_classes = set()
        for model_dets in detections.values():
            for det in model_dets:
                all_classes.add(det.class_id)

        # Filter to target classes if specified
        if context.target_classes:
            all_classes = all_classes.intersection(set(context.target_classes))

        # Analyze each class
        for class_id in sorted(all_classes):
            class_stats = {
                "total_detections": 0,
                "models_detecting": 0,
                "avg_confidence": 0,
                "model_contributions": {},
            }

            class_confidences = []
            models_with_class = 0

            for model_name, model_dets in detections.items():
                class_dets = [d for d in model_dets if d.class_id == class_id]

                if class_dets:
                    models_with_class += 1
                    model_conf = float(np.mean([d.confidence for d in class_dets]))
                    class_stats["model_contributions"][model_name] = {
                        "count": len(class_dets),
                        "avg_confidence": model_conf,
                    }
                    class_confidences.extend([d.confidence for d in class_dets])

            class_stats["total_detections"] = len(class_confidences)
            class_stats["models_detecting"] = models_with_class
            class_stats["avg_confidence"] = (
                float(np.mean(class_confidences)) if class_confidences else 0
            )

            result_data["class_statistics"][class_id] = class_stats

        return AnalysisResult(name=self.name, data=result_data)


@AnalysisRegistry.register("spatial")
class SpatialAnalysisHandler(AnalysisHandler):
    """Analyzes spatial distribution of detections."""

    description = "Analyze spatial distribution of detections"

    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze spatial distribution."""
        error = self.validate(context)
        if error:
            return AnalysisResult(name=self.name, success=False, error=error)

        detections = context.detections
        result_data = {"model_spatial_stats": {}, "coverage_analysis": {}}

        for model_name, model_dets in detections.items():
            if not model_dets:
                continue

            # Center coordinates
            x_centers = [d.x for d in model_dets]
            y_centers = [d.y for d in model_dets]

            # Box sizes
            areas = [d.w * d.h for d in model_dets]

            result_data["model_spatial_stats"][model_name] = {
                "x_range": [float(min(x_centers)), float(max(x_centers))],
                "y_range": [float(min(y_centers)), float(max(y_centers))],
                "avg_area": float(np.mean(areas)),
                "area_std": float(np.std(areas)),
                "x_centroid": float(np.mean(x_centers)),
                "y_centroid": float(np.mean(y_centers)),
            }

            # Coverage (approximate)
            x_range = max(x_centers) - min(x_centers)
            y_range = max(y_centers) - min(y_centers)
            result_data["coverage_analysis"][model_name] = {
                "coverage_area": float(x_range * y_range),
                "density": len(model_dets) / (x_range * y_range) if x_range * y_range > 0 else 0,
            }

        return AnalysisResult(name=self.name, data=result_data)


@AnalysisRegistry.register("performance")
class PerformanceAnalysisHandler(AnalysisHandler):
    """Analyzes overall detection performance metrics."""

    description = "Analyze overall detection performance"

    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze performance metrics."""
        error = self.validate(context)
        if error:
            return AnalysisResult(name=self.name, success=False, error=error)

        detections = context.detections
        result_data = {"model_performance": {}, "comparison": {}}

        all_counts = []
        all_confidences = []

        for model_name, model_dets in detections.items():
            count = len(model_dets)
            avg_conf = float(np.mean([d.confidence for d in model_dets])) if model_dets else 0
            classes = len(set(d.class_id for d in model_dets))

            result_data["model_performance"][model_name] = {
                "detection_count": count,
                "avg_confidence": avg_conf,
                "unique_classes": classes,
            }

            all_counts.append(count)
            if model_dets:
                all_confidences.extend([d.confidence for d in model_dets])

        # Comparison stats
        if all_counts:
            result_data["comparison"] = {
                "total_detections": sum(all_counts),
                "avg_per_model": float(np.mean(all_counts)),
                "overall_avg_confidence": float(np.mean(all_confidences)) if all_confidences else 0,
            }

        return AnalysisResult(name=self.name, data=result_data)


__all__ = [
    "AgreementAnalysisHandler",
    "ConfidenceAnalysisHandler",
    "ClassWiseAnalysisHandler",
    "SpatialAnalysisHandler",
    "PerformanceAnalysisHandler",
]
