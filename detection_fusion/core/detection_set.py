from typing import Dict, Iterator, List, Set

from .detection import Detection


class DetectionSet:
    """Rich aggregate for detections with filtering, grouping, and statistics."""

    def __init__(self, detections: Dict[str, List[Detection]]):
        self._data: Dict[str, List[Detection]] = detections

    def by_model(self, name: str) -> List[Detection]:
        return self._data.get(name, [])

    def by_image(self, image_name: str) -> "DetectionSet":
        filtered: Dict[str, List[Detection]] = {}
        for model, dets in self._data.items():
            matching = [d for d in dets if d.image_name == image_name]
            if matching:
                filtered[model] = matching
        return DetectionSet(filtered)

    def all_detections(self) -> List[Detection]:
        result: List[Detection] = []
        for dets in self._data.values():
            result.extend(dets)
        return result

    def filter_by_confidence(self, min_conf: float) -> "DetectionSet":
        filtered: Dict[str, List[Detection]] = {}
        for model, dets in self._data.items():
            matching = [d for d in dets if d.confidence >= min_conf]
            if matching:
                filtered[model] = matching
        return DetectionSet(filtered)

    def filter_by_class(self, class_ids: List[int]) -> "DetectionSet":
        class_set = set(class_ids)
        filtered: Dict[str, List[Detection]] = {}
        for model, dets in self._data.items():
            matching = [d for d in dets if d.class_id in class_set]
            if matching:
                filtered[model] = matching
        return DetectionSet(filtered)

    def group_by_image(self) -> Dict[str, "DetectionSet"]:
        groups: Dict[str, Dict[str, List[Detection]]] = {}
        for model, dets in self._data.items():
            for det in dets:
                if det.image_name not in groups:
                    groups[det.image_name] = {}
                if model not in groups[det.image_name]:
                    groups[det.image_name][model] = []
                groups[det.image_name][model].append(det)
        return {img: DetectionSet(data) for img, data in groups.items()}

    def group_by_class(self) -> Dict[int, "DetectionSet"]:
        groups: Dict[int, Dict[str, List[Detection]]] = {}
        for model, dets in self._data.items():
            for det in dets:
                if det.class_id not in groups:
                    groups[det.class_id] = {}
                if model not in groups[det.class_id]:
                    groups[det.class_id][model] = []
                groups[det.class_id][model].append(det)
        return {cls: DetectionSet(data) for cls, data in groups.items()}

    def confidence_stats(self) -> Dict[str, float]:
        all_dets = self.all_detections()
        if not all_dets:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        confidences = [d.confidence for d in all_dets]
        n = len(confidences)
        mean = sum(confidences) / n
        variance = sum((c - mean) ** 2 for c in confidences) / n
        std = variance**0.5

        return {
            "mean": mean,
            "std": std,
            "min": min(confidences),
            "max": max(confidences),
        }

    def class_distribution(self) -> Dict[int, int]:
        distribution: Dict[int, int] = {}
        for det in self.all_detections():
            distribution[det.class_id] = distribution.get(det.class_id, 0) + 1
        return distribution

    @property
    def model_names(self) -> List[str]:
        return list(self._data.keys())

    @property
    def image_names(self) -> List[str]:
        names: Set[str] = set()
        for dets in self._data.values():
            for det in dets:
                if det.image_name:
                    names.add(det.image_name)
        return sorted(names)

    @property
    def total_count(self) -> int:
        return sum(len(dets) for dets in self._data.values())

    @property
    def raw_data(self) -> Dict[str, List[Detection]]:
        return self._data

    def __iter__(self) -> Iterator[Detection]:
        for dets in self._data.values():
            yield from dets

    def __len__(self) -> int:
        return self.total_count

    def __repr__(self) -> str:
        return f"DetectionSet(models={len(self._data)}, detections={self.total_count})"
