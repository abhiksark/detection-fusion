from pathlib import Path
from typing import Dict, List, Optional, Set

from detection_fusion.core.detection import Detection
from detection_fusion.core.detection_set import DetectionSet

from .formats import FormatRegistry
from .formats.base import AnnotationReader

DEFAULT_EXCLUDED_DIRS = {"unified", "GT", "__pycache__", ".git"}


class FileDetectionLoader:
    """Load detections from filesystem with format auto-detection."""

    def __init__(
        self,
        labels_dir: str,
        format: str = "auto",
        excluded_dirs: Optional[Set[str]] = None,
    ):
        self._labels_dir = Path(labels_dir)
        self._format = format
        self._excluded = excluded_dirs or DEFAULT_EXCLUDED_DIRS
        self._reader: Optional[AnnotationReader] = None

    def _get_reader(self) -> AnnotationReader:
        if self._reader is None:
            if self._format == "auto":
                self._reader = FormatRegistry.auto_detect_reader(self._labels_dir)
            else:
                self._reader = FormatRegistry.get_reader(self._format)
        return self._reader

    def find_all_models(self) -> List[str]:
        models: List[str] = []
        for path in self._labels_dir.iterdir():
            if path.is_dir() and path.name not in self._excluded:
                models.append(path.name)
        return sorted(models)

    def load(self, model_name: str) -> List[Detection]:
        model_dir = self._labels_dir / model_name
        if not model_dir.exists():
            return []

        reader = self._get_reader()
        result_dict = reader.read_directory(model_dir)

        detections: List[Detection] = []
        for dets in result_dict.values():
            for det in dets:
                detections.append(det.with_source(model_name))

        return detections

    def load_all(self) -> Dict[str, List[Detection]]:
        result: Dict[str, List[Detection]] = {}
        for model_name in self.find_all_models():
            dets = self.load(model_name)
            if dets:
                result[model_name] = dets
        return result

    def load_as_set(self) -> DetectionSet:
        return DetectionSet(self.load_all())


def load_detections(
    path: str,
    format: str = "auto",
    model_name: Optional[str] = None,
) -> List[Detection]:
    """Convenience function to load detections from any format."""
    path_obj = Path(path)

    if format == "auto":
        reader = FormatRegistry.auto_detect_reader(path_obj)
    else:
        reader = FormatRegistry.get_reader(format)

    if path_obj.is_file():
        dets = reader.read_file(path_obj)
    else:
        result_dict = reader.read_directory(path_obj)
        dets = []
        for det_list in result_dict.values():
            dets.extend(det_list)

    if model_name:
        dets = [d.with_source(model_name) for d in dets]

    return dets
