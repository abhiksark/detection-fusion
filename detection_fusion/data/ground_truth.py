from pathlib import Path
from typing import Dict, List, Optional

from detection_fusion.core.detection import Detection

from .formats import FormatRegistry
from .formats.base import AnnotationReader


class GroundTruthRepository:
    """Repository for loading and caching ground truth annotations."""

    def __init__(
        self,
        gt_dir: str,
        format: str = "auto",
    ):
        self._gt_dir = Path(gt_dir)
        self._format = format
        self._cache: Optional[Dict[str, List[Detection]]] = None
        self._reader: Optional[AnnotationReader] = None

    def _get_reader(self) -> AnnotationReader:
        if self._reader is None:
            if self._format == "auto":
                self._reader = FormatRegistry.auto_detect_reader(self._gt_dir)
            else:
                self._reader = FormatRegistry.get_reader(self._format)
        return self._reader

    def _ensure_loaded(self) -> Dict[str, List[Detection]]:
        if self._cache is None:
            reader = self._get_reader()
            self._cache = reader.read_directory(self._gt_dir)
        return self._cache

    def exists(self) -> bool:
        return self._gt_dir.exists() and self._gt_dir.is_dir()

    def load(self, image_name: Optional[str] = None) -> List[Detection]:
        data = self._ensure_loaded()

        if image_name is not None:
            return data.get(image_name, [])

        result: List[Detection] = []
        for dets in data.values():
            result.extend(dets)
        return result

    def load_by_image(self) -> Dict[str, List[Detection]]:
        return self._ensure_loaded()

    def image_names(self) -> List[str]:
        return sorted(self._ensure_loaded().keys())

    def clear_cache(self) -> None:
        self._cache = None

    @property
    def path(self) -> Path:
        return self._gt_dir
