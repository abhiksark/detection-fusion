"""
Detection loading abstractions for CLI tools.

Provides a unified interface for loading detections in different modes
(image-based vs single-file), eliminating mode branching in CLI tools.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ..core.detection import Detection
from ..data.formats import FormatRegistry

# Get YOLO reader once for reuse
_yolo_reader = FormatRegistry.get_reader("yolo")


def _read_detections(
    file_path: str,
    model_name: str = "",
    default_confidence: float = 1.0,
    image_name: str = "",
) -> List[Detection]:
    """Read detections and set model_source."""
    detections = _yolo_reader.read_file(file_path)
    result = []
    for d in detections:
        # Apply model_source and image_name if provided
        if model_name or image_name:
            d = d.with_source(model_name) if model_name else d
            if image_name:
                d = Detection(
                    class_id=d.class_id,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    confidence=d.confidence,
                    model_source=d.model_source,
                    image_name=image_name,
                )
        result.append(d)
    return result


@dataclass
class LoadResult:
    """Result of loading detections."""

    detections: Dict[str, List[Detection]]
    """Model name -> list of detections"""

    models: List[str]
    """List of model names found"""

    image_detections: Dict[str, Dict[str, List[Detection]]] = field(default_factory=dict)
    """Image name -> model name -> detections (for image mode)"""

    total_count: int = 0
    """Total number of detections loaded"""

    @property
    def model_count(self) -> int:
        """Number of models loaded."""
        return len(self.models)


class DetectionLoader(ABC):
    """Abstract base class for detection loaders."""

    def __init__(
        self,
        labels_dir: str,
        default_confidence: float = 1.0,
        excluded_dirs: Optional[List[str]] = None,
    ):
        self.labels_dir = Path(labels_dir)
        self.default_confidence = default_confidence
        self.excluded_dirs = excluded_dirs or ["unified", "__pycache__", "GT"]

    @abstractmethod
    def load(self) -> LoadResult:
        """Load detections and return a LoadResult."""
        pass

    def _get_model_dirs(self) -> List[Path]:
        """Get all valid model directories."""
        if not self.labels_dir.exists():
            return []

        return [
            d for d in self.labels_dir.iterdir() if d.is_dir() and d.name not in self.excluded_dirs
        ]


class SingleFileLoader(DetectionLoader):
    """Loader for single detection file per model (detections.txt mode)."""

    def __init__(
        self,
        labels_dir: str,
        filename: str = "detections.txt",
        default_confidence: float = 1.0,
        excluded_dirs: Optional[List[str]] = None,
    ):
        super().__init__(labels_dir, default_confidence, excluded_dirs)
        self.filename = filename

    def load(self) -> LoadResult:
        """Load detections from single file per model."""
        detections = {}
        models = []
        total = 0

        for model_dir in self._get_model_dirs():
            model_name = model_dir.name
            file_path = model_dir / self.filename

            if file_path.exists():
                model_dets = _read_detections(str(file_path), model_name, self.default_confidence)
                detections[model_name] = model_dets
                models.append(model_name)
                total += len(model_dets)

        return LoadResult(detections=detections, models=models, total_count=total)


class ImageModeLoader(DetectionLoader):
    """Loader for per-image detection files (image mode)."""

    def __init__(
        self,
        labels_dir: str,
        default_confidence: float = 1.0,
        excluded_dirs: Optional[List[str]] = None,
    ):
        super().__init__(labels_dir, default_confidence, excluded_dirs)

    def load(self) -> LoadResult:
        """Load detections from per-image files."""
        detections = {}
        models = []
        image_detections = defaultdict(dict)
        total = 0

        for model_dir in self._get_model_dirs():
            model_name = model_dir.name
            models.append(model_name)
            detections[model_name] = []

            # Load all .txt files in the model directory
            for txt_file in model_dir.glob("*.txt"):
                image_name = txt_file.stem
                image_dets = _read_detections(
                    str(txt_file), model_name, self.default_confidence, image_name
                )
                image_detections[image_name][model_name] = image_dets
                detections[model_name].extend(image_dets)
                total += len(image_dets)

        return LoadResult(
            detections=detections,
            models=models,
            image_detections=dict(image_detections),
            total_count=total,
        )

    def iter_images(self) -> Iterator[Tuple[str, Dict[str, List[Detection]]]]:
        """Iterate over images, yielding (image_name, model_detections) pairs."""
        result = self.load()
        for image_name, model_dets in result.image_detections.items():
            yield image_name, model_dets


class GroundTruthLoader(DetectionLoader):
    """Loader for ground truth detections."""

    def __init__(
        self, gt_dir: str, filename: str = "detections.txt", default_confidence: float = 1.0
    ):
        super().__init__(gt_dir, default_confidence, [])
        self.filename = filename
        self._cache = {}

    def load(self) -> LoadResult:
        """Load ground truth detections."""
        gt_path = self.labels_dir / self.filename

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        # Check cache
        cache_key = str(gt_path)
        if cache_key in self._cache:
            dets = self._cache[cache_key]
        else:
            dets = _read_detections(str(gt_path), "GT", self.default_confidence)
            self._cache[cache_key] = dets

        return LoadResult(detections={"GT": dets}, models=["GT"], total_count=len(dets))


def get_loader(
    labels_dir: str,
    image_mode: bool = False,
    filename: str = "detections.txt",
    default_confidence: float = 1.0,
) -> DetectionLoader:
    """Factory function to get the appropriate loader.

    Args:
        labels_dir: Directory containing model subdirectories
        image_mode: If True, use per-image file loading
        filename: Detection filename (for single file mode)
        default_confidence: Default confidence for detections without one

    Returns:
        Appropriate DetectionLoader instance
    """
    if image_mode:
        return ImageModeLoader(labels_dir, default_confidence)
    else:
        return SingleFileLoader(labels_dir, filename, default_confidence)


__all__ = [
    "LoadResult",
    "DetectionLoader",
    "SingleFileLoader",
    "ImageModeLoader",
    "GroundTruthLoader",
    "get_loader",
]
