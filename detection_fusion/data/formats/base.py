from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_fusion.core.detection import Detection


class AnnotationReader(ABC):
    """Abstract base class for reading annotations from various formats."""

    format_name: str
    file_extensions: List[str]

    @abstractmethod
    def read_file(
        self,
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        ...

    @abstractmethod
    def read_directory(self, dir_path: Path) -> Dict[str, List[Detection]]:
        ...

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        ...


class AnnotationWriter(ABC):
    """Abstract base class for writing annotations to various formats."""

    format_name: str
    file_extension: str

    @abstractmethod
    def write_file(
        self,
        detections: List[Detection],
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        ...

    @abstractmethod
    def write_directory(
        self,
        detections: Dict[str, List[Detection]],
        dir_path: Path,
    ) -> None:
        ...
