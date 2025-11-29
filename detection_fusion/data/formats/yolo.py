from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_fusion.core.detection import Detection

from .base import AnnotationReader, AnnotationWriter
from .registry import FormatRegistry


@FormatRegistry.register_reader("yolo")
class YOLOReader(AnnotationReader):
    """YOLO txt format reader (default internal format)."""

    format_name = "yolo"
    file_extensions = [".txt"]

    def read_file(
        self,
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        path = Path(path)
        detections: List[Detection] = []

        if not path.exists():
            return detections

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                confidence = float(parts[5]) if len(parts) > 5 else 1.0

                det = Detection(
                    class_id=class_id,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    image_name=path.stem,
                )
                detections.append(det)

        return detections

    def read_directory(self, dir_path: Path) -> Dict[str, List[Detection]]:
        dir_path = Path(dir_path)
        result: Dict[str, List[Detection]] = {}

        for txt_file in dir_path.glob("*.txt"):
            image_name = txt_file.stem
            dets = self.read_file(txt_file)
            if dets:
                result[image_name] = dets

        return result

    @classmethod
    def can_read(cls, path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".txt"
        if path.is_dir():
            return any(path.glob("*.txt"))
        return False


@FormatRegistry.register_writer("yolo")
class YOLOWriter(AnnotationWriter):
    """YOLO txt format writer."""

    format_name = "yolo"
    file_extension = ".txt"

    def write_file(
        self,
        detections: List[Detection],
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for det in detections:
                line = f"{det.class_id} {det.x:.6f} {det.y:.6f} {det.w:.6f} {det.h:.6f}"
                if det.confidence < 1.0:
                    line += f" {det.confidence:.6f}"
                f.write(line + "\n")

    def write_directory(
        self,
        detections: Dict[str, List[Detection]],
        dir_path: Path,
    ) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for image_name, dets in detections.items():
            file_path = dir_path / f"{image_name}.txt"
            self.write_file(dets, file_path)
