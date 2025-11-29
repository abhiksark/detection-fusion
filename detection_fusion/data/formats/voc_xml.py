import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_fusion.core.detection import Detection
from detection_fusion.exceptions import FormatError

from .base import AnnotationReader, AnnotationWriter
from .registry import FormatRegistry


@FormatRegistry.register_reader("voc_xml")
class VOCXMLReader(AnnotationReader):
    """Pascal VOC XML format reader."""

    format_name = "voc_xml"
    file_extensions = [".xml"]

    def __init__(self, class_map: Optional[Dict[str, int]] = None):
        self._class_map = class_map or {}
        self._next_class_id = 0

    def _get_class_id(self, name: str) -> int:
        if name not in self._class_map:
            self._class_map[name] = self._next_class_id
            self._next_class_id += 1
        return self._class_map[name]

    def read_file(
        self,
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        path = Path(path)
        detections: List[Detection] = []

        if not path.exists():
            return detections

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise FormatError(f"Failed to parse XML file {path}: {e}")

        size_elem = root.find("size")
        if size_elem is not None:
            width = int(size_elem.findtext("width", "0"))
            height = int(size_elem.findtext("height", "0"))
        elif image_size:
            width, height = image_size
        else:
            raise FormatError(
                f"Image size not found in {path} and not provided"
            )

        if width <= 0 or height <= 0:
            raise FormatError(f"Invalid image size in {path}: {width}x{height}")

        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            class_id = self._get_class_id(name)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            # Convert to normalized center format
            x = ((xmin + xmax) / 2) / width
            y = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Clamp to valid range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            det = Detection(
                class_id=class_id,
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=1.0,
                image_name=path.stem,
            )
            detections.append(det)

        return detections

    def read_directory(self, dir_path: Path) -> Dict[str, List[Detection]]:
        dir_path = Path(dir_path)
        result: Dict[str, List[Detection]] = {}

        for xml_file in dir_path.glob("*.xml"):
            image_name = xml_file.stem
            dets = self.read_file(xml_file)
            if dets:
                result[image_name] = dets

        return result

    @classmethod
    def can_read(cls, path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            if path.suffix.lower() != ".xml":
                return False
            # Check if it looks like VOC format
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                return root.tag == "annotation" and root.find("object") is not None
            except Exception:
                return False
        if path.is_dir():
            for xml_file in path.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    if root.tag == "annotation":
                        return True
                except Exception:
                    continue
        return False


@FormatRegistry.register_writer("voc_xml")
class VOCXMLWriter(AnnotationWriter):
    """Pascal VOC XML format writer."""

    format_name = "voc_xml"
    file_extension = ".xml"

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self._class_names = class_names or {}

    def _get_class_name(self, class_id: int) -> str:
        return self._class_names.get(class_id, f"class_{class_id}")

    def write_file(
        self,
        detections: List[Detection],
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not image_size:
            raise FormatError("Image size required for VOC XML format")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        width, height = image_size

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = path.stem + ".jpg"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        for det in detections:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = self._get_class_name(det.class_id)

            # Convert from normalized center to absolute corners
            xmin = int((det.x - det.w / 2) * width)
            ymin = int((det.y - det.h / 2) * height)
            xmax = int((det.x + det.w / 2) * width)
            ymax = int((det.y + det.h / 2) * height)

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(max(0, xmin))
            ET.SubElement(bndbox, "ymin").text = str(max(0, ymin))
            ET.SubElement(bndbox, "xmax").text = str(min(width, xmax))
            ET.SubElement(bndbox, "ymax").text = str(min(height, ymax))

        tree = ET.ElementTree(root)
        tree.write(path, encoding="unicode", xml_declaration=True)

    def write_directory(
        self,
        detections: Dict[str, List[Detection]],
        dir_path: Path,
    ) -> None:
        raise FormatError(
            "VOC XML writer requires image_size per file. "
            "Use write_file() for each image individually."
        )
