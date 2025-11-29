import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from detection_fusion.core.detection import Detection
from detection_fusion.exceptions import FormatError

from .base import AnnotationReader, AnnotationWriter
from .registry import FormatRegistry


@FormatRegistry.register_reader("coco")
class COCOReader(AnnotationReader):
    """COCO JSON format reader."""

    format_name = "coco"
    file_extensions = [".json"]

    def read_file(
        self,
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        path = Path(path)

        if not path.exists():
            return []

        with open(path, "r") as f:
            data = json.load(f)

        # Build image lookup
        images: Dict[int, Dict] = {}
        for img in data.get("images", []):
            images[img["id"]] = img

        # Build category lookup
        categories: Dict[int, int] = {}
        for idx, cat in enumerate(data.get("categories", [])):
            categories[cat["id"]] = idx

        detections: List[Detection] = []

        for ann in data.get("annotations", []):
            image_id = ann["image_id"]
            if image_id not in images:
                continue

            img = images[image_id]
            width = img["width"]
            height = img["height"]
            image_name = Path(img.get("file_name", str(image_id))).stem

            # COCO bbox format: [x, y, width, height] in absolute pixels
            bbox = ann["bbox"]
            abs_x, abs_y, abs_w, abs_h = bbox

            # Convert to normalized center format
            x = (abs_x + abs_w / 2) / width
            y = (abs_y + abs_h / 2) / height
            w = abs_w / width
            h = abs_h / height

            # Clamp to valid range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            category_id = ann["category_id"]
            class_id = categories.get(category_id, category_id)
            confidence = ann.get("score", 1.0)

            det = Detection(
                class_id=class_id,
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                image_name=image_name,
            )
            detections.append(det)

        return detections

    def read_directory(self, dir_path: Path) -> Dict[str, List[Detection]]:
        dir_path = Path(dir_path)
        result: Dict[str, List[Detection]] = {}

        for json_file in dir_path.glob("*.json"):
            dets = self.read_file(json_file)
            for det in dets:
                if det.image_name not in result:
                    result[det.image_name] = []
                result[det.image_name].append(det)

        return result

    @classmethod
    def can_read(cls, path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            if path.suffix.lower() != ".json":
                return False
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return "images" in data and "annotations" in data
            except Exception:
                return False
        if path.is_dir():
            for json_file in path.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    if "images" in data and "annotations" in data:
                        return True
                except Exception:
                    continue
        return False


@FormatRegistry.register_writer("coco")
class COCOWriter(AnnotationWriter):
    """COCO JSON format writer."""

    format_name = "coco"
    file_extension = ".json"

    def __init__(
        self,
        category_names: Optional[Dict[int, str]] = None,
        image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self._category_names = category_names or {}
        self._image_sizes = image_sizes or {}

    def write_file(
        self,
        detections: List[Detection],
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        # Group detections by image
        by_image: Dict[str, List[Detection]] = {}
        for det in detections:
            if det.image_name not in by_image:
                by_image[det.image_name] = []
            by_image[det.image_name].append(det)

        # Build output structure
        images = []
        annotations = []
        categories_seen = set()

        ann_id = 1
        for img_id, (image_name, dets) in enumerate(by_image.items(), start=1):
            size = self._image_sizes.get(image_name, image_size)
            if not size:
                raise FormatError(
                    f"Image size required for {image_name} in COCO format"
                )

            width, height = size
            images.append({
                "id": img_id,
                "file_name": f"{image_name}.jpg",
                "width": width,
                "height": height,
            })

            for det in dets:
                categories_seen.add(det.class_id)

                # Convert from normalized center to COCO format
                abs_w = det.w * width
                abs_h = det.h * height
                abs_x = (det.x * width) - (abs_w / 2)
                abs_y = (det.y * height) - (abs_h / 2)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": det.class_id,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                    "score": det.confidence,
                })
                ann_id += 1

        categories = []
        for cat_id in sorted(categories_seen):
            name = self._category_names.get(cat_id, f"class_{cat_id}")
            categories.append({"id": cat_id, "name": name})

        output = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def write_directory(
        self,
        detections: Dict[str, List[Detection]],
        dir_path: Path,
    ) -> None:
        all_dets = []
        for dets in detections.values():
            all_dets.extend(dets)

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        output_path = dir_path / "annotations.json"
        self.write_file(all_dets, output_path)
