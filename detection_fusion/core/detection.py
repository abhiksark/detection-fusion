from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Represents a single object detection with validation."""

    model_config = {"frozen": True}

    class_id: int = Field(ge=0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_source: str = ""
    image_name: str = ""

    @property
    def bbox(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]

    @property
    def xyxy(self) -> List[float]:
        return [
            self.x - self.w / 2,
            self.y - self.h / 2,
            self.x + self.w / 2,
            self.y + self.h / 2,
        ]

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def area(self) -> float:
        return self.w * self.h

    def with_confidence(self, confidence: float) -> "Detection":
        return self.model_copy(update={"confidence": confidence})

    def with_source(self, source: str) -> "Detection":
        return self.model_copy(update={"model_source": source})

    def with_image(self, image_name: str) -> "Detection":
        return self.model_copy(update={"image_name": image_name})

    def iou_with(self, other: "Detection") -> float:
        x1_min, y1_min, x1_max, y1_max = self.xyxy
        x2_min, y2_min, x2_max, y2_max = other.xyxy

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = self.area + other.area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "model_source": self.model_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        bbox = data["bbox"]
        return cls(
            class_id=data["class_id"],
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
            confidence=data["confidence"],
            model_source=data.get("model_source", ""),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.class_id,
                round(self.x, 6),
                round(self.y, 6),
                round(self.w, 6),
                round(self.h, 6),
                round(self.confidence, 6),
                self.model_source,
                self.image_name,
            )
        )
