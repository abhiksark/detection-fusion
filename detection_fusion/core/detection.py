from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    """Represents a single object detection."""
    
    class_id: int
    x: float  # center x
    y: float  # center y
    w: float  # width
    h: float  # height
    confidence: float
    model_source: str = ""
    image_name: str = ""  # Image this detection belongs to
    
    @property
    def bbox(self) -> List[float]:
        """Returns bounding box in [x, y, w, h] format."""
        return [self.x, self.y, self.w, self.h]
    
    @property
    def xyxy(self) -> List[float]:
        """Returns bounding box in [x1, y1, x2, y2] format."""
        return [
            self.x - self.w/2, 
            self.y - self.h/2,
            self.x + self.w/2, 
            self.y + self.h/2
        ]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Returns center point (x, y)."""
        return (self.x, self.y)
    
    @property
    def area(self) -> float:
        """Returns area of bounding box."""
        return self.w * self.h
    
    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            'class_id': self.class_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'model_source': self.model_source
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create detection from dictionary."""
        bbox = data['bbox']
        return cls(
            class_id=data['class_id'],
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
            confidence=data['confidence'],
            model_source=data.get('model_source', '')
        )
    
    def __hash__(self) -> int:
        """Make Detection hashable for use in sets/dicts."""
        return hash((
            self.class_id,
            round(self.x, 6),
            round(self.y, 6),
            round(self.w, 6),
            round(self.h, 6),
            round(self.confidence, 6),
            self.model_source,
            self.image_name
        ))
    
    def __eq__(self, other) -> bool:
        """Check equality between detections."""
        if not isinstance(other, Detection):
            return False
        return (
            self.class_id == other.class_id and
            abs(self.x - other.x) < 1e-6 and
            abs(self.y - other.y) < 1e-6 and
            abs(self.w - other.w) < 1e-6 and
            abs(self.h - other.h) < 1e-6 and
            abs(self.confidence - other.confidence) < 1e-6 and
            self.model_source == other.model_source and
            self.image_name == other.image_name
        )