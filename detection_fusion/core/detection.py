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