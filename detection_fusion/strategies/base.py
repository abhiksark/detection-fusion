from abc import ABC, abstractmethod
from typing import List, Dict
from ..core.detection import Detection


class BaseStrategy(ABC):
    """Base class for all ensemble strategies."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    @abstractmethod
    def merge(self, detections: Dict[str, List[Detection]], **kwargs) -> List[Detection]:
        """
        Merge detections from multiple models.
        
        Args:
            detections: Dictionary mapping model names to their detections
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of merged detections
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass