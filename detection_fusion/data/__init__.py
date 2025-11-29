from .formats import FormatRegistry
from .ground_truth import GroundTruthRepository
from .loader import FileDetectionLoader

__all__ = [
    "FileDetectionLoader",
    "GroundTruthRepository",
    "FormatRegistry",
]
