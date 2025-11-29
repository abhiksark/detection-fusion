from .registry import FormatRegistry
from .base import AnnotationReader, AnnotationWriter

from . import yolo
from . import voc_xml
from . import coco

__all__ = [
    "FormatRegistry",
    "AnnotationReader",
    "AnnotationWriter",
]
