from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

from detection_fusion.exceptions import FormatError

from .base import AnnotationReader, AnnotationWriter


class FormatRegistry:
    """Registry for annotation format readers and writers."""

    _readers: Dict[str, Type[AnnotationReader]] = {}
    _writers: Dict[str, Type[AnnotationWriter]] = {}

    @classmethod
    def register_reader(cls, name: Optional[str] = None) -> Callable:
        def decorator(reader_class: Type[AnnotationReader]) -> Type[AnnotationReader]:
            reg_name = name or reader_class.format_name
            cls._readers[reg_name] = reader_class
            return reader_class

        return decorator

    @classmethod
    def register_writer(cls, name: Optional[str] = None) -> Callable:
        def decorator(writer_class: Type[AnnotationWriter]) -> Type[AnnotationWriter]:
            reg_name = name or writer_class.format_name
            cls._writers[reg_name] = writer_class
            return writer_class

        return decorator

    @classmethod
    def get_reader(cls, format_name: str) -> AnnotationReader:
        if format_name not in cls._readers:
            available = ", ".join(cls._readers.keys())
            raise FormatError(f"Unknown format: {format_name}. Available: {available}")
        return cls._readers[format_name]()

    @classmethod
    def get_writer(cls, format_name: str) -> AnnotationWriter:
        if format_name not in cls._writers:
            available = ", ".join(cls._writers.keys())
            raise FormatError(f"Unknown format: {format_name}. Available: {available}")
        return cls._writers[format_name]()

    @classmethod
    def auto_detect_reader(cls, path: Path) -> AnnotationReader:
        path = Path(path)
        for reader_cls in cls._readers.values():
            if reader_cls.can_read(path):
                return reader_cls()
        raise FormatError(f"Could not auto-detect format for: {path}")

    @classmethod
    def list_formats(cls) -> Dict[str, List[str]]:
        return {
            "readers": list(cls._readers.keys()),
            "writers": list(cls._writers.keys()),
        }
