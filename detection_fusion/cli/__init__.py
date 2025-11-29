"""
CLI infrastructure for DetectionFusion.

Provides output formatting, detection loading, and analysis handlers.
"""

from .loaders import DetectionLoader, ImageModeLoader, SingleFileLoader, get_loader
from .main import cli
from .output import OutputFormatter, console

__all__ = [
    "cli",
    "OutputFormatter",
    "console",
    "DetectionLoader",
    "ImageModeLoader",
    "SingleFileLoader",
    "get_loader",
]
