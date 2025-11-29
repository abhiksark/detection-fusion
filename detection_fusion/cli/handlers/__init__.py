"""
Analysis handlers for CLI tools.

Provides a registry-based approach to analysis types, replacing
if-statement chains with pluggable handlers.
"""

from .analysis import (
    AgreementAnalysisHandler,
    ClassWiseAnalysisHandler,
    ConfidenceAnalysisHandler,
    PerformanceAnalysisHandler,
    SpatialAnalysisHandler,
)
from .base import AnalysisHandler, AnalysisResult
from .registry import AnalysisRegistry, get_handler, list_handlers

__all__ = [
    "AnalysisHandler",
    "AnalysisResult",
    "AnalysisRegistry",
    "get_handler",
    "list_handlers",
    "AgreementAnalysisHandler",
    "ConfidenceAnalysisHandler",
    "ClassWiseAnalysisHandler",
    "SpatialAnalysisHandler",
    "PerformanceAnalysisHandler",
]
