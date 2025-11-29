"""
Base classes for analysis handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.detection import Detection


@dataclass
class AnalysisResult:
    """Result of an analysis handler."""

    name: str
    """Name of the analysis performed"""

    data: Dict[str, Any] = field(default_factory=dict)
    """Analysis results data"""

    success: bool = True
    """Whether the analysis completed successfully"""

    error: Optional[str] = None
    """Error message if analysis failed"""

    def __bool__(self) -> bool:
        return self.success

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dict."""
        return self.data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        """Update data with another dict."""
        self.data.update(other)


@dataclass
class AnalysisContext:
    """Context passed to analysis handlers."""

    detections: Dict[str, List[Detection]]
    """Model name -> list of detections"""

    iou_threshold: float = 0.5
    """IoU threshold for matching"""

    confidence_threshold: float = 0.1
    """Minimum confidence threshold"""

    confidence_bins: int = 10
    """Number of bins for confidence histograms"""

    target_classes: Optional[List[int]] = None
    """Specific classes to analyze (None = all)"""

    verbose: bool = False
    """Verbose output"""

    quiet: bool = False
    """Suppress output"""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Additional context-specific data"""


class AnalysisHandler(ABC):
    """Abstract base class for analysis handlers.

    Subclasses implement specific analysis types (agreement, confidence, etc.)
    and are registered with the AnalysisRegistry.
    """

    # Handler metadata
    name: str = "base"
    description: str = "Base analysis handler"

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Perform the analysis.

        Args:
            context: Analysis context with detections and parameters

        Returns:
            AnalysisResult with data and status
        """
        pass

    def validate(self, context: AnalysisContext) -> Optional[str]:
        """Validate that analysis can be performed.

        Args:
            context: Analysis context

        Returns:
            Error message if validation fails, None otherwise
        """
        if not context.detections:
            return "No detections provided"
        return None

    def format_summary(self, result: AnalysisResult) -> str:
        """Format a summary of the analysis result.

        Args:
            result: Analysis result

        Returns:
            Formatted summary string
        """
        return f"{self.name}: {len(result.data)} items analyzed"


__all__ = ["AnalysisHandler", "AnalysisResult", "AnalysisContext"]
