"""
DetectionFusion Package

A comprehensive toolkit for fusing multiple object detection results with 
ground truth validation and error analysis.
"""

__version__ = "0.2.0"
__author__ = "DetectionFusion Team"

from .core.detection import Detection
from .core.ensemble import EnsembleVoting, AdvancedEnsemble
from .core.analyzer import MultiModelAnalyzer

# Ground truth evaluation components
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import EvaluationMetrics
from .evaluation.error_analysis import ErrorAnalyzer
from .evaluation.optimization import StrategyOptimizer

# Ground truth rectification system (imported from root level)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gt_rectify import GTRectifier, GTRectificationError
except ImportError:
    # Fallback for cases where gt_rectify might not be available
    GTRectifier = None
    GTRectificationError = None

__all__ = [
    "Detection",
    "EnsembleVoting", 
    "AdvancedEnsemble",
    "MultiModelAnalyzer",
    "Evaluator",
    "EvaluationMetrics", 
    "ErrorAnalyzer",
    "StrategyOptimizer"
]

# Add rectification classes if available
if GTRectifier is not None:
    __all__.extend(["GTRectifier", "GTRectificationError"])