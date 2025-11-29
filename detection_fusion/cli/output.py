"""
Output formatting utilities for CLI tools.

Provides consistent, emoji-enhanced output formatting across all CLI commands.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OutputFormatter:
    """Formatter for CLI output with emoji support.

    Provides consistent formatting for success, error, warning messages,
    and structured output like strategy results and statistics.
    """

    # Emoji constants
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    ARROW = "→"
    CHECK = "✓"
    CROSS = "✗"
    STAR = "⭐"
    CHART = "📊"
    FOLDER = "📁"
    FILE = "📄"
    SEARCH = "🔍"
    MERGE = "🔀"
    SAVE = "💾"

    verbose: bool = False

    def success(self, message: str) -> str:
        """Format a success message."""
        return f"{self.SUCCESS} {message}"

    def error(self, message: str) -> str:
        """Format an error message."""
        return f"{self.ERROR} {message}"

    def warning(self, message: str) -> str:
        """Format a warning message."""
        return f"{self.WARNING} {message}"

    def info(self, message: str) -> str:
        """Format an info message."""
        return f"{self.INFO} {message}"

    def header(self, title: str, width: int = 60) -> str:
        """Format a section header."""
        return f"\n{'=' * width}\n{title}\n{'=' * width}"

    def subheader(self, title: str) -> str:
        """Format a subsection header."""
        return f"\n{self.ARROW} {title}"

    def item(self, label: str, value: Any, indent: int = 2) -> str:
        """Format a labeled item."""
        spaces = " " * indent
        return f"{spaces}{label}: {value}"

    def bullet(self, text: str, indent: int = 2) -> str:
        """Format a bullet point."""
        spaces = " " * indent
        return f"{spaces}• {text}"

    def strategy_result(
        self, name: str, count: int, avg_conf: float, extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a strategy result line."""
        base = f"  {name}: {count} detections (avg conf: {avg_conf:.3f})"
        if extra and self.verbose:
            extras = ", ".join(f"{k}={v}" for k, v in extra.items())
            base += f" [{extras}]"
        return base

    def model_summary(self, model_name: str, count: int) -> str:
        """Format a model detection summary."""
        return f"  {self.FOLDER} {model_name}: {count} detections"

    def file_saved(self, path: str) -> str:
        """Format a file saved message."""
        return f"{self.SAVE} Saved to {path}"

    def loading(self, what: str) -> str:
        """Format a loading message."""
        return f"{self.SEARCH} Loading {what}..."

    def merging(self, strategy: str) -> str:
        """Format a merging message."""
        return f"{self.MERGE} Running {strategy} strategy..."

    def stats_table(self, stats: Dict[str, Dict[str, Any]]) -> str:
        """Format a statistics table."""
        lines = [f"\n{self.CHART} Strategy Statistics:"]
        lines.append("-" * 50)

        for strategy, data in stats.items():
            total = data.get("total_detections", 0)
            avg_conf = data.get("avg_confidence", 0)
            classes = data.get("unique_classes", 0)
            lines.append(f"  {strategy}:")
            lines.append(f"    Detections: {total}")
            lines.append(f"    Avg Confidence: {avg_conf:.3f}")
            lines.append(f"    Unique Classes: {classes}")

        return "\n".join(lines)

    def evaluation_result(
        self, precision: float, recall: float, f1: float, tp: int = 0, fp: int = 0, fn: int = 0
    ) -> str:
        """Format evaluation metrics."""
        lines = [
            f"  Precision: {precision:.4f}",
            f"  Recall:    {recall:.4f}",
            f"  F1 Score:  {f1:.4f}",
        ]
        if tp or fp or fn:
            lines.extend(
                [
                    f"  TP: {tp}, FP: {fp}, FN: {fn}",
                ]
            )
        return "\n".join(lines)

    def progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Format a simple progress bar."""
        pct = current / total if total > 0 else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {current}/{total} ({pct * 100:.1f}%)"


# Global console instance for convenience
console = OutputFormatter()


def print_success(message: str) -> None:
    """Print a success message."""
    print(console.success(message))


def print_error(message: str) -> None:
    """Print an error message."""
    print(console.error(message))


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(console.warning(message))


def print_info(message: str) -> None:
    """Print an info message."""
    print(console.info(message))


__all__ = [
    "OutputFormatter",
    "console",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
]
