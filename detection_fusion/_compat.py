_TORCH_AVAILABLE = None
_MATPLOTLIB_AVAILABLE = None
_RICH_AVAILABLE = None


def _check_torch() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch

            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_matplotlib() -> bool:
    global _MATPLOTLIB_AVAILABLE
    if _MATPLOTLIB_AVAILABLE is None:
        try:
            import matplotlib

            _MATPLOTLIB_AVAILABLE = True
        except ImportError:
            _MATPLOTLIB_AVAILABLE = False
    return _MATPLOTLIB_AVAILABLE


def _check_rich() -> bool:
    global _RICH_AVAILABLE
    if _RICH_AVAILABLE is None:
        try:
            import rich

            _RICH_AVAILABLE = True
        except ImportError:
            _RICH_AVAILABLE = False
    return _RICH_AVAILABLE


def has_torch() -> bool:
    return _check_torch()


def has_matplotlib() -> bool:
    return _check_matplotlib()


def has_rich() -> bool:
    return _check_rich()


def require_torch(feature_name: str = "This feature") -> None:
    if not has_torch():
        raise ImportError(
            f"{feature_name} requires PyTorch. Install with: pip install detection-fusion[torch]"
        )
