"""
Registry for analysis handlers.
"""

from typing import Dict, List, Optional, Type

from .base import AnalysisHandler


class AnalysisRegistry:
    """Registry for analysis handlers.

    Provides discovery and instantiation of analysis handlers by name.
    """

    _handlers: Dict[str, Type[AnalysisHandler]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an analysis handler.

        Args:
            name: Name to register the handler under

        Returns:
            Decorator function
        """

        def decorator(handler_class: Type[AnalysisHandler]) -> Type[AnalysisHandler]:
            handler_class.name = name
            cls._handlers[name] = handler_class
            return handler_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[AnalysisHandler]:
        """Get an analysis handler by name.

        Args:
            name: Handler name

        Returns:
            Handler instance or None if not found
        """
        cls._ensure_loaded()
        handler_class = cls._handlers.get(name)
        if handler_class:
            return handler_class()
        return None

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered handler names.

        Returns:
            Sorted list of handler names
        """
        cls._ensure_loaded()
        return sorted(cls._handlers.keys())

    @classmethod
    def list_with_descriptions(cls) -> Dict[str, str]:
        """List handlers with their descriptions.

        Returns:
            Dict mapping handler names to descriptions
        """
        cls._ensure_loaded()
        result = {}
        for name, handler_class in cls._handlers.items():
            instance = handler_class()
            result[name] = instance.description
        return result

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Ensure handlers are loaded."""
        if not cls._handlers:
            # Import to trigger registration
            from . import analysis  # noqa


def get_handler(name: str) -> Optional[AnalysisHandler]:
    """Get an analysis handler by name.

    Args:
        name: Handler name

    Returns:
        Handler instance or None
    """
    return AnalysisRegistry.get(name)


def list_handlers() -> List[str]:
    """List all available handler names.

    Returns:
        List of handler names
    """
    return AnalysisRegistry.list_all()


__all__ = ["AnalysisRegistry", "get_handler", "list_handlers"]
