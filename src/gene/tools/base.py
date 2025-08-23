"""Base classes for tool implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must implement :meth:`can_handle` and :meth:`handle`. The
    registry uses :meth:`can_handle` to route incoming queries to an
    appropriate tool.
    """

    name: str = "base"

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Return ``True`` if this tool can process ``query``."""

    @abstractmethod
    def handle(self, query: str) -> str:
        """Process ``query`` and return a result."""
