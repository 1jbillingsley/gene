"""Tool registry and discovery utilities."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from .base import BaseTool

__all__ = ["BaseTool", "register", "get_tool", "handle"]

_REGISTRY: list[BaseTool] = []


def _discover_tools() -> None:
    """Import all modules in this package to populate the registry."""
    package = __name__
    for module_info in iter_modules(__path__):
        if module_info.name == "base":
            continue
        module = import_module(f"{package}.{module_info.name}")
        for obj in module.__dict__.values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
            ):
                _REGISTRY.append(obj())


def register(tool: BaseTool) -> None:
    """Manually register a tool instance."""
    _REGISTRY.append(tool)


def get_tool(query: str) -> BaseTool | None:
    """Return the first tool able to handle ``query``."""
    for tool in _REGISTRY:
        if tool.can_handle(query):
            return tool
    return None


def handle(query: str) -> str:
    """Dispatch ``query`` to an appropriate tool and return its result."""
    tool = get_tool(query)
    if tool is None:
        raise ValueError(f"No tool found to handle: {query!r}")
    return tool.handle(query)


# Discover tools on import.
_discover_tools()
