"""Tool that reverses text after the ``reverse`` prefix."""

from .base import BaseTool


class ReverseTool(BaseTool):
    """Reverse the text that follows ``reverse ``."""

    name = "reverse"

    def can_handle(self, query: str) -> bool:
        return query.startswith("reverse ")

    def handle(self, query: str) -> str:
        return query[len("reverse ") :][::-1]
