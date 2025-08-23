"""Dummy echo tool used for testing the tool registry."""

from .base import BaseTool


class EchoTool(BaseTool):
    """Tool that echoes the text after the ``echo`` prefix."""

    name = "echo"

    def can_handle(self, query: str) -> bool:
        return query.startswith("echo ")

    def handle(self, query: str) -> str:
        return query[5:]
