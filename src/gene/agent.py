"""Core agent interface for message processing."""

from __future__ import annotations

from typing import Protocol

from .models import ActionResult, Message
from .tools import get_tool


class OpenAIClient(Protocol):
    """Minimal protocol for an OpenAI-like client.

    This protocol defines the subset of methods the agent expects from an
    OpenAI client. A real implementation can be provided at runtime via
    :func:`set_client`.
    """

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        """Generate a completion for the given prompt."""
        ...


_client: OpenAIClient | None = None


def set_client(client: OpenAIClient | None) -> None:
    """Inject the OpenAI client used for message processing.

    ``client`` may be ``None`` to reset the agent to placeholder mode.
    """
    global _client
    _client = client


def process_message(message: Message) -> ActionResult:
    """Process a message and return the agent's response.

    If a tool is able to handle the message, it is invoked first. Otherwise,
    if no client has been configured, a canned placeholder response is
    returned. When a real client is available, the message body is forwarded to
    it and its output is wrapped in an :class:`ActionResult`.
    """

    tool = get_tool(message.body)
    if tool is not None:
        return ActionResult(
            reply=tool.handle(message.body), metadata={"tool": tool.name}
        )

    if _client is None:
        reply = "This is a placeholder response."
    else:  # pragma: no cover - depends on external service
        reply = _client.complete(message.body)
    return ActionResult(reply=reply)
