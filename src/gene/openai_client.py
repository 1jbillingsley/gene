"""Factory for the OpenAI API client used by the agent."""

from __future__ import annotations

import logging

from .config import settings
from .agent import OpenAIClient

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - importlib and runtime errors
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

_client: OpenAIClient | None = None


class _OpenAIWrapper:
    """Light wrapper implementing :class:`OpenAIClient`."""

    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:  # pragma: no cover - network call
        response = self._client.responses.create(model=self._model, input=prompt)
        return response.output[0].content[0].text


def get_client() -> OpenAIClient | None:
    """Return a cached OpenAI client instance.

    If the required credentials are missing or the OpenAI package is not
    installed, ``None`` is returned and the agent will fall back to placeholder
    responses.
    """

    global _client
    if _client is not None:
        return _client

    if not settings.openai_api_key or not settings.openai_model:
        logger.warning("OpenAI credentials not configured; running in placeholder mode")
        return None

    if OpenAI is None:
        raise RuntimeError("openai package is required but not installed")

    _client = _OpenAIWrapper(settings.openai_api_key, settings.openai_model)
    return _client
