"""Data models for the gene API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Generic message payload.

    This model provides the core fields shared by all messages processed by
    the API. It can be extended or subclassed to support additional
    message types as the system grows.
    """

    body: str = Field(
        ..., description="Main textual content of the message to interpret."
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional dictionary with context about the message, such as "
            "environment details or source identifiers. This allows external "
            "integrations to supply information that may influence processing "
            "logic."
        ),
    )
