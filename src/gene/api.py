"""FastAPI application for the gene project."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request

from .agent import process_message, set_client
from .config import settings
from .models import ActionResult, Message
from .openai_client import get_client

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)

set_client(get_client())

app = FastAPI()

__all__ = ["app"]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log basic information about incoming requests and outgoing responses."""
    logger.info("Request %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response %s %s", response.status_code, request.url.path)
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns a simple status dictionary indicating the API is running.
    """
    return {"status": "ok"}


@app.post("/messages")
async def create_message(message: Message) -> ActionResult:
    """Accept a message and process it via the agent."""

    return process_message(message)
