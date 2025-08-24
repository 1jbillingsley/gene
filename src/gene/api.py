"""FastAPI application for the gene project."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import process_message, set_client
from .config import settings
from .models import Message
from .openai_client import get_client, OpenAIException

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


@app.post("/messages", response_model=None)
async def create_message(message: Message):
    """Accept a message and process it via the agent.

    Returns:
        ActionResult: The processed message response with metadata and conversation context headers

    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    try:
        # Validate message body is not empty
        if not message.body or not message.body.strip():
            logger.warning("Empty message body received")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Message body cannot be empty",
                    "error_type": "validation_error",
                },
            )

        # Process the message and get the result
        result = process_message(message)

        # Extract conversation context and agent metadata for response headers
        headers = {}
        if result.metadata:
            # Add agent identification headers
            if "agent_id" in result.metadata:
                headers["X-Agent-ID"] = result.metadata["agent_id"]

            # Add conversation context headers
            if "conversation_id" in result.metadata:
                headers["X-Conversation-ID"] = result.metadata["conversation_id"]

            # Add processing source information
            if "source" in result.metadata:
                headers["X-Processing-Source"] = result.metadata["source"]

            # Add model information for Agents SDK responses
            if "model" in result.metadata:
                headers["X-Model"] = result.metadata["model"]

            # Add structured output flag
            if "structured_output" in result.metadata:
                headers["X-Structured-Output"] = str(
                    result.metadata["structured_output"]
                ).lower()

        # Return ActionResult with enhanced headers for Agents SDK responses
        if headers:
            # Convert ActionResult to dict for JSONResponse with headers
            response_content = {"reply": result.reply, "metadata": result.metadata}
            return JSONResponse(
                status_code=200, content=response_content, headers=headers
            )
        else:
            # Return standard ActionResult for non-Agents SDK responses
            return result

    except OpenAIException as e:
        # Handle structured OpenAI/Agents SDK errors with appropriate HTTP status codes
        logger.error(
            f"OpenAI/Agents SDK error in /messages endpoint: {e.error.message}"
        )

        # Prepare error response with consistent structure
        error_response = {"error": e.error.message, "error_type": e.error.error_type}

        # Add optional fields if present
        if e.error.retry_after is not None:
            error_response["retry_after"] = e.error.retry_after

        if e.error.agent_id is not None:
            error_response["agent_id"] = e.error.agent_id

        if e.error.details is not None:
            error_response["details"] = e.error.details

        # Prepare response headers for agent-specific errors
        headers = {}

        # Add retry_after header for rate limit errors
        if e.error.retry_after is not None:
            headers["Retry-After"] = str(e.error.retry_after)

        # Add agent identification headers for agent-specific errors
        if e.error.agent_id is not None:
            headers["X-Agent-ID"] = e.error.agent_id

        # Add error source header to distinguish between standard OpenAI and Agents SDK errors
        if e.error.error_type.startswith("agent_"):
            headers["X-Error-Source"] = "agents_sdk"
        else:
            headers["X-Error-Source"] = "openai_api"

        return JSONResponse(
            status_code=e.error.status_code, content=error_response, headers=headers
        )

    except ValueError as e:
        # Handle configuration and validation errors
        logger.error(f"Configuration error in /messages endpoint: {e}")
        error_message = str(e)

        # Determine if this is a configuration error based on message content
        if "api key" in error_message.lower() or "openai" in error_message.lower():
            return JSONResponse(
                status_code=500,
                content={"error": error_message, "error_type": "configuration_error"},
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": error_message, "error_type": "validation_error"},
            )

    except RuntimeError as e:
        # Handle runtime errors (like missing OpenAI package)
        logger.error(f"Runtime error in /messages endpoint: {e}")
        return JSONResponse(
            status_code=500, content={"error": str(e), "error_type": "runtime_error"}
        )

    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Unexpected error in /messages endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "error_type": "internal_error"},
        )
