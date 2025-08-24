import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from gene.agent import set_client
from gene.api import app
from gene.openai_client import OpenAIError, OpenAIException


class DummyClient:
    def complete(self, prompt: str) -> str:
        return f"echo: {prompt}"


class ErrorClient:
    """Client that raises OpenAI exceptions for testing error handling."""
    
    def __init__(self, error: OpenAIError):
        self.error = error
    
    def complete(self, prompt: str) -> str:
        raise OpenAIException(self.error)


def test_messages_endpoint_placeholder_response():
    set_client(None)
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "This is a placeholder response."
    assert data["metadata"]["source"] == "placeholder"
    assert "processing_time" in data["metadata"]


@patch('gene.config.settings')
def test_messages_endpoint_uses_injected_client(mock_settings):
    mock_settings.openai_model = "gpt-3.5-turbo"
    set_client(DummyClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "echo: hello"
    assert data["metadata"]["source"] == "agents_sdk"
    assert data["metadata"]["model"] == "gpt-3.5-turbo"
    assert data["metadata"]["structured_output"] is True
    assert "processing_time" in data["metadata"]
    set_client(None)


def test_messages_endpoint_uses_tool_when_available():
    set_client(DummyClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "reverse abc"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "cba"
    assert data["metadata"]["source"] == "tool"
    assert data["metadata"]["tool"] == "reverse"
    assert "processing_time" in data["metadata"]
    set_client(None)


def test_messages_endpoint_handles_rate_limit_error():
    """Test that rate limit errors return proper HTTP status and headers."""
    error = OpenAIError(
        message="Rate limit exceeded",
        status_code=429,
        retry_after=60,
        error_type="rate_limit"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"
    data = resp.json()
    assert data["error"] == "Rate limit exceeded"
    assert data["error_type"] == "rate_limit"
    assert data["retry_after"] == 60
    set_client(None)


def test_messages_endpoint_handles_api_error():
    """Test that API errors return proper HTTP status."""
    error = OpenAIError(
        message="Invalid API key",
        status_code=401,
        error_type="authentication"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"] == "Invalid API key"
    assert data["error_type"] == "authentication"
    assert "retry_after" not in data
    set_client(None)


def test_messages_endpoint_handles_connection_error():
    """Test that connection errors return proper HTTP status."""
    error = OpenAIError(
        message="Unable to connect to OpenAI service",
        status_code=502,
        error_type="connection"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 502
    data = resp.json()
    assert data["error"] == "Unable to connect to OpenAI service"
    assert data["error_type"] == "connection"
    set_client(None)


def test_messages_endpoint_handles_timeout_error():
    """Test that timeout errors return proper HTTP status."""
    error = OpenAIError(
        message="OpenAI request timed out. Please try again.",
        status_code=504,
        error_type="timeout"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 504
    data = resp.json()
    assert data["error"] == "OpenAI request timed out. Please try again."
    assert data["error_type"] == "timeout"
    set_client(None)


def test_messages_endpoint_handles_bad_request_error():
    """Test that bad request errors return proper HTTP status."""
    error = OpenAIError(
        message="Invalid request to OpenAI API",
        status_code=400,
        error_type="bad_request"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "Invalid request to OpenAI API"
    assert data["error_type"] == "bad_request"
    set_client(None)


def test_messages_endpoint_handles_empty_message_body():
    """Test that empty message bodies return validation error."""
    set_client(DummyClient())
    client = TestClient(app)
    
    # Test completely empty body
    resp = client.post("/messages", json={"body": ""})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "Message body cannot be empty"
    assert data["error_type"] == "validation_error"
    
    # Test whitespace-only body
    resp = client.post("/messages", json={"body": "   "})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "Message body cannot be empty"
    assert data["error_type"] == "validation_error"
    
    set_client(None)


def test_messages_endpoint_handles_configuration_error():
    """Test that configuration errors return proper HTTP status."""
    
    class ConfigErrorClient:
        def complete(self, prompt: str) -> str:
            raise ValueError("OpenAI API key not configured")
    
    set_client(ConfigErrorClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "OpenAI API key not configured"
    assert data["error_type"] == "configuration_error"
    set_client(None)


def test_messages_endpoint_handles_runtime_error():
    """Test that runtime errors return proper HTTP status."""
    
    class RuntimeErrorClient:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("openai package is required but not installed")
    
    set_client(RuntimeErrorClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "openai package is required but not installed"
    assert data["error_type"] == "runtime_error"
    set_client(None)


def test_messages_endpoint_handles_unexpected_error():
    """Test that unexpected errors return proper HTTP status."""
    
    class UnexpectedErrorClient:
        def complete(self, prompt: str) -> str:
            raise KeyError("Unexpected error occurred")
    
    set_client(UnexpectedErrorClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "Internal server error"
    assert data["error_type"] == "internal_error"
    set_client(None)


def test_messages_endpoint_maintains_consistent_success_format():
    """Test that successful responses maintain consistent ActionResult format."""
    set_client(DummyClient())
    client = TestClient(app)
    
    resp = client.post("/messages", json={"body": "hello world"})
    assert resp.status_code == 200
    data = resp.json()
    
    # Verify ActionResult structure
    assert "reply" in data
    assert "metadata" in data
    assert isinstance(data["reply"], str)
    assert isinstance(data["metadata"], dict)
    assert data["metadata"]["source"] == "agents_sdk"
    
    set_client(None)


def test_messages_endpoint_error_response_structure():
    """Test that all error responses have consistent structure."""
    error = OpenAIError(
        message="Test error",
        status_code=500,
        error_type="test_error"
    )
    set_client(ErrorClient(error))
    
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "test"})
    
    assert resp.status_code == 500
    data = resp.json()
    
    # Verify error response structure
    assert "error" in data
    assert "error_type" in data
    assert isinstance(data["error"], str)
    assert isinstance(data["error_type"], str)
    # retry_after should only be present for rate limit errors
    assert "retry_after" not in data
    
    set_client(None)
