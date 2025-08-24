"""Integration tests for Agents SDK error handling through the API layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gene.api import app
from gene.agent import set_client
from test_utils import ErrorMockOpenAIClient, OpenAIErrorFactory


class TestAPIAgentsSDKErrorHandling:
    """Test Agents SDK error handling through the /messages API endpoint."""

    def setup_method(self):
        """Set up test environment."""
        # Reset client state before each test
        set_client(None)

    def teardown_method(self):
        """Clean up after each test."""
        # Reset client state after each test
        set_client(None)

    def test_agent_rate_limit_error_response(self):
        """Test API response for Agents SDK rate limit errors."""
        # Inject client that raises agent rate limit error
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_rate_limit_error(retry_after=120, agent_id="agent_456")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "120"
        
        data = response.json()
        assert data["error"] == "Agents SDK rate limit exceeded. Please try again later."
        assert data["error_type"] == "agent_rate_limit"
        assert data["retry_after"] == 120
        assert data["agent_id"] == "agent_456"
        assert data["details"] == "Rate limit exceeded for agent requests"

    def test_agent_authentication_error_response(self):
        """Test API response for Agents SDK authentication errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_authentication_error(agent_id="agent_789")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 401
        
        data = response.json()
        assert data["error"] == "Invalid OpenAI Agents SDK credentials or insufficient permissions"
        assert data["error_type"] == "agent_authentication"
        assert data["agent_id"] == "agent_789"
        assert data["details"] == "Authentication failed for Agents SDK"

    def test_agent_connection_error_response(self):
        """Test API response for Agents SDK connection errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_connection_error(agent_id="agent_abc")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 502
        
        data = response.json()
        assert data["error"] == "Unable to connect to OpenAI Agents SDK service. Please check your internet connection and try again."
        assert data["error_type"] == "agent_connection"
        assert data["agent_id"] == "agent_abc"
        assert data["details"] == "Network connection failed for Agents SDK"

    def test_agent_timeout_error_response(self):
        """Test API response for Agents SDK timeout errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_timeout_error(agent_id="agent_def")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 504
        
        data = response.json()
        assert data["error"] == "Agents SDK request timed out. The service may be experiencing high load. Please try again."
        assert data["error_type"] == "agent_timeout"
        assert data["agent_id"] == "agent_def"
        assert data["details"] == "Request timeout for Agents SDK"

    def test_agent_conversation_error_response(self):
        """Test API response for Agents SDK conversation errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_conversation_error(agent_id="agent_ghi")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 502
        
        data = response.json()
        assert data["error"] == "Agent conversation context error. The conversation state may be corrupted. Please start a new conversation."
        assert data["error_type"] == "agent_conversation"
        assert data["agent_id"] == "agent_ghi"
        assert data["details"] == "Conversation context corruption in Agents SDK"

    def test_agent_processing_failure_response(self):
        """Test API response for Agents SDK processing failures."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_processing_failure(agent_id="agent_jkl")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 502
        
        data = response.json()
        assert data["error"] == "Agent processing failed. The AI service may be temporarily unavailable. Please try again."
        assert data["error_type"] == "agent_processing_failure"
        assert data["agent_id"] == "agent_jkl"
        assert data["details"] == "Agent processing pipeline failure"

    def test_agent_validation_error_response(self):
        """Test API response for Agents SDK validation errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_validation_error(agent_id="agent_mno")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 400
        
        data = response.json()
        assert data["error"] == "Invalid request to Agents SDK: malformed input"
        assert data["error_type"] == "agent_validation_error"
        assert data["agent_id"] == "agent_mno"
        assert data["details"] == "Request validation failed for Agents SDK"

    def test_agent_service_unavailable_response(self):
        """Test API response for Agents SDK service unavailable errors."""
        error_client = ErrorMockOpenAIClient(
            OpenAIErrorFactory.agent_service_unavailable_error(agent_id="agent_pqr")
        )
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 503
        
        data = response.json()
        assert data["error"] == "OpenAI Agents SDK service is temporarily unavailable. Please try again later."
        assert data["error_type"] == "agent_service_unavailable"
        assert data["agent_id"] == "agent_pqr"
        assert data["details"] == "Agents SDK service maintenance or overload"

    def test_error_response_without_optional_fields(self):
        """Test error response when optional fields are not present."""
        # Create error without agent_id and details
        from gene.openai_client import OpenAIError, OpenAIException
        
        error = OpenAIError(
            message="Basic error message",
            status_code=500,
            error_type="basic_error"
            # No agent_id, details, or retry_after
        )
        
        error_client = ErrorMockOpenAIClient(OpenAIException(error))
        set_client(error_client)
        
        client = TestClient(app)
        response = client.post("/messages", json={"body": "test message"})
        
        assert response.status_code == 500
        
        data = response.json()
        assert data["error"] == "Basic error message"
        assert data["error_type"] == "basic_error"
        # Optional fields should not be present in response
        assert "retry_after" not in data
        assert "agent_id" not in data
        assert "details" not in data

    def test_multiple_error_scenarios_consistency(self):
        """Test that all error scenarios return consistent response format."""
        error_scenarios = [
            (OpenAIErrorFactory.agent_rate_limit_error(), 429),
            (OpenAIErrorFactory.agent_authentication_error(), 401),
            (OpenAIErrorFactory.agent_connection_error(), 502),
            (OpenAIErrorFactory.agent_timeout_error(), 504),
            (OpenAIErrorFactory.agent_conversation_error(), 502),
            (OpenAIErrorFactory.agent_processing_failure(), 502),
            (OpenAIErrorFactory.agent_validation_error(), 400),
            (OpenAIErrorFactory.agent_service_unavailable_error(), 503),
        ]
        
        client = TestClient(app)
        
        for error_exception, expected_status in error_scenarios:
            # Reset and inject error client
            set_client(None)
            error_client = ErrorMockOpenAIClient(error_exception)
            set_client(error_client)
            
            response = client.post("/messages", json={"body": "test message"})
            
            assert response.status_code == expected_status
            
            data = response.json()
            # All responses should have these required fields
            assert "error" in data
            assert "error_type" in data
            assert isinstance(data["error"], str)
            assert isinstance(data["error_type"], str)
            
            # Agent-specific errors should have agent_id and details
            if data["error_type"].startswith("agent_"):
                assert "agent_id" in data
                assert "details" in data
                assert data["agent_id"] == "agent_123"  # Default from factory
                assert isinstance(data["details"], str)

    def test_retry_after_header_only_for_rate_limits(self):
        """Test that Retry-After header is only set for rate limit errors."""
        client = TestClient(app)
        
        # Test rate limit error has Retry-After header
        set_client(ErrorMockOpenAIClient(OpenAIErrorFactory.agent_rate_limit_error(retry_after=90)))
        response = client.post("/messages", json={"body": "test message"})
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "90"
        
        # Test non-rate-limit error does not have Retry-After header
        set_client(ErrorMockOpenAIClient(OpenAIErrorFactory.agent_authentication_error()))
        response = client.post("/messages", json={"body": "test message"})
        assert response.status_code == 401
        assert "Retry-After" not in response.headers