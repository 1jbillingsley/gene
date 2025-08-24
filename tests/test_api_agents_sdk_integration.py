"""Integration tests for /messages endpoint with Agents SDK scenarios."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from gene.api import app
from gene.agent import set_client
from gene.openai_client import OpenAIException, OpenAIError


class MockAgentsSDKClient:
    """Mock Agents SDK client for testing enhanced API responses."""
    
    def __init__(self, agent_id: str = "test_agent_123", conversation_id: str = "conv_456"):
        self._agent_id = agent_id
        self._conversation_context = {"conversation_id": conversation_id}
        self._agent = Mock()
        self._agent.last_response_tokens = 150
        
    def complete(self, prompt: str) -> str:
        """Mock completion that returns enhanced response."""
        return f"Mock Agents SDK response for: {prompt}"


class MockAgentsSDKClientWithError:
    """Mock Agents SDK client that raises specific errors for testing."""
    
    def __init__(self, error_type: str = "rate_limit", agent_id: str = "error_agent_789"):
        self._agent_id = agent_id
        self._conversation_context = {"conversation_id": "error_conv_123"}
        self.error_type = error_type
        
    def complete(self, prompt: str) -> str:
        """Mock completion that raises specific errors."""
        if self.error_type == "rate_limit":
            error = OpenAIError(
                message="Agents SDK rate limit exceeded. Please try again later.",
                status_code=429,
                retry_after=60,
                error_type="agent_rate_limit",
                agent_id=self._agent_id
            )
            raise OpenAIException(error)
        elif self.error_type == "conversation":
            error = OpenAIError(
                message="Agent conversation context error. The conversation state may be corrupted.",
                status_code=502,
                error_type="agent_conversation",
                agent_id=self._agent_id,
                details="Conversation memory overflow"
            )
            raise OpenAIException(error)
        elif self.error_type == "authentication":
            error = OpenAIError(
                message="Invalid OpenAI Agents SDK credentials or insufficient permissions",
                status_code=401,
                error_type="agent_authentication",
                agent_id=self._agent_id
            )
            raise OpenAIException(error)
        else:
            error = OpenAIError(
                message="Unexpected agent processing error. Please try again or contact support.",
                status_code=500,
                error_type="agent_internal_error",
                agent_id=self._agent_id,
                details="Mock internal error for testing"
            )
            raise OpenAIException(error)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_agents_sdk_client():
    """Create mock Agents SDK client."""
    return MockAgentsSDKClient()


@pytest.fixture
def mock_agents_sdk_error_client():
    """Create mock Agents SDK client that raises errors."""
    return MockAgentsSDKClientWithError()


class TestAgentsSDKResponseHeaders:
    """Test enhanced response headers for Agents SDK responses."""
    
    def test_agents_sdk_response_includes_conversation_headers(self, client, mock_agents_sdk_client):
        """Test that Agents SDK responses include conversation context headers."""
        set_client(mock_agents_sdk_client)
        
        response = client.post("/messages", json={"body": "Test message for agents SDK"})
        
        assert response.status_code == 200
        
        # Check response headers for Agents SDK metadata
        assert "X-Agent-ID" in response.headers
        assert response.headers["X-Agent-ID"] == "test_agent_123"
        
        assert "X-Conversation-ID" in response.headers
        assert response.headers["X-Conversation-ID"] == "conv_456"
        
        assert "X-Processing-Source" in response.headers
        assert response.headers["X-Processing-Source"] == "agents_sdk"
        
        assert "X-Structured-Output" in response.headers
        assert response.headers["X-Structured-Output"] == "true"
    
    def test_agents_sdk_response_includes_model_header(self, client, mock_agents_sdk_client):
        """Test that Agents SDK responses include model information in headers."""
        set_client(mock_agents_sdk_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 200
        assert "X-Model" in response.headers
        # The model comes from settings.openai_model which defaults to "gpt-3.5-turbo"
        assert response.headers["X-Model"] in ["gpt-3.5-turbo", "gpt-4"]
    
    def test_tool_response_no_agent_headers(self, client):
        """Test that tool responses don't include agent-specific headers."""
        set_client(None)  # Ensure no client is set to force tool usage
        
        response = client.post("/messages", json={"body": "echo test message"})
        
        assert response.status_code == 200
        
        # Tool responses should not have agent-specific headers
        assert "X-Agent-ID" not in response.headers
        assert "X-Conversation-ID" not in response.headers
        
        # But should still have processing source
        data = response.json()
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "echo"
    
    def test_placeholder_response_no_agent_headers(self, client):
        """Test that placeholder responses don't include agent-specific headers."""
        set_client(None)  # No client and no tool match
        
        response = client.post("/messages", json={"body": "regular message"})
        
        assert response.status_code == 200
        
        # Placeholder responses should not have agent-specific headers
        assert "X-Agent-ID" not in response.headers
        assert "X-Conversation-ID" not in response.headers
        
        # But should still indicate processing source
        data = response.json()
        assert data["metadata"]["source"] == "placeholder"


class TestAgentsSDKErrorHandling:
    """Test enhanced error handling for Agents SDK specific errors."""
    
    def test_agents_sdk_rate_limit_error_with_headers(self, client):
        """Test Agents SDK rate limit error includes proper headers and retry information."""
        error_client = MockAgentsSDKClientWithError(error_type="rate_limit")
        set_client(error_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 429
        
        # Check error response structure
        data = response.json()
        assert data["error"] == "Agents SDK rate limit exceeded. Please try again later."
        assert data["error_type"] == "agent_rate_limit"
        assert data["retry_after"] == 60
        assert data["agent_id"] == "error_agent_789"
        
        # Check response headers
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"
        assert "X-Agent-ID" in response.headers
        assert response.headers["X-Agent-ID"] == "error_agent_789"
        assert "X-Error-Source" in response.headers
        assert response.headers["X-Error-Source"] == "agents_sdk"
    
    def test_agents_sdk_conversation_error_with_details(self, client):
        """Test Agents SDK conversation error includes detailed error information."""
        error_client = MockAgentsSDKClientWithError(error_type="conversation")
        set_client(error_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 502
        
        # Check error response structure
        data = response.json()
        assert data["error"] == "Agent conversation context error. The conversation state may be corrupted."
        assert data["error_type"] == "agent_conversation"
        assert data["agent_id"] == "error_agent_789"
        assert data["details"] == "Conversation memory overflow"
        
        # Check response headers
        assert "X-Agent-ID" in response.headers
        assert response.headers["X-Agent-ID"] == "error_agent_789"
        assert "X-Error-Source" in response.headers
        assert response.headers["X-Error-Source"] == "agents_sdk"
    
    def test_agents_sdk_authentication_error(self, client):
        """Test Agents SDK authentication error handling."""
        error_client = MockAgentsSDKClientWithError(error_type="authentication")
        set_client(error_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 401
        
        # Check error response structure
        data = response.json()
        assert data["error"] == "Invalid OpenAI Agents SDK credentials or insufficient permissions"
        assert data["error_type"] == "agent_authentication"
        assert data["agent_id"] == "error_agent_789"
        
        # Check response headers
        assert "X-Agent-ID" in response.headers
        assert response.headers["X-Agent-ID"] == "error_agent_789"
        assert "X-Error-Source" in response.headers
        assert response.headers["X-Error-Source"] == "agents_sdk"
    
    def test_agents_sdk_internal_error_with_details(self, client):
        """Test Agents SDK internal error includes error details."""
        error_client = MockAgentsSDKClientWithError(error_type="internal")
        set_client(error_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 500
        
        # Check error response structure
        data = response.json()
        assert data["error"] == "Unexpected agent processing error. Please try again or contact support."
        assert data["error_type"] == "agent_internal_error"
        assert data["agent_id"] == "error_agent_789"
        assert data["details"] == "Mock internal error for testing"
        
        # Check response headers
        assert "X-Agent-ID" in response.headers
        assert response.headers["X-Agent-ID"] == "error_agent_789"
        assert "X-Error-Source" in response.headers
        assert response.headers["X-Error-Source"] == "agents_sdk"


class TestAgentsSDKResponseFormat:
    """Test consistent ActionResult format with enhanced agent metadata."""
    
    def test_agents_sdk_response_format_consistency(self, client, mock_agents_sdk_client):
        """Test that Agents SDK responses maintain consistent ActionResult format."""
        set_client(mock_agents_sdk_client)
        
        response = client.post("/messages", json={"body": "Test message for format check"})
        
        assert response.status_code == 200
        
        # Check response structure matches ActionResult
        data = response.json()
        assert "reply" in data
        assert "metadata" in data
        
        # Check enhanced metadata for Agents SDK
        metadata = data["metadata"]
        assert metadata["source"] == "agents_sdk"
        assert metadata["agent_id"] == "test_agent_123"
        assert metadata["conversation_id"] == "conv_456"
        assert metadata["structured_output"] is True
        assert "processing_time" in metadata
        assert "model" in metadata
        assert "tokens_used" in metadata
        assert metadata["tokens_used"] == 150
    
    def test_agents_sdk_metadata_includes_conversation_context_size(self, client, mock_agents_sdk_client):
        """Test that Agents SDK metadata includes conversation context information."""
        set_client(mock_agents_sdk_client)
        
        response = client.post("/messages", json={"body": "Test conversation context"})
        
        assert response.status_code == 200
        
        data = response.json()
        metadata = data["metadata"]
        
        # Check conversation context metadata
        assert "conversation_context_size" in metadata
        assert metadata["conversation_context_size"] == 1  # Only conversation_id in mock
    
    def test_response_format_backward_compatibility(self, client):
        """Test that response format remains backward compatible for non-Agents SDK responses."""
        set_client(None)  # Use placeholder mode
        
        response = client.post("/messages", json={"body": "regular message"})
        
        assert response.status_code == 200
        
        # Check that response still follows ActionResult format
        data = response.json()
        assert "reply" in data
        assert "metadata" in data
        
        # Placeholder responses should have minimal metadata
        metadata = data["metadata"]
        assert metadata["source"] == "placeholder"
        assert "processing_time" in metadata
        
        # Should not have agent-specific fields
        assert "agent_id" not in metadata
        assert "conversation_id" not in metadata
        assert "structured_output" not in metadata


class TestAgentsSDKConversationContext:
    """Test conversation context management in API responses."""
    
    def test_conversation_context_persistence_across_requests(self, client):
        """Test that conversation context is maintained across multiple requests."""
        # Create a mock client that tracks conversation state
        mock_client = MockAgentsSDKClient(agent_id="persistent_agent", conversation_id="persistent_conv")
        set_client(mock_client)
        
        # First request
        response1 = client.post("/messages", json={"body": "First message"})
        assert response1.status_code == 200
        assert response1.headers["X-Conversation-ID"] == "persistent_conv"
        
        # Update conversation context to simulate state change
        mock_client._conversation_context["message_count"] = 1
        mock_client._conversation_context["last_message"] = "First message"
        
        # Second request should maintain conversation context
        response2 = client.post("/messages", json={"body": "Second message"})
        assert response2.status_code == 200
        assert response2.headers["X-Conversation-ID"] == "persistent_conv"
        
        # Check that conversation context size increased
        data2 = response2.json()
        assert data2["metadata"]["conversation_context_size"] == 3  # conversation_id + message_count + last_message
    
    def test_conversation_context_headers_missing_when_unavailable(self, client):
        """Test that conversation context headers are omitted when not available."""
        # Create a simple mock client class instead of using Mock
        class SimpleClient:
            def __init__(self):
                self._conversation_context = {}
                self._agent = None
                # Deliberately don't set _agent_id
            
            def complete(self, prompt: str) -> str:
                return "Response without context"
        
        mock_client = SimpleClient()
        set_client(mock_client)
        
        response = client.post("/messages", json={"body": "Test message"})
        
        assert response.status_code == 200
        
        # Should not have conversation context headers
        assert "X-Agent-ID" not in response.headers
        assert "X-Conversation-ID" not in response.headers
        
        # But should still have basic processing information
        data = response.json()
        assert data["metadata"]["source"] == "agents_sdk"


class TestAgentsSDKIntegrationScenarios:
    """Test complete integration scenarios with Agents SDK."""
    
    def test_complete_agents_sdk_message_processing_flow(self, client, mock_agents_sdk_client):
        """Test the complete message processing flow with Agents SDK."""
        set_client(mock_agents_sdk_client)
        
        test_message = "Analyze this complex business scenario"
        
        response = client.post("/messages", json={
            "body": test_message,
            "metadata": {"source": "business_system", "priority": "high"}
        })
        
        assert response.status_code == 200
        
        # Verify response structure
        data = response.json()
        assert data["reply"] == f"Mock Agents SDK response for: {test_message}"
        
        # Verify enhanced metadata
        metadata = data["metadata"]
        assert metadata["source"] == "agents_sdk"
        assert metadata["agent_id"] == "test_agent_123"
        assert metadata["conversation_id"] == "conv_456"
        assert metadata["structured_output"] is True
        assert metadata["tokens_used"] == 150
        
        # Verify response headers
        assert response.headers["X-Agent-ID"] == "test_agent_123"
        assert response.headers["X-Conversation-ID"] == "conv_456"
        assert response.headers["X-Processing-Source"] == "agents_sdk"
        assert response.headers["X-Structured-Output"] == "true"
    
    def test_agents_sdk_with_tool_fallback_precedence(self, client, mock_agents_sdk_client):
        """Test that tools still take precedence over Agents SDK when applicable."""
        set_client(mock_agents_sdk_client)
        
        # Send a message that should be handled by echo tool
        response = client.post("/messages", json={"body": "echo test message"})
        
        assert response.status_code == 200
        
        # Should be handled by tool, not Agents SDK
        data = response.json()
        assert data["reply"] == "test message"  # Echo tool response
        assert data["metadata"]["source"] == "tool"
        assert data["metadata"]["tool"] == "echo"
        
        # Should not have agent-specific headers
        assert "X-Agent-ID" not in response.headers
        assert "X-Conversation-ID" not in response.headers
    
    def test_agents_sdk_error_recovery_and_logging(self, client):
        """Test error recovery and proper logging for Agents SDK errors."""
        error_client = MockAgentsSDKClientWithError(error_type="rate_limit")
        set_client(error_client)
        
        with patch('gene.api.logger') as mock_logger:
            response = client.post("/messages", json={"body": "Test error handling"})
            
            assert response.status_code == 429
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            log_call = mock_logger.error.call_args[0][0]
            assert "OpenAI/Agents SDK error in /messages endpoint" in log_call
            
            # Verify error response format
            data = response.json()
            assert data["error_type"] == "agent_rate_limit"
            assert data["agent_id"] == "error_agent_789"