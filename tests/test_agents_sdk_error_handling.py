"""Comprehensive tests for Agents SDK error handling."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from gene.openai_client import (
    AgentsSDKWrapper,
    OpenAIError,
    OpenAIException,
    create_client,
)


class TestAgentsSDKErrorHandling:
    """Test comprehensive error handling for Agents SDK integration."""

    def test_agents_sdk_not_available_error(self):
        """Test error when Agents SDK is not installed."""
        with patch("gene.openai_client.Agent", None):
            with pytest.raises(RuntimeError, match="openai-agents package is required"):
                create_client("sk-test", "gpt-4")

    @patch("gene.openai_client.Agent")
    def test_agent_creation_failure(self, mock_agent_class):
        """Test error handling when agent creation fails."""
        mock_agent_class.side_effect = Exception("Agent creation failed")

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 500
        assert error.error_type == "agent_initialization"
        assert "Failed to initialize OpenAI Agent" in error.message
        assert error.details == "Agent creation failed"

    @patch("gene.openai_client.Agent")
    def test_rate_limit_error_handling(self, mock_agent_class):
        """Test Agents SDK rate limit error handling."""
        # Create mock agent that raises rate limit error
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Rate limit exceeded")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 429
        assert error.error_type == "agent_rate_limit"
        assert "Agents SDK rate limit exceeded" in error.message
        assert error.retry_after == 60  # Default retry_after
        assert error.details == "Rate limit exceeded"

    @patch("gene.openai_client.Agent")
    def test_rate_limit_error_with_custom_retry_after(self, mock_agent_class):
        """Test rate limit error with custom retry_after value."""

        # Create mock error with retry_after attribute
        class MockRateLimitError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.retry_after = 120

        mock_agent = Mock()
        mock_agent.process_message.side_effect = MockRateLimitError("Too many requests")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 429
        assert error.retry_after == 120
        assert error.error_type == "agent_rate_limit"

    @patch("gene.openai_client.Agent")
    def test_authentication_error_handling(self, mock_agent_class):
        """Test authentication error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Invalid API key")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 401
        assert error.error_type == "agent_authentication"
        assert "Invalid OpenAI Agents SDK credentials" in error.message
        assert error.details == "Invalid API key"

    @patch("gene.openai_client.Agent")
    def test_connection_error_handling(self, mock_agent_class):
        """Test network connection error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Connection refused")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "agent_connection"
        assert "Unable to connect to OpenAI Agents SDK service" in error.message
        assert error.details == "Connection refused"

    @patch("gene.openai_client.Agent")
    def test_timeout_error_handling(self, mock_agent_class):
        """Test timeout error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Request timed out")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 504
        assert error.error_type == "agent_timeout"
        assert "Agents SDK request timed out" in error.message
        assert error.details == "Request timed out"

    @patch("gene.openai_client.Agent")
    def test_conversation_error_handling(self, mock_agent_class):
        """Test conversation context error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception(
            "Conversation context corrupted"
        )
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "agent_conversation"
        assert "Agent conversation context error" in error.message
        assert error.details == "Conversation context corrupted"

    @patch("gene.openai_client.Agent")
    def test_agent_processing_failure_handling(self, mock_agent_class):
        """Test agent processing failure handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Agent processing failed")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "agent_processing_failure"
        assert "Agent processing failed" in error.message
        assert error.details == "Agent processing failed"

    @patch("gene.openai_client.Agent")
    def test_validation_error_handling(self, mock_agent_class):
        """Test validation error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Invalid request format")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 400
        assert error.error_type == "agent_validation_error"
        assert "Invalid request to Agents SDK" in error.message
        assert error.details == "Invalid request format"

    @patch("gene.openai_client.Agent")
    def test_service_unavailable_error_handling(self, mock_agent_class):
        """Test service unavailable error handling."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception(
            "Service temporarily unavailable"
        )
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 503
        assert error.error_type == "agent_service_unavailable"
        assert "OpenAI Agents SDK service is temporarily unavailable" in error.message
        assert error.details == "Service temporarily unavailable"

    @patch("gene.openai_client.Agent")
    def test_unexpected_error_handling(self, mock_agent_class):
        """Test handling of unexpected errors."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception("Unexpected error occurred")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 500
        assert error.error_type == "agent_internal_error"
        assert "Unexpected agent processing error" in error.message
        assert error.details == "Unexpected error occurred"

    @patch("gene.openai_client.Agent")
    def test_agent_id_included_in_errors(self, mock_agent_class):
        """Test that agent_id is included in error responses when available."""
        # Create mock agent with id attribute
        mock_agent = Mock()
        mock_agent.id = "agent_123"
        mock_agent.process_message.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.agent_id == "agent_123"

    @patch("gene.openai_client.Agent")
    def test_agent_id_fallback_to_agent_id_attribute(self, mock_agent_class):
        """Test fallback to agent_id attribute if id is not available."""
        # Create mock agent with agent_id attribute instead of id
        mock_agent = Mock()
        mock_agent.agent_id = "agent_456"
        # Remove id attribute
        del mock_agent.id
        mock_agent.process_message.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.agent_id == "agent_456"

    @patch("gene.openai_client.Agent")
    def test_empty_prompt_validation(self, mock_agent_class):
        """Test validation of empty prompts."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.complete("   ")

    @patch("gene.openai_client.Agent")
    def test_response_validation_error(self, mock_agent_class):
        """Test response validation error handling."""
        # Create mock agent that returns invalid response
        mock_agent = Mock()
        mock_agent.process_message.return_value = None
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.status_code == 502
        assert error.error_type == "invalid_agent_response"
        assert "Invalid response from Agents SDK" in error.message

    @patch("gene.openai_client.Agent")
    def test_successful_response_processing(self, mock_agent_class):
        """Test successful response processing with conversation context."""
        # Create mock agent with successful response
        mock_response = Mock()
        mock_response.content = "Test response from agent"
        mock_response.context = {"conversation_id": "conv_123"}

        mock_agent = Mock()
        mock_agent.process_message.return_value = mock_response
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")
        result = wrapper.complete("test prompt")

        assert result == "Test response from agent"
        assert wrapper._conversation_context == {"conversation_id": "conv_123"}

    @patch("gene.openai_client.Agent")
    def test_response_formats_handling(self, mock_agent_class):
        """Test handling of different response formats from Agents SDK."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        # Test string response
        mock_agent.process_message.return_value = "Direct string response"
        result = wrapper.complete("test prompt")
        assert result == "Direct string response"

        # Test response with message attribute (no content attribute)
        mock_response = Mock()
        mock_response.message = "Response with message attribute"
        # Ensure content attribute doesn't exist or is None
        mock_response.content = None
        mock_agent.process_message.return_value = mock_response
        result = wrapper.complete("test prompt")
        assert result == "Response with message attribute"

        # Test dict response
        mock_agent.process_message.return_value = {"content": "Dict response content"}
        result = wrapper.complete("test prompt")
        assert result == "Dict response content"


class TestAgentsSDKErrorFactory:
    """Test factory methods for creating Agents SDK specific errors."""

    def test_agent_rate_limit_error_creation(self):
        """Test creation of agent rate limit errors."""
        error = OpenAIError(
            message="Agents SDK rate limit exceeded. Please try again later.",
            status_code=429,
            retry_after=90,
            error_type="agent_rate_limit",
            agent_id="agent_123",
            details="Rate limit details",
        )

        assert (
            error.message == "Agents SDK rate limit exceeded. Please try again later."
        )
        assert error.status_code == 429
        assert error.retry_after == 90
        assert error.error_type == "agent_rate_limit"
        assert error.agent_id == "agent_123"
        assert error.details == "Rate limit details"

    def test_agent_authentication_error_creation(self):
        """Test creation of agent authentication errors."""
        error = OpenAIError(
            message="Invalid OpenAI Agents SDK credentials",
            status_code=401,
            error_type="agent_authentication",
            agent_id="agent_456",
            details="Auth error details",
        )

        assert error.message == "Invalid OpenAI Agents SDK credentials"
        assert error.status_code == 401
        assert error.error_type == "agent_authentication"
        assert error.agent_id == "agent_456"
        assert error.details == "Auth error details"

    def test_agent_connection_error_creation(self):
        """Test creation of agent connection errors."""
        error = OpenAIError(
            message="Unable to connect to OpenAI Agents SDK service",
            status_code=502,
            error_type="agent_connection",
            agent_id="agent_789",
            details="Connection error details",
        )

        assert error.message == "Unable to connect to OpenAI Agents SDK service"
        assert error.status_code == 502
        assert error.error_type == "agent_connection"
        assert error.agent_id == "agent_789"
        assert error.details == "Connection error details"


class TestErrorClassNameMatching:
    """Test error classification based on class names and error messages."""

    @patch("gene.openai_client.Agent")
    def test_rate_limit_error_class_name_matching(self, mock_agent_class):
        """Test rate limit error detection by class name."""

        class AgentRateLimitError(Exception):
            pass

        mock_agent = Mock()
        mock_agent.process_message.side_effect = AgentRateLimitError("Rate limited")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.error_type == "agent_rate_limit"

    @patch("gene.openai_client.Agent")
    def test_authentication_error_class_name_matching(self, mock_agent_class):
        """Test authentication error detection by class name."""

        class AgentAuthenticationError(Exception):
            pass

        mock_agent = Mock()
        mock_agent.process_message.side_effect = AgentAuthenticationError("Auth failed")
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.error_type == "agent_authentication"

    @patch("gene.openai_client.Agent")
    def test_error_message_content_matching(self, mock_agent_class):
        """Test error detection by message content."""
        mock_agent = Mock()
        mock_agent.process_message.side_effect = Exception(
            "quota exceeded for requests"
        )
        mock_agent_class.return_value = mock_agent

        wrapper = AgentsSDKWrapper("sk-test", "gpt-4")

        with pytest.raises(OpenAIException) as exc_info:
            wrapper.complete("test prompt")

        error = exc_info.value.error
        assert error.error_type == "agent_rate_limit"
