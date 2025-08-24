"""Tests for the agent module with enhanced metadata and error propagation."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import (
    OpenAIClient, 
    process_message, 
    set_client,
    get_conversation_context,
    update_conversation_context,
    clear_conversation_context
)
from gene.models import ActionResult, Message
from gene.openai_client import OpenAIError, OpenAIException, AgentsSDKWrapper


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, response: str = "Mock AI response"):
        self.response = response
        self.call_count = 0
        self.last_prompt = None
    
    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


class MockAgentsSDKWrapper:
    """Mock Agents SDK wrapper for testing enhanced metadata."""
    
    def __init__(self, response: str = "Mock agent response", agent_id: str = "test_agent_123"):
        self.response = response
        self.call_count = 0
        self.last_prompt = None
        self._agent_id = agent_id
        self._conversation_context = {"conversation_id": "conv_456", "turn_count": 1}
        self._agent = Mock()
        self._agent.last_response_tokens = 150
    
    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        # Simulate conversation context update
        self._conversation_context["turn_count"] += 1
        return self.response


class ErrorMockOpenAIClient:
    """Mock OpenAI client that raises exceptions."""
    
    def __init__(self, exception: Exception):
        self.exception = exception
        self.call_count = 0
    
    def complete(self, prompt: str) -> str:
        self.call_count += 1
        raise self.exception


class TestProcessMessageMetadata:
    """Test enhanced metadata in process_message responses."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_placeholder_response_metadata(self):
        """Test that placeholder responses include correct metadata with processing time."""
        set_client(None)
        message = Message(body="test message")
        
        result = process_message(message)
        
        assert result.reply == "This is a placeholder response."
        assert result.metadata["source"] == "placeholder"
        assert "processing_time" in result.metadata
        assert isinstance(result.metadata["processing_time"], float)
        assert result.metadata["processing_time"] >= 0
    
    @patch('gene.config.settings')
    def test_openai_response_metadata(self, mock_settings):
        """Test that OpenAI responses include model and source metadata with processing time."""
        mock_settings.openai_model = "gpt-4"
        mock_client = MockOpenAIClient("AI response")
        set_client(mock_client)
        
        message = Message(body="test message")
        result = process_message(message)
        
        assert result.reply == "AI response"
        assert result.metadata["source"] == "agents_sdk"
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["structured_output"] == True
        assert "processing_time" in result.metadata
        assert isinstance(result.metadata["processing_time"], float)
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "test message"
    
    def test_tool_response_metadata(self):
        """Test that tool responses include tool name and source metadata with processing time."""
        mock_client = MockOpenAIClient("Should not be called")
        set_client(mock_client)
        
        # Use reverse tool
        message = Message(body="reverse hello")
        result = process_message(message)
        
        assert result.reply == "olleh"
        assert result.metadata["source"] == "tool"
        assert result.metadata["tool"] == "reverse"
        assert "processing_time" in result.metadata
        assert isinstance(result.metadata["processing_time"], float)
        # Ensure OpenAI client was not called when tool handled the message
        assert mock_client.call_count == 0
    
    @patch('gene.config.settings')
    def test_agents_sdk_enhanced_metadata(self, mock_settings):
        """Test that Agents SDK responses include enhanced metadata."""
        mock_settings.openai_model = "gpt-4"
        mock_client = MockAgentsSDKWrapper("Enhanced agent response")
        set_client(mock_client)
        
        message = Message(body="test message")
        result = process_message(message)
        
        assert result.reply == "Enhanced agent response"
        assert result.metadata["source"] == "agents_sdk"
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["structured_output"] == True
        assert result.metadata["agent_id"] == "test_agent_123"
        assert result.metadata["conversation_id"] == "conv_456"
        assert result.metadata["conversation_context_size"] == 2
        assert result.metadata["tokens_used"] == 150
        assert "processing_time" in result.metadata
        assert isinstance(result.metadata["processing_time"], float)
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "test message"


class TestErrorPropagation:
    """Test proper error propagation from OpenAI client."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_openai_exception_propagation(self):
        """Test that OpenAI exceptions are properly propagated."""
        error = OpenAIError(
            message="Rate limit exceeded",
            status_code=429,
            retry_after=60,
            error_type="rate_limit"
        )
        exception = OpenAIException(error)
        error_client = ErrorMockOpenAIClient(exception)
        set_client(error_client)
        
        message = Message(body="test message")
        
        with pytest.raises(OpenAIException) as exc_info:
            process_message(message)
        
        assert exc_info.value.error.message == "Rate limit exceeded"
        assert exc_info.value.error.status_code == 429
        assert exc_info.value.error.retry_after == 60
        assert exc_info.value.error.error_type == "rate_limit"
        assert error_client.call_count == 1
    
    def test_generic_exception_propagation(self):
        """Test that generic exceptions are properly propagated."""
        generic_error = RuntimeError("Network connection failed")
        error_client = ErrorMockOpenAIClient(generic_error)
        set_client(error_client)
        
        message = Message(body="test message")
        
        with pytest.raises(RuntimeError) as exc_info:
            process_message(message)
        
        assert str(exc_info.value) == "Network connection failed"
        assert error_client.call_count == 1
    
    def test_tool_errors_not_affected(self):
        """Test that tool processing is not affected by OpenAI client errors."""
        # Set up a client that would error, but shouldn't be called
        error = OpenAIError(
            message="API Error",
            status_code=500,
            error_type="api_error"
        )
        exception = OpenAIException(error)
        error_client = ErrorMockOpenAIClient(exception)
        set_client(error_client)
        
        # Use a tool - should not trigger OpenAI client
        message = Message(body="reverse test")
        result = process_message(message)
        
        assert result.reply == "tset"
        assert result.metadata["source"] == "tool"
        assert result.metadata["tool"] == "reverse"
        assert "processing_time" in result.metadata
        # Ensure error client was never called
        assert error_client.call_count == 0


class TestConversationContextManagement:
    """Test conversation context management functions."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    def test_get_conversation_context_no_client(self):
        """Test getting conversation context when no client is set."""
        set_client(None)
        context = get_conversation_context()
        assert context == {}
    
    def test_get_conversation_context_non_agents_sdk(self):
        """Test getting conversation context with non-Agents SDK client."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        context = get_conversation_context()
        assert context == {}
    
    def test_get_conversation_context_agents_sdk(self):
        """Test getting conversation context with Agents SDK client."""
        mock_client = MockAgentsSDKWrapper()
        set_client(mock_client)
        context = get_conversation_context()
        assert context == {"conversation_id": "conv_456", "turn_count": 1}
    
    def test_update_conversation_context_no_client(self):
        """Test updating conversation context when no client is set."""
        set_client(None)
        # Should not raise an error, just log a warning
        update_conversation_context({"new_key": "new_value"})
        context = get_conversation_context()
        assert context == {}
    
    def test_update_conversation_context_non_agents_sdk(self):
        """Test updating conversation context with non-Agents SDK client."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        # Should not raise an error, just log a warning
        update_conversation_context({"new_key": "new_value"})
        context = get_conversation_context()
        assert context == {}
    
    def test_update_conversation_context_agents_sdk(self):
        """Test updating conversation context with Agents SDK client."""
        mock_client = MockAgentsSDKWrapper()
        set_client(mock_client)
        
        # Update context
        update_conversation_context({"new_key": "new_value", "turn_count": 5})
        
        context = get_conversation_context()
        assert context["conversation_id"] == "conv_456"
        assert context["turn_count"] == 5  # Updated
        assert context["new_key"] == "new_value"  # Added
    
    def test_update_conversation_context_invalid_input(self):
        """Test updating conversation context with invalid input."""
        mock_client = MockAgentsSDKWrapper()
        set_client(mock_client)
        
        # Should not raise an error, just log a warning
        update_conversation_context("not a dict")
        
        context = get_conversation_context()
        # Original context should be unchanged
        assert context == {"conversation_id": "conv_456", "turn_count": 1}
    
    def test_clear_conversation_context_no_client(self):
        """Test clearing conversation context when no client is set."""
        set_client(None)
        # Should not raise an error, just log a warning
        clear_conversation_context()
    
    def test_clear_conversation_context_non_agents_sdk(self):
        """Test clearing conversation context with non-Agents SDK client."""
        mock_client = MockOpenAIClient()
        set_client(mock_client)
        # Should not raise an error, just log a warning
        clear_conversation_context()
    
    def test_clear_conversation_context_agents_sdk(self):
        """Test clearing conversation context with Agents SDK client."""
        mock_client = MockAgentsSDKWrapper()
        set_client(mock_client)
        
        # Verify initial context exists
        context = get_conversation_context()
        assert len(context) > 0
        
        # Clear context
        clear_conversation_context()
        
        # Verify context is cleared
        context = get_conversation_context()
        assert context == {}


class TestProcessMessageIntegration:
    """Integration tests for process_message with various scenarios."""
    
    def setup_method(self):
        """Reset client state before each test."""
        set_client(None)
    
    def teardown_method(self):
        """Clean up after each test."""
        set_client(None)
    
    @patch('gene.config.settings')
    def test_message_with_metadata_preserved(self, mock_settings):
        """Test that input message metadata doesn't interfere with response metadata."""
        mock_settings.openai_model = "gpt-3.5-turbo"
        mock_client = MockOpenAIClient("Response with context")
        set_client(mock_client)
        
        # Message with its own metadata
        message = Message(
            body="analyze this", 
            metadata={"source_system": "monitoring", "priority": "high"}
        )
        
        result = process_message(message)
        
        assert result.reply == "Response with context"
        assert result.metadata["source"] == "agents_sdk"
        assert result.metadata["model"] == "gpt-3.5-turbo"
        assert result.metadata["structured_output"] == True
        assert "processing_time" in result.metadata
        # Input message metadata should not affect response metadata
        assert "source_system" not in result.metadata
        assert "priority" not in result.metadata
    
    def test_empty_message_body_handling(self):
        """Test handling of empty message bodies."""
        mock_client = MockOpenAIClient("Should handle empty")
        set_client(mock_client)
        
        message = Message(body="")
        
        # This should propagate any validation errors from the client
        # The actual validation happens in the OpenAI client wrapper
        result = process_message(message)
        
        assert result.reply == "Should handle empty"
        assert result.metadata["source"] == "agents_sdk"
    
    @patch('gene.agent.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs for different processing paths."""
        # Test placeholder logging
        set_client(None)
        message = Message(body="test")
        process_message(message)
        mock_logger.info.assert_called_with("Processing message with placeholder response")
        
        # Test tool logging
        mock_logger.reset_mock()
        message = Message(body="reverse test")
        process_message(message)
        mock_logger.info.assert_called_with("Processing message with tool: reverse")
        
        # Test Agents SDK logging
        mock_logger.reset_mock()
        mock_client = MockOpenAIClient("AI response")
        set_client(mock_client)
        message = Message(body="test")
        process_message(message)
        mock_logger.info.assert_called_with("Processing message with Agents SDK")