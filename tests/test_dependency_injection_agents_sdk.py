"""Tests for dependency injection support with Agents SDK mocking."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import (
    process_message,
    set_client,
    get_conversation_context,
    update_conversation_context,
    clear_conversation_context,
)
from gene.models import ActionResult, Message
from gene.openai_client import OpenAIException

from test_utils import (
    MockAgentsSDKClient,
    create_mock_agents_sdk_client,
    inject_mock_agents_sdk_client,
    OpenAIErrorFactory,
    ClientInjector,
    reset_client,
)


class TestMockAgentsSDKClient:
    """Test the MockAgentsSDKClient utility."""

    def test_default_behavior(self):
        """Test mock Agents SDK client with default configuration."""
        client = MockAgentsSDKClient()

        result = client.complete("test prompt")

        assert result == "Mock agent response"
        assert client.call_count == 1
        assert client.last_prompt == "test prompt"
        assert client.call_history == ["test prompt"]
        assert client._agent_id == "mock_agent_123"
        assert client._conversation_context["conversation_id"] == "mock_conv_456"
        assert client._conversation_context["turn_count"] == 1

    def test_custom_agent_metadata(self):
        """Test mock client with custom agent metadata."""
        client = MockAgentsSDKClient(
            agent_id="custom_agent_789", conversation_id="custom_conv_012"
        )

        result = client.complete("test")

        assert client._agent_id == "custom_agent_789"
        assert client._conversation_context["conversation_id"] == "custom_conv_012"
        assert client._conversation_context["turn_count"] == 1

    def test_specific_responses(self):
        """Test mock client with specific prompt-response mappings."""
        responses = {
            "hello": "Hi there from agent!",
            "goodbye": "See you later from agent!",
        }
        client = MockAgentsSDKClient(responses=responses)

        # Test specific responses
        assert client.complete("hello") == "Hi there from agent!"
        assert client.complete("goodbye") == "See you later from agent!"
        # Test default response for unmapped prompt
        assert client.complete("other") == "Mock agent response"

        assert client.call_count == 3
        assert client.call_history == ["hello", "goodbye", "other"]

    def test_exception_raising(self):
        """Test mock client that raises exceptions."""
        exception = ValueError("Agent processing error")
        client = MockAgentsSDKClient(should_raise=exception)

        with pytest.raises(ValueError, match="Agent processing error"):
            client.complete("test")

        assert client.call_count == 1
        assert client.call_history == ["test"]

    def test_conversation_context_management(self):
        """Test conversation context tracking."""
        client = MockAgentsSDKClient()

        # Initial state
        context = client.get_conversation_context()
        assert context["turn_count"] == 0
        assert context["conversation_id"] == "mock_conv_456"

        # After first message
        client.complete("first message")
        context = client.get_conversation_context()
        assert context["turn_count"] == 1
        assert context["last_prompt"] == "first message"

        # After second message
        client.complete("second message")
        context = client.get_conversation_context()
        assert context["turn_count"] == 2
        assert context["last_prompt"] == "second message"

        # Update context
        client.update_conversation_context({"custom_key": "custom_value"})
        context = client.get_conversation_context()
        assert context["custom_key"] == "custom_value"
        assert context["turn_count"] == 2  # Preserved

        # Clear context (using the mock client's method directly)
        client.clear_conversation_context()
        context = client.get_conversation_context()
        assert context["turn_count"] == 0
        assert "custom_key" not in context
        assert context["conversation_id"] == "mock_conv_456"  # Preserved

    def test_tool_suggestion_simulation(self):
        """Test tool suggestion and execution simulation."""
        client = MockAgentsSDKClient()

        # Test echo tool suggestion
        result = client.complete("suggest echo hello world")
        assert "echo tool" in result.lower()
        assert "Tool execution result: hello world" in result
        assert client.tool_suggestions == ["echo"]
        assert len(client.tool_executions) == 1
        assert client.tool_executions[0]["tool"] == "echo"
        assert client.tool_executions[0]["input"] == "hello world"
        assert client.tool_executions[0]["output"] == "hello world"

        # Test reverse tool suggestion
        client.reset()
        result = client.complete("suggest reverse hello")
        assert "reverse tool" in result.lower()
        assert "Tool execution result: olleh" in result
        assert client.tool_suggestions == ["reverse"]
        assert len(client.tool_executions) == 1
        assert client.tool_executions[0]["tool"] == "reverse"
        assert client.tool_executions[0]["input"] == "hello"
        assert client.tool_executions[0]["output"] == "olleh"

    def test_reset_functionality(self):
        """Test reset functionality."""
        client = MockAgentsSDKClient()

        # Make some calls and updates
        client.complete("test1")
        client.complete("test2")
        client.update_conversation_context({"custom": "value"})
        client.complete("suggest echo test")

        # Verify state before reset
        assert client.call_count == 3
        assert len(client.call_history) == 3
        assert client._conversation_context["turn_count"] == 3
        assert client._conversation_context["custom"] == "value"
        assert len(client.tool_suggestions) == 1
        assert len(client.tool_executions) == 1

        # Reset
        client.reset()

        # Verify state after reset
        assert client.call_count == 0
        assert len(client.call_history) == 0
        assert client.last_prompt is None
        assert client._conversation_context["turn_count"] == 0
        assert "custom" not in client._conversation_context
        assert len(client.tool_suggestions) == 0
        assert len(client.tool_executions) == 0
        # Agent ID and conversation ID should be preserved
        assert client._agent_id == "mock_agent_123"
        assert client._conversation_context["conversation_id"] == "mock_conv_456"


class TestAgentsSDKDependencyInjection:
    """Test dependency injection with Agents SDK mock clients."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    def test_set_client_with_mock_agents_sdk(self):
        """Test that set_client works correctly with mock Agents SDK clients."""
        mock_client = MockAgentsSDKClient("Injected agent response")

        # Inject the client
        set_client(mock_client)

        # Verify injection worked
        from gene.agent import _client

        assert _client is mock_client

        # Test message processing
        message = Message(body="test message")
        result = process_message(message)

        assert result.reply == "Injected agent response"
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "test message"

    def test_conversation_context_functions_with_mock(self):
        """Test conversation context functions work with mock Agents SDK client."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # Test get_conversation_context
        context = get_conversation_context()
        assert context["conversation_id"] == "mock_conv_456"
        assert context["turn_count"] == 0

        # Test update_conversation_context
        update_conversation_context({"test_key": "test_value", "turn_count": 5})
        context = get_conversation_context()
        assert context["test_key"] == "test_value"
        assert context["turn_count"] == 5
        assert context["conversation_id"] == "mock_conv_456"  # Preserved

        # Test clear_conversation_context
        clear_conversation_context()
        context = get_conversation_context()
        # After clearing, the context should be completely empty
        assert context == {}
        assert "test_key" not in context
        assert "turn_count" not in context
        assert "conversation_id" not in context

    def test_context_manager_injection(self):
        """Test client injection with context manager."""
        mock_client = MockAgentsSDKClient("Context manager response")

        # Verify no client initially
        import gene.agent

        assert gene.agent._client is None

        # Use context manager
        with ClientInjector(mock_client) as injected:
            assert gene.agent._client is mock_client
            assert injected is mock_client

            # Test message processing within context
            message = Message(body="test in context")
            result = process_message(message)
            assert result.reply == "Context manager response"

        # Verify restoration
        assert gene.agent._client is None

    def test_convenience_functions(self):
        """Test convenience functions for creating and injecting mock clients."""
        # Test create_mock_agents_sdk_client
        client = create_mock_agents_sdk_client(
            response="Custom response",
            agent_id="custom_agent",
            conversation_id="custom_conv",
        )
        assert isinstance(client, MockAgentsSDKClient)
        assert client.complete("test") == "Custom response"
        assert client._agent_id == "custom_agent"
        assert client._conversation_context["conversation_id"] == "custom_conv"

        # Test inject_mock_agents_sdk_client
        injected_client = inject_mock_agents_sdk_client(
            response="Injected response", agent_id="injected_agent"
        )

        # Verify injection
        from gene.agent import _client

        assert _client is injected_client

        # Test message processing
        message = Message(body="test injection")
        result = process_message(message)
        assert result.reply == "Injected response"
        assert injected_client.call_count == 1


class TestAgentsSDKIntegrationFlow:
    """Integration tests for full message processing flow with mock Agents SDK."""

    def setup_method(self):
        """Reset client state before each test."""
        reset_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_client()

    @patch("gene.config.settings")
    def test_full_message_processing_flow(self, mock_settings):
        """Test complete message processing flow with mock Agents SDK."""
        mock_settings.openai_model = "gpt-4"

        mock_client = MockAgentsSDKClient(
            "Comprehensive agent response",
            agent_id="integration_agent",
            conversation_id="integration_conv",
        )
        set_client(mock_client)

        message = Message(body="analyze this complex scenario")
        result = process_message(message)

        # Verify response
        assert result.reply == "Comprehensive agent response"

        # Verify enhanced metadata
        assert result.metadata["source"] == "agents_sdk"
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["structured_output"] == True
        assert result.metadata["agent_id"] == "integration_agent"
        assert result.metadata["conversation_id"] == "integration_conv"
        # The context includes: conversation_id, turn_count, created_at, last_prompt
        assert result.metadata["conversation_context_size"] == 4
        assert result.metadata["tokens_used"] == 150
        assert "processing_time" in result.metadata

        # Verify client state
        assert mock_client.call_count == 1
        assert mock_client.last_prompt == "analyze this complex scenario"
        assert mock_client._conversation_context["turn_count"] == 1
        assert (
            mock_client._conversation_context["last_prompt"]
            == "analyze this complex scenario"
        )

    def test_conversation_context_across_requests(self):
        """Test conversation context management across multiple requests."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # First message
        message1 = Message(body="first message")
        result1 = process_message(message1)

        context = get_conversation_context()
        assert context["turn_count"] == 1
        assert context["last_prompt"] == "first message"

        # Second message
        message2 = Message(body="second message")
        result2 = process_message(message2)

        context = get_conversation_context()
        assert context["turn_count"] == 2
        assert context["last_prompt"] == "second message"

        # Verify both calls were made
        assert mock_client.call_count == 2
        assert mock_client.call_history == ["first message", "second message"]

    def test_tool_precedence_with_agents_sdk(self):
        """Test that tool precedence is maintained with Agents SDK."""
        mock_client = MockAgentsSDKClient("Should not be called")
        set_client(mock_client)

        # Use reverse tool - should not call Agents SDK
        message = Message(body="reverse hello")
        result = process_message(message)

        assert result.reply == "olleh"
        assert result.metadata["source"] == "tool"
        assert result.metadata["tool"] == "reverse"
        # Agents SDK client should not have been called
        assert mock_client.call_count == 0

    def test_error_handling_with_mock_agents_sdk(self):
        """Test error handling with mock Agents SDK exceptions."""
        # Test rate limit error
        rate_limit_error = OpenAIErrorFactory.agent_rate_limit_error(
            retry_after=120, agent_id="test_agent"
        )
        mock_client = MockAgentsSDKClient(should_raise=rate_limit_error)
        set_client(mock_client)

        message = Message(body="test message")

        with pytest.raises(OpenAIException) as exc_info:
            process_message(message)

        assert exc_info.value.error.status_code == 429
        assert exc_info.value.error.retry_after == 120
        assert exc_info.value.error.error_type == "agent_rate_limit"
        assert exc_info.value.error.agent_id == "test_agent"
        assert mock_client.call_count == 1

    def test_agent_tool_integration_simulation(self):
        """Test agent and tool integration scenarios."""
        mock_client = MockAgentsSDKClient()
        set_client(mock_client)

        # Test agent suggesting echo tool usage
        message = Message(body="suggest echo hello world")
        result = process_message(message)

        # Should get agent response with tool execution
        assert "echo tool" in result.reply.lower()
        assert "Tool execution result: hello world" in result.reply

        # Verify conversation context includes tool usage
        context = get_conversation_context()
        assert "last_tool_used" in context
        assert context["last_tool_used"]["tool_name"] == "echo"
        assert context["last_tool_used"]["input"] == "hello world"
        assert context["last_tool_used"]["output"] == "hello world"

        # Verify tool tracking in mock client
        assert mock_client.tool_suggestions == ["echo"]
        assert len(mock_client.tool_executions) == 1

    def test_response_format_consistency(self):
        """Test response format consistency across processing paths."""
        mock_client = MockAgentsSDKClient("Agent response")
        set_client(mock_client)

        # Test agent response format
        agent_message = Message(body="agent test")
        agent_result = process_message(agent_message)
        assert isinstance(agent_result, ActionResult)
        assert isinstance(agent_result.reply, str)
        assert isinstance(agent_result.metadata, dict)
        assert agent_result.metadata["source"] == "agents_sdk"

        # Test tool response format
        tool_message = Message(body="echo test")
        tool_result = process_message(tool_message)
        assert isinstance(tool_result, ActionResult)
        assert isinstance(tool_result.reply, str)
        assert isinstance(tool_result.metadata, dict)
        assert tool_result.metadata["source"] == "tool"

        # Test placeholder response format
        set_client(None)
        placeholder_message = Message(body="placeholder test")
        placeholder_result = process_message(placeholder_message)
        assert isinstance(placeholder_result, ActionResult)
        assert isinstance(placeholder_result.reply, str)
        assert isinstance(placeholder_result.metadata, dict)
        assert placeholder_result.metadata["source"] == "placeholder"

    def test_configuration_and_initialization(self):
        """Test configuration loading and agent initialization scenarios."""
        # Test with different agent configurations
        clients = [
            MockAgentsSDKClient(agent_id="agent_1", conversation_id="conv_1"),
            MockAgentsSDKClient(agent_id="agent_2", conversation_id="conv_2"),
            MockAgentsSDKClient(agent_id="agent_3", conversation_id="conv_3"),
        ]

        for i, client in enumerate(clients):
            set_client(client)

            message = Message(body=f"test message {i}")
            result = process_message(message)

            # Verify agent-specific metadata
            assert result.metadata["agent_id"] == f"agent_{i+1}"
            assert result.metadata["conversation_id"] == f"conv_{i+1}"

            # Verify client state
            assert client.call_count == 1
            assert client.last_prompt == f"test message {i}"

    def test_backward_compatibility(self):
        """Test backward compatibility with existing API contracts."""
        mock_client = MockAgentsSDKClient("Backward compatible response")
        set_client(mock_client)

        # Test that existing Message and ActionResult interfaces work
        message = Message(body="compatibility test")
        result = process_message(message)

        # Verify ActionResult structure
        assert hasattr(result, "reply")
        assert hasattr(result, "metadata")
        assert isinstance(result.reply, str)
        assert isinstance(result.metadata, dict)

        # Verify required metadata fields
        assert "source" in result.metadata
        assert "processing_time" in result.metadata

        # Verify enhanced metadata doesn't break existing contracts
        enhanced_fields = [
            "agent_id",
            "conversation_id",
            "structured_output",
            "tokens_used",
        ]
        for field in enhanced_fields:
            if field in result.metadata:
                assert result.metadata[field] is not None
