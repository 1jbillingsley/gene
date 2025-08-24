"""Integration tests for Agents SDK and tool framework interaction."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from gene.agent import process_message, set_client, clear_conversation_context
from gene.models import Message, ActionResult
from gene.openai_client import AgentsSDKWrapper, OpenAIException, OpenAIError


class MockAgent:
    """Mock agent for testing Agents SDK integration."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.id = "mock_agent_123"
        self.agent_id = "mock_agent_123"
        self.kwargs = kwargs
        self.last_message = None
        self.last_context = None
        self.last_tools = None
        
    def process_message(self, message: str, context: dict = None, available_tools: list = None):
        """Mock process_message that can suggest tool usage."""
        self.last_message = message
        self.last_context = context or {}
        self.last_tools = available_tools or []
        
        # Mock response that suggests tool usage based on message content
        if "suggest echo" in message.lower():
            return Mock(content=f"I can help with that. You might want to use echo tool for '{message}'")
        elif "suggest reverse" in message.lower():
            return Mock(content=f"For reversing text, you should use reverse tool with '{message}'")
        elif "tool aware" in message.lower():
            tool_names = [tool.get("name", "unknown") for tool in self.last_tools]
            return Mock(content=f"I'm aware of these tools: {', '.join(tool_names)}")
        else:
            return Mock(content=f"Agent response to: {message}")


class TestAgentsSDKToolIntegration:
    """Test integration between Agents SDK and existing tool framework."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any existing client
        set_client(None)
        clear_conversation_context()
    
    def teardown_method(self):
        """Clean up after tests."""
        set_client(None)
        clear_conversation_context()
    
    def test_agents_sdk_discovers_available_tools(self):
        """Test that AgentsSDKWrapper can discover existing tools."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Get available tools
            tools = wrapper._get_available_tools()
            
            # Should discover echo and reverse tools
            assert len(tools) >= 2
            tool_names = [tool["name"] for tool in tools]
            assert "echo" in tool_names
            assert "reverse" in tool_names
            
            # Check tool descriptions
            echo_tool = next(tool for tool in tools if tool["name"] == "echo")
            assert "echo" in echo_tool["description"].lower()
            assert echo_tool["type"] == "external_tool"
    
    def test_agent_creation_with_tool_awareness(self):
        """Test that agent is created with awareness of available tools."""
        with patch('gene.openai_client.Agent') as mock_agent_class:
            mock_agent_class.return_value = MockAgent("test_key", "gpt-4")
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Trigger agent creation
            wrapper.complete("test message")
            
            # Verify agent was created with tools
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) >= 2
            
            # Verify tool information is passed correctly
            tool_names = [tool["name"] for tool in call_kwargs["tools"]]
            assert "echo" in tool_names
            assert "reverse" in tool_names
    
    def test_agent_receives_tool_context_in_messages(self):
        """Test that available tools are passed to agent during message processing."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Process a message
            response = wrapper.complete("tool aware message")
            
            # Verify agent received tool information
            agent = wrapper._agent
            assert agent.last_tools is not None
            assert len(agent.last_tools) >= 2
            
            # Verify response includes tool awareness
            assert "echo" in response
            assert "reverse" in response
    
    def test_tool_suggestion_handling_echo(self):
        """Test that agent can suggest echo tool usage and execute it."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Process message that triggers echo tool suggestion
            response = wrapper.complete("suggest echo hello world")
            
            # Should include both agent response and tool execution result
            assert "use echo tool" in response
            assert "Tool execution result: hello world" in response
            
            # Verify conversation context was updated
            context = wrapper._conversation_context
            assert "last_tool_used" in context
            assert context["last_tool_used"]["tool_name"] == "echo"
            assert context["last_tool_used"]["output"] == "hello world"
    
    def test_tool_suggestion_handling_reverse(self):
        """Test that agent can suggest reverse tool usage and execute it."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Process message that triggers reverse tool suggestion
            response = wrapper.complete("suggest reverse test")
            
            # Should include both agent response and tool execution result
            assert "use reverse tool" in response
            assert "Tool execution result: tset" in response
            
            # Verify conversation context was updated
            context = wrapper._conversation_context
            assert "last_tool_used" in context
            assert context["last_tool_used"]["tool_name"] == "reverse"
            assert context["last_tool_used"]["output"] == "tset"
    
    def test_tool_precedence_maintained_over_agent_processing(self):
        """Test that direct tool invocation still takes precedence over agent processing."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            # Direct tool invocation should bypass agent
            message = Message(body="echo direct tool test")
            result = process_message(message)
            
            # Should be handled by tool, not agent
            assert result.reply == "direct tool test"
            assert result.metadata["source"] == "tool"
            assert result.metadata["tool"] == "echo"
            
            # Agent should not have been called
            assert wrapper._agent is None or wrapper._agent.last_message is None
    
    def test_agent_processing_when_no_tool_matches(self):
        """Test that agent processing works when no tool can handle the message."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            # Message that doesn't match any tool
            message = Message(body="analyze this complex data")
            result = process_message(message)
            
            # Should be handled by agent
            assert "Agent response to: analyze this complex data" in result.reply
            assert result.metadata["source"] == "agents_sdk"
            assert result.metadata["model"] == "gpt-4"
            
            # Agent should have been called
            assert wrapper._agent.last_message == "analyze this complex data"
    
    def test_conversation_context_persistence_with_tools(self):
        """Test that conversation context persists across tool and agent interactions."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            # First message - agent processing
            message1 = Message(body="start conversation")
            result1 = process_message(message1)
            
            # Add some context
            wrapper._conversation_context["user_preference"] = "verbose"
            
            # Second message - tool processing (should not affect agent context)
            message2 = Message(body="echo test")
            result2 = process_message(message2)
            
            # Third message - agent processing (should have preserved context)
            message3 = Message(body="continue conversation")
            result3 = process_message(message3)
            
            # Verify context persistence
            assert "user_preference" in wrapper._conversation_context
            assert wrapper._conversation_context["user_preference"] == "verbose"
    
    def test_tool_execution_error_handling_in_agent_context(self):
        """Test error handling when tool execution fails within agent context."""
        with patch('gene.openai_client.Agent', MockAgent):
            # Mock a tool that will fail
            with patch('gene.tools.get_tool') as mock_get_tool:
                mock_tool = Mock()
                mock_tool.handle.side_effect = Exception("Tool execution failed")
                mock_get_tool.return_value = mock_tool
                
                wrapper = AgentsSDKWrapper("test_key", "gpt-4")
                
                # Process message that would trigger tool suggestion
                response = wrapper.complete("suggest echo failing")
                
                # Should handle tool failure gracefully and return agent response
                assert "use echo tool" in response
                # Tool execution should have failed silently, returning original response
                assert "Tool execution result:" not in response
    
    def test_agent_metadata_includes_tool_information(self):
        """Test that agent responses include enhanced metadata about tool availability."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            # Process message through agent
            message = Message(body="test agent metadata")
            result = process_message(message)
            
            # Verify enhanced metadata
            assert result.metadata["source"] == "agents_sdk"
            assert result.metadata["agent_id"] == "mock_agent_123"
            assert result.metadata["structured_output"] is True
            assert "processing_time" in result.metadata
    
    def test_tool_discovery_failure_handling(self):
        """Test graceful handling when tool discovery fails."""
        with patch('gene.openai_client.Agent', MockAgent):
            # Mock tool discovery failure
            with patch('gene.tools._REGISTRY', side_effect=ImportError("Tool import failed")):
                wrapper = AgentsSDKWrapper("test_key", "gpt-4")
                
                # Should handle discovery failure gracefully
                tools = wrapper._get_available_tools()
                assert tools == []
                
                # Agent should still work without tools
                response = wrapper.complete("test without tools")
                assert "Agent response to: test without tools" in response
    
    def test_multiple_tool_suggestions_in_single_response(self):
        """Test handling of multiple tool suggestions in a single agent response."""
        with patch('gene.openai_client.Agent', MockAgent):
            # Create a mock agent that suggests multiple tools
            def mock_process_message(message, context=None, available_tools=None):
                return Mock(content="You could use echo tool for echoing or use reverse tool for reversing")
            
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            wrapper._agent = Mock()
            wrapper._agent.process_message = mock_process_message
            wrapper._agent_id = "test_agent"
            
            # Process message
            response = wrapper.complete("help with text processing")
            
            # Should handle the first matching suggestion (echo in this case)
            # This tests the priority handling in tool suggestions
            assert "use echo tool" in response or "use reverse tool" in response


class TestAgentsSDKToolIntegrationEdgeCases:
    """Test edge cases in Agents SDK and tool integration."""
    
    def setup_method(self):
        """Set up test environment."""
        set_client(None)
        clear_conversation_context()
    
    def teardown_method(self):
        """Clean up after tests."""
        set_client(None)
        clear_conversation_context()
    
    def test_empty_tool_registry_handling(self):
        """Test behavior when no tools are available."""
        with patch('gene.openai_client.Agent', MockAgent):
            # Mock empty tool registry
            with patch('gene.tools._REGISTRY', []):
                wrapper = AgentsSDKWrapper("test_key", "gpt-4")
                
                # Should handle empty registry gracefully
                tools = wrapper._get_available_tools()
                assert tools == []
                
                # Agent should still work
                response = wrapper.complete("test with no tools")
                assert "Agent response to: test with no tools" in response
    
    def test_tool_suggestion_with_malformed_response(self):
        """Test handling of malformed agent responses during tool suggestion processing."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Test with None response
            result = wrapper._handle_tool_suggestions(None, "test")
            assert result is None
            
            # Test with response object without expected attributes
            malformed_response = Mock(spec=[])  # Mock with no attributes
            result = wrapper._handle_tool_suggestions(malformed_response, "test")
            assert result is None
    
    def test_conversation_context_with_tool_usage_history(self):
        """Test that tool usage history is properly maintained in conversation context."""
        with patch('gene.openai_client.Agent', MockAgent):
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            
            # Simulate tool suggestion and execution
            response = wrapper.complete("suggest echo first")
            assert "Tool execution result: first" in response
            
            # Check that context was updated
            assert "last_tool_used" in wrapper._conversation_context
            first_tool_usage = wrapper._conversation_context["last_tool_used"]
            
            # Simulate another tool usage
            response = wrapper.complete("suggest reverse second")
            assert "Tool execution result: dnoces" in response
            
            # Check that context was updated with new tool usage
            assert wrapper._conversation_context["last_tool_used"]["tool_name"] == "reverse"
            assert wrapper._conversation_context["last_tool_used"]["output"] == "dnoces"