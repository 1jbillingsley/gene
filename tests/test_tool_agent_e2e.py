"""End-to-end tests for tool and agent integration."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from gene.agent import set_client, clear_conversation_context
from gene.api import app


class MockAgentForE2E:
    """Mock agent that simulates real Agents SDK behavior for E2E testing."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.id = "e2e_agent_123"
        self.agent_id = "e2e_agent_123"
        self.kwargs = kwargs
        self.last_message = None
        self.last_context = None
        self.last_tools = None
        
    def process_message(self, message: str, context: dict = None, available_tools: list = None):
        """Mock process_message that demonstrates tool awareness."""
        self.last_message = message
        self.last_context = context or {}
        self.last_tools = available_tools or []
        
        # Simulate agent being aware of tools and suggesting their use
        if "help with echo" in message.lower():
            return Mock(content="I can help you with that. You might want to use echo tool for echoing text.")
        elif "help with reverse" in message.lower():
            return Mock(content="For reversing text, you should use reverse tool with your input.")
        elif "what tools" in message.lower():
            tool_names = [tool.get("name", "unknown") for tool in self.last_tools]
            return Mock(content=f"I have access to these tools: {', '.join(tool_names)}. I can help you use them effectively.")
        else:
            return Mock(content=f"Agent response to: {message}")


class TestToolAgentEndToEnd:
    """End-to-end tests for tool and agent integration through the API."""
    
    def setup_method(self):
        """Set up test environment."""
        set_client(None)
        clear_conversation_context()
    
    def teardown_method(self):
        """Clean up after tests."""
        set_client(None)
        clear_conversation_context()
    
    def test_direct_tool_usage_bypasses_agent(self):
        """Test that direct tool usage still works and bypasses agent processing."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Direct echo tool usage
            response = client.post("/messages", json={"body": "echo Hello World"})
            assert response.status_code == 200
            data = response.json()
            
            # Should be handled by tool, not agent
            assert data["reply"] == "Hello World"
            assert data["metadata"]["source"] == "tool"
            assert data["metadata"]["tool"] == "echo"
            
            # Agent should not have been called
            assert wrapper._agent is None or wrapper._agent.last_message is None
    
    def test_agent_tool_awareness_through_api(self):
        """Test that agent is aware of available tools when processing messages."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Ask agent about available tools
            response = client.post("/messages", json={"body": "what tools are available?"})
            assert response.status_code == 200
            data = response.json()
            
            # Should be handled by agent with tool awareness
            assert data["metadata"]["source"] == "agents_sdk"
            assert "echo" in data["reply"]
            assert "reverse" in data["reply"]
            assert "tools" in data["reply"].lower()
    
    def test_agent_suggests_tool_usage(self):
        """Test that agent can suggest appropriate tool usage."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Ask for help with echoing
            response = client.post("/messages", json={"body": "I need help with echo functionality"})
            assert response.status_code == 200
            data = response.json()
            
            # Should suggest echo tool usage
            assert data["metadata"]["source"] == "agents_sdk"
            assert "echo tool" in data["reply"].lower()
            assert "echoing" in data["reply"].lower()
    
    def test_tool_precedence_maintained_with_agents_sdk(self):
        """Test that tool precedence is maintained even with Agents SDK integration."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Test echo tool precedence
            response = client.post("/messages", json={"body": "echo test message"})
            assert response.status_code == 200
            data = response.json()
            assert data["reply"] == "test message"
            assert data["metadata"]["source"] == "tool"
            
            # Test reverse tool precedence
            response = client.post("/messages", json={"body": "reverse hello"})
            assert response.status_code == 200
            data = response.json()
            assert data["reply"] == "olleh"
            assert data["metadata"]["source"] == "tool"
            
            # Test agent processing when no tool matches
            response = client.post("/messages", json={"body": "analyze this data"})
            assert response.status_code == 200
            data = response.json()
            assert "Agent response to: analyze this data" in data["reply"]
            assert data["metadata"]["source"] == "agents_sdk"
    
    def test_agent_enhanced_metadata_includes_tool_info(self):
        """Test that agent responses include enhanced metadata about tool availability."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Process message through agent
            response = client.post("/messages", json={"body": "general question"})
            assert response.status_code == 200
            data = response.json()
            
            # Verify enhanced metadata
            metadata = data["metadata"]
            assert metadata["source"] == "agents_sdk"
            assert metadata["agent_id"] == "e2e_agent_123"
            assert metadata["structured_output"] is True
            assert "processing_time" in metadata
            assert metadata["model"] == "gpt-4"
    
    def test_conversation_context_persistence_across_requests(self):
        """Test that conversation context persists across multiple API requests."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # First request - establish context
            response1 = client.post("/messages", json={"body": "start conversation"})
            assert response1.status_code == 200
            
            # Manually add some context (simulating agent behavior)
            wrapper._conversation_context["user_name"] = "test_user"
            
            # Second request - tool usage (should not affect context)
            response2 = client.post("/messages", json={"body": "echo test"})
            assert response2.status_code == 200
            assert response2.json()["metadata"]["source"] == "tool"
            
            # Third request - agent processing (should preserve context)
            response3 = client.post("/messages", json={"body": "continue conversation"})
            assert response3.status_code == 200
            assert response3.json()["metadata"]["source"] == "agents_sdk"
            
            # Verify context was preserved
            assert "user_name" in wrapper._conversation_context
            assert wrapper._conversation_context["user_name"] == "test_user"
    
    def test_error_handling_with_tool_and_agent_integration(self):
        """Test error handling when both tools and agents are available."""
        with patch('gene.openai_client.Agent', MockAgentForE2E):
            from gene.openai_client import AgentsSDKWrapper
            wrapper = AgentsSDKWrapper("test_key", "gpt-4")
            set_client(wrapper)
            
            client = TestClient(app)
            
            # Test with valid tool usage
            response = client.post("/messages", json={"body": "echo valid"})
            assert response.status_code == 200
            assert response.json()["reply"] == "valid"
            
            # Test with valid agent usage
            response = client.post("/messages", json={"body": "help me understand"})
            assert response.status_code == 200
            assert response.json()["metadata"]["source"] == "agents_sdk"
            
            # Test with empty message (should return 400 but not crash)
            response = client.post("/messages", json={"body": ""})
            assert response.status_code == 400  # Should return bad request for empty body
            assert "error" in response.json()  # Should include error message