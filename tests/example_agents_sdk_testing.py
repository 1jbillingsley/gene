#!/usr/bin/env python3
"""
Example demonstrating how to use the mock Agents SDK client for testing.

This script shows various testing scenarios with the MockAgentsSDKClient.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import process_message, set_client, get_conversation_context, update_conversation_context, clear_conversation_context
from gene.models import Message
from test_utils import (
    MockAgentsSDKClient,
    create_mock_agents_sdk_client,
    inject_mock_agents_sdk_client,
    ClientInjector,
    OpenAIErrorFactory
)


def demo_basic_usage():
    """Demonstrate basic mock Agents SDK client usage."""
    print("=== Basic Mock Agents SDK Client Usage ===")
    
    # Create a mock client
    mock_client = MockAgentsSDKClient(
        default_response="Hello from mock agent!",
        agent_id="demo_agent_001",
        conversation_id="demo_conv_001"
    )
    
    # Inject it into the system
    set_client(mock_client)
    
    # Process a message
    message = Message(body="Hello, agent!")
    result = process_message(message)
    
    print(f"Message: {message.body}")
    print(f"Response: {result.reply}")
    print(f"Source: {result.metadata['source']}")
    print(f"Agent ID: {result.metadata.get('agent_id', 'N/A')}")
    print(f"Conversation ID: {result.metadata.get('conversation_id', 'N/A')}")
    print(f"Processing time: {result.metadata['processing_time']:.4f}s")
    
    # Check conversation context
    context = get_conversation_context()
    print(f"Conversation turns: {context.get('turn_count', 0)}")
    print()


def demo_conversation_context():
    """Demonstrate conversation context management."""
    print("=== Conversation Context Management ===")
    
    mock_client = MockAgentsSDKClient()
    set_client(mock_client)
    
    # First message
    message1 = Message(body="What's the weather like?")
    result1 = process_message(message1)
    print(f"Message 1: {message1.body}")
    print(f"Response 1: {result1.reply}")
    
    context = get_conversation_context()
    print(f"After message 1 - Turn count: {context.get('turn_count', 0)}")
    
    # Second message
    message2 = Message(body="And tomorrow?")
    result2 = process_message(message2)
    print(f"Message 2: {message2.body}")
    print(f"Response 2: {result2.reply}")
    
    context = get_conversation_context()
    print(f"After message 2 - Turn count: {context.get('turn_count', 0)}")
    
    # Update context with custom data
    update_conversation_context({"user_preference": "celsius", "location": "New York"})
    context = get_conversation_context()
    print(f"Custom context added: {context.get('user_preference')}, {context.get('location')}")
    
    # Clear context
    clear_conversation_context()
    context = get_conversation_context()
    print(f"After clearing - Context size: {len(context)}")
    print()


def demo_tool_integration():
    """Demonstrate tool integration simulation."""
    print("=== Tool Integration Simulation ===")
    
    mock_client = MockAgentsSDKClient()
    set_client(mock_client)
    
    # Test agent suggesting echo tool
    message = Message(body="suggest echo Hello World!")
    result = process_message(message)
    
    print(f"Message: {message.body}")
    print(f"Response: {result.reply}")
    
    # Check tool tracking
    context = get_conversation_context()
    if "last_tool_used" in context:
        tool_info = context["last_tool_used"]
        print(f"Tool used: {tool_info['tool_name']}")
        print(f"Tool input: {tool_info['input']}")
        print(f"Tool output: {tool_info['output']}")
    
    print(f"Tool suggestions made: {mock_client.tool_suggestions}")
    print(f"Tool executions: {len(mock_client.tool_executions)}")
    print()


def demo_error_handling():
    """Demonstrate error handling with mock exceptions."""
    print("=== Error Handling Demonstration ===")
    
    # Create a client that raises rate limit errors
    rate_limit_error = OpenAIErrorFactory.agent_rate_limit_error(
        retry_after=30,
        agent_id="rate_limited_agent"
    )
    
    mock_client = MockAgentsSDKClient(should_raise=rate_limit_error)
    set_client(mock_client)
    
    try:
        message = Message(body="This will trigger a rate limit error")
        result = process_message(message)
        print("This shouldn't be reached")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}")
        if hasattr(e, 'error'):
            print(f"Error type: {e.error.error_type}")
            print(f"Status code: {e.error.status_code}")
            print(f"Retry after: {e.error.retry_after}s")
            print(f"Agent ID: {e.error.agent_id}")
    print()


def demo_convenience_functions():
    """Demonstrate convenience functions for testing."""
    print("=== Convenience Functions ===")
    
    # Using create_mock_agents_sdk_client
    client1 = create_mock_agents_sdk_client(
        response="Response from convenience function",
        agent_id="convenience_agent"
    )
    print(f"Created client with agent ID: {client1._agent_id}")
    
    # Using inject_mock_agents_sdk_client
    client2 = inject_mock_agents_sdk_client(
        response="Injected response",
        agent_id="injected_agent"
    )
    
    message = Message(body="Test convenience injection")
    result = process_message(message)
    print(f"Injected client response: {result.reply}")
    print(f"Agent ID from metadata: {result.metadata.get('agent_id')}")
    
    # Using context manager
    with ClientInjector(client1) as injected:
        message = Message(body="Test context manager")
        result = process_message(message)
        print(f"Context manager response: {result.reply}")
    
    print("Context manager automatically restored previous client")
    print()


def demo_call_tracking():
    """Demonstrate call tracking and verification."""
    print("=== Call Tracking and Verification ===")
    
    mock_client = MockAgentsSDKClient()
    set_client(mock_client)
    
    # Make several calls
    messages = [
        "First message",
        "Second message", 
        "Third message"
    ]
    
    for msg_text in messages:
        message = Message(body=msg_text)
        process_message(message)
    
    print(f"Total calls made: {mock_client.call_count}")
    print(f"Last prompt: {mock_client.last_prompt}")
    print(f"Call history: {mock_client.call_history}")
    
    # Reset and verify
    mock_client.reset()
    print(f"After reset - Call count: {mock_client.call_count}")
    print(f"After reset - History length: {len(mock_client.call_history)}")
    print()


if __name__ == "__main__":
    print("Mock Agents SDK Client Testing Examples")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_conversation_context()
    demo_tool_integration()
    demo_error_handling()
    demo_convenience_functions()
    demo_call_tracking()
    
    print("All demonstrations completed successfully!")