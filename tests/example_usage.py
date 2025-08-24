"""Example usage of the dependency injection test utilities.

This file demonstrates how to use the new test utilities for easy
client injection in test scenarios.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.agent import process_message
from gene.models import Message

from test_utils import (
    MockOpenAIClient,
    ClientInjector,
    inject_mock_client,
    OpenAIErrorFactory,
    ErrorMockOpenAIClient,
    reset_client,
)


def example_basic_mock_usage():
    """Example: Basic mock client usage."""
    print("=== Basic Mock Client Usage ===")

    # Create and inject a simple mock client
    mock_client = inject_mock_client("This is a mock AI response!")

    # Process a message
    message = Message(body="Analyze this error log")
    result = process_message(message)

    print(f"Response: {result.reply}")
    print(f"Metadata: {result.metadata}")
    print(f"Mock was called {mock_client.call_count} times")

    reset_client()


def example_specific_responses():
    """Example: Mock client with specific responses."""
    print("\n=== Specific Response Mapping ===")

    # Configure specific responses for different prompts
    responses = {
        "analyze error": "Error analysis: Database connection timeout detected",
        "summarize logs": "Log summary: 15 warnings, 3 errors in the last hour",
        "check status": "System status: All services operational",
    }

    mock_client = inject_mock_client(responses=responses)

    # Test different prompts
    test_prompts = ["analyze error", "summarize logs", "unknown prompt"]

    for prompt in test_prompts:
        message = Message(body=prompt)
        result = process_message(message)
        print(f"Prompt: '{prompt}' -> Response: '{result.reply}'")

    reset_client()


def example_context_manager():
    """Example: Using ClientInjector context manager."""
    print("\n=== Context Manager Usage ===")

    # Verify no client initially
    result_before = process_message(Message(body="test"))
    print(f"Before injection: {result_before.metadata['source']}")

    # Use context manager for clean injection/restoration
    mock_client = MockOpenAIClient("Context manager response")

    with ClientInjector(mock_client):
        result_during = process_message(Message(body="test"))
        print(f"During injection: {result_during.reply}")
        print(f"Source: {result_during.metadata['source']}")

    # Verify restoration
    result_after = process_message(Message(body="test"))
    print(f"After context: {result_after.metadata['source']}")


def example_error_testing():
    """Example: Testing error scenarios."""
    print("\n=== Error Scenario Testing ===")

    # Test rate limit error
    rate_limit_error = OpenAIErrorFactory.rate_limit_error(retry_after=120)
    error_client = ErrorMockOpenAIClient(rate_limit_error)

    with ClientInjector(error_client):
        try:
            process_message(Message(body="This will fail"))
        except Exception as e:
            print(f"Caught expected error: {type(e).__name__}")
            print(f"Error details: {e.error.message}")
            print(f"Status code: {e.error.status_code}")
            print(f"Retry after: {e.error.retry_after} seconds")


def example_call_verification():
    """Example: Verifying mock client calls."""
    print("\n=== Call Verification ===")

    mock_client = MockOpenAIClient("Verification response")

    with ClientInjector(mock_client):
        # Process multiple messages
        messages = [
            "Analyze system performance",
            "Check error rates",
            "Generate report",
        ]

        for msg_text in messages:
            process_message(Message(body=msg_text))

        # Verify calls
        print(f"Total calls made: {mock_client.call_count}")
        print(f"Call history: {mock_client.call_history}")
        print(f"Last prompt: {mock_client.last_prompt}")


if __name__ == "__main__":
    print("Dependency Injection Test Utilities - Example Usage\n")

    example_basic_mock_usage()
    example_specific_responses()
    example_context_manager()
    example_error_testing()
    example_call_verification()

    print("\n=== All examples completed successfully! ===")
