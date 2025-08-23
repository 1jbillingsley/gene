# Vision & Problem Statement

## Vision

Build a platform-agnostic REST API that accepts messages from diverse systems, leverages OpenAI's Agent framework to interpret intent and decide actions, and returns actionable outputs. This service acts as a central "brain" for heterogeneous applications, applying AI to reason about failures, notifications, or other events and deliver tailored responses.

### Key Principles
- **Platform independence**: Request payloads support a generic message body plus optional metadata describing source and context.
- **Extensible processing**: The OpenAI Agent framework orchestrates tools and models so new message types and actions can be added with minimal code.
- **Best practices**: Modular architecture, testability, observability, security, and thorough documentation guide implementation.

## Problem Statement

Organizations rely on many services that produce notifications and logs, yet these messages are siloed and require manual interpretation. Developers, for example, often sift through CI pipeline logs to diagnose failures and notify stakeholders. This manual process is slow and error-prone.

We need a single endpoint capable of receiving arbitrary messages (e.g., failed GitLab CI pipelines) and using AI to analyze, decide, and summarize appropriate follow-up actions. The system must accommodate new message sources without redesign, letting teams integrate once and rely on AI-driven decision-making for many scenarios.

### Success Metrics
- Able to ingest messages from multiple sources with minimal customization.
- Generates accurate, actionable AI summaries and decisions.
- New message types are integrated quickly through configuration and tool plug-ins.

## Configuration

The application reads settings from environment variables or a `.env` file in
the project root. Available options include:

- `LOG_LEVEL` – controls verbosity of log output. Defaults to `INFO`.

Example `.env` file:

```
LOG_LEVEL=DEBUG
```

Settings are loaded at startup by `src/gene/config.py`.
