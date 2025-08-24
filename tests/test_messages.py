import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from gene.agent import set_client
from gene.api import app


class DummyClient:
    def complete(self, prompt: str) -> str:
        return f"echo: {prompt}"


def test_messages_endpoint_placeholder_response():
    set_client(None)
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "This is a placeholder response."


def test_messages_endpoint_uses_injected_client():
    set_client(DummyClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "echo: hello"
    set_client(None)


def test_messages_endpoint_uses_tool_when_available():
    set_client(DummyClient())
    client = TestClient(app)
    resp = client.post("/messages", json={"body": "reverse abc"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "cba"
    assert data["metadata"] == {"tool": "reverse"}
    set_client(None)
