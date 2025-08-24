import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.tools import get_tool, handle


def test_echo_tool_discovery_and_handling():
    tool = get_tool("echo hello")
    assert tool is not None
    assert tool.handle("echo world") == "world"
    assert handle("echo hi") == "hi"


def test_reverse_tool_discovery_and_handling():
    tool = get_tool("reverse abc")
    assert tool is not None
    assert tool.handle("reverse abc") == "cba"
