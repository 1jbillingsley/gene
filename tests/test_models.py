import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gene.models import Message


def test_message_requires_body():
    with pytest.raises(ValidationError):
        Message()


def test_message_accepts_metadata_dict():
    msg = Message(body="hello", metadata={"source": "unit"})
    assert msg.metadata == {"source": "unit"}


def test_message_rejects_non_dict_metadata():
    with pytest.raises(ValidationError):
        Message(body="hi", metadata="invalid")
