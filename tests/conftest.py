import pytest


class FakeLLM:
    """Replaces the real model: returns the scripted replies in order."""

    model_id = "fake"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def generate(self, messages, max_new_tokens=None) -> str:
        return self.replies.pop(0)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


@pytest.fixture
def fake_llm():
    return FakeLLM