import os

import pytest

# The tests must not depend on the developer's .env (it holds the real endpoint and its
# API key). This is read by agent_core.config.load_env, and conftest is imported before
# any test module, so it takes effect before agent_core.llm is loaded.
os.environ.setdefault("ARCA_SKIP_DOTENV", "1")


class FakeLLM:
    """Replaces the real model: returns the scripted replies in order."""

    model_id = "fake"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.last_tools = None

    def generate(self, messages, max_new_tokens=None, tools=None) -> str:
        self.last_tools = tools  # the loop must hand the definitions to the brain
        return self.replies.pop(0)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


@pytest.fixture
def fake_llm():
    return FakeLLM