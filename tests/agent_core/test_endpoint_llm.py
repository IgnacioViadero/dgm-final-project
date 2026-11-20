"""The endpoint backend, with ``requests`` mocked: no network, no GPU."""

import pytest

from agent_core.llm import (
    DEFAULT_MODEL,
    LLM,
    EndpointLLM,
    build_llm,
    enable_thinking_from_env,
)


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def captured_post(monkeypatch):
    """Patch ``requests.post`` and record the call the backend makes."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(
            {
                "choices": [{"message": {"role": "assistant", "content": "READY"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
            }
        )

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


# -- configuration -------------------------------------------------------------------


def test_backend_is_hf_by_default(monkeypatch):
    monkeypatch.delenv("ARCA_LLM_BACKEND", raising=False)
    llm = build_llm()
    assert isinstance(llm, LLM)
    assert llm.model_id == DEFAULT_MODEL


def test_backend_endpoint_comes_from_environment(monkeypatch):
    monkeypatch.setenv("ARCA_LLM_BACKEND", "endpoint")
    monkeypatch.setenv("ARCA_LLM_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("ARCA_LLM_MODEL", "qwen3.8-27b-fp8")
    monkeypatch.setenv("ARCA_LLM_API_KEY", "EMPTY")
    llm = build_llm()
    assert isinstance(llm, EndpointLLM)
    assert llm.url == "http://127.0.0.1:9000/v1/chat/completions"
    assert llm.model_id == "qwen3.8-27b-fp8"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("ARCA_LLM_BACKEND", "ollama")
    with pytest.raises(ValueError, match="ARCA_LLM_BACKEND"):
        build_llm()


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://h:8000", "http://h:8000/v1/chat/completions"),
        ("http://h:8000/", "http://h:8000/v1/chat/completions"),
        ("http://h:8000/v1", "http://h:8000/v1/chat/completions"),
        ("http://h:8000/v1/chat/completions", "http://h:8000/v1/chat/completions"),
    ],
)
def test_url_is_normalised(base, expected):
    assert EndpointLLM(url=base).url == expected


def test_nothing_is_requested_on_construction(captured_post):
    EndpointLLM(url="http://h:8000")
    assert captured_post == []


# -- thinking ------------------------------------------------------------------------


def test_thinking_is_off_by_default(monkeypatch):
    """The 1.7B scored 8/8 without thinking and 6/8 with it: see EXPERIMENTS.md."""
    monkeypatch.delenv("ARCA_ENABLE_THINKING", raising=False)
    assert enable_thinking_from_env() is False
    assert LLM().enable_thinking is False
    assert EndpointLLM(url="http://h:8000").enable_thinking is False


def test_thinking_reads_the_environment(monkeypatch):
    monkeypatch.setenv("ARCA_ENABLE_THINKING", "1")
    assert enable_thinking_from_env() is True
    assert LLM().enable_thinking is True
    assert EndpointLLM(url="http://h:8000").enable_thinking is True
    monkeypatch.setenv("ARCA_ENABLE_THINKING", "false")
    assert LLM().enable_thinking is False


def test_invalid_thinking_flag_is_rejected(monkeypatch):
    monkeypatch.setenv("ARCA_ENABLE_THINKING", "maybe")
    with pytest.raises(ValueError, match="ARCA_ENABLE_THINKING"):
        enable_thinking_from_env()


def test_thinking_travels_as_chat_template_kwargs(captured_post):
    llm = EndpointLLM(url="http://h:8000", enable_thinking=False)
    llm.generate([{"role": "user", "content": "hi"}])
    assert captured_post[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}


# -- generate ------------------------------------------------------------------------


def test_generate_sends_an_openai_payload_and_returns_the_content(captured_post):
    llm = EndpointLLM(
        url="http://h:8000/v1", model_id="qwen3.8-27b-fp8", api_key="EMPTY", enable_thinking=True
    )
    messages = [{"role": "user", "content": "Reply with the word READY."}]
    assert llm.generate(messages, max_new_tokens=64) == "READY"

    call = captured_post[0]
    assert call["url"] == "http://h:8000/v1/chat/completions"
    assert call["json"] == {
        "model": "qwen3.8-27b-fp8",
        "messages": messages,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert call["headers"]["Authorization"] == "Bearer EMPTY"
    assert call["timeout"] == llm.timeout_s


def test_generate_falls_back_to_the_configured_token_budget(captured_post):
    EndpointLLM(url="http://h:8000", max_new_tokens=123).generate([])
    assert captured_post[0]["json"]["max_tokens"] == 123


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_a_separate_reasoning_trace_is_rewrapped(field):
    """The DGX's vLLM 0.29 names the field "reasoning"; other builds "reasoning_content"."""
    body = {"choices": [{"message": {"content": "<tool_call>{}</tool_call>", field: "lo pienso"}}]}
    expected = "<think>\nlo pienso\n</think>\n\n<tool_call>{}</tool_call>"
    assert EndpointLLM._text_of(body) == expected


def test_a_reasoning_only_reply_is_not_lost():
    """A reply whose content is empty must still hand the loop what the model produced."""
    body = {"choices": [{"message": {"content": None, "reasoning": "lo pienso"}}]}
    assert EndpointLLM._text_of(body) == "<think>\nlo pienso\n</think>"


def test_an_empty_reply_is_empty_text():
    body = {"choices": [{"message": {"content": None, "reasoning": None}}]}
    assert EndpointLLM._text_of(body) == ""


def test_http_error_is_raised(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"error": "boom"}, status=500)
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        EndpointLLM(url="http://h:8000").generate([])


def test_malformed_reply_is_reported(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"choices": []}))
    with pytest.raises(ValueError, match="unexpected chat-completions reply"):
        EndpointLLM(url="http://h:8000").generate([])


def test_count_tokens_is_estimated_without_network(captured_post):
    """count_tokens only sizes the memory budget; the charged tokens come from usage."""
    assert EndpointLLM(url="http://h:8000").count_tokens("a" * 40) == 10
    assert captured_post == []


# -- token usage ---------------------------------------------------------------------


def post_returning(monkeypatch, body):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body))


def reply(content="READY", usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def test_no_usage_before_the_first_call():
    llm = EndpointLLM(url="http://h:8000")
    assert llm.last_usage is None
    assert llm.total_usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def test_usage_field_is_read_instead_of_being_estimated(captured_post):
    llm = EndpointLLM(url="http://h:8000")
    llm.generate([{"role": "user", "content": "x" * 4000}])
    assert llm.last_usage == {
        "prompt_tokens": 120,
        "completion_tokens": 7,
        "total_tokens": 127,
        "estimated": False,
    }


def test_usage_accumulates_over_calls(captured_post):
    llm = EndpointLLM(url="http://h:8000")
    for _ in range(3):
        llm.generate([{"role": "user", "content": "hi"}])
    assert llm.total_usage == {
        "prompt_tokens": 360,
        "completion_tokens": 21,
        "total_tokens": 381,
        "calls": 3,
    }
    assert llm.last_usage["total_tokens"] == 127  # unchanged by the accumulator


def test_total_tokens_is_recomputed_not_trusted(monkeypatch):
    """A service that reports an inconsistent total must not corrupt the accounting."""
    post_returning(
        monkeypatch,
        reply(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 999}),
    )
    llm = EndpointLLM(url="http://h:8000")
    llm.generate([])
    assert llm.last_usage["total_tokens"] == 15


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 10},
        {"prompt_tokens": None, "completion_tokens": None},
        {"prompt_tokens": "10", "completion_tokens": "5"},
    ],
)
def test_estimate_is_the_fallback_when_usage_is_missing(monkeypatch, usage):
    post_returning(monkeypatch, reply(content="a" * 40, usage=usage))
    llm = EndpointLLM(url="http://h:8000")
    llm.generate([{"role": "user", "content": "b" * 80}])
    assert llm.last_usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "estimated": True,
    }
    assert llm.total_usage["calls"] == 1
