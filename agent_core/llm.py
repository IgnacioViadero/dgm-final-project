"""The shared brain, behind one interface and two interchangeable backends.

Every agent (specialists, router, synthesis) talks to the same two methods::

    generate(messages, max_new_tokens=None) -> str
    count_tokens(text) -> int

``ARCA_LLM_BACKEND`` decides who implements them:

* ``hf`` (default): ``LLM``, Qwen3-1.7B loaded locally with transformers. Lazy load, one
  generation at a time on the shared GPU.
* ``endpoint``: ``EndpointLLM``, an OpenAI-compatible chat-completions client, for the
  central Qwen3.8-27B-FP8 vLLM service of the DGX (``ARCA_LLM_URL``, ``ARCA_LLM_MODEL``,
  ``ARCA_LLM_API_KEY``). The key is a secret: it lives in ``.env``, which git ignores, and
  is read from the environment -- never written in the code or in ``.env.example``.

Both take the tool definitions as an optional ``tools`` argument. The local backend
ignores them (the ReAct system prompt already describes the tools, and the model answers
with ``<tool_call>`` text). The endpoint MUST use them: the DGX's vLLM runs a tool-call
parser that strips ``<tool_call>`` blocks out of ``content`` and, with no ``tools`` in the
request, drops them entirely -- the model generates the call and the reply arrives empty.
Sent natively, the call comes back in ``tool_calls`` and is re-serialised to the same
``<tool_call>`` text, so the parser, the ReAct loop and the agents stay unchanged.

``ARCA_ENABLE_THINKING`` switches Qwen's reasoning traces on or off in both backends: in
``hf`` through the chat template, in ``endpoint`` through ``chat_template_kwargs``. It is
OFF by default: on the 1.7B the smoke test of 2026-11-19 scored 8/8 without thinking and
6/8 with it, at about half the tokens (see EXPERIMENTS.md). Use
``build_llm()`` instead of instantiating a backend, so swapping the brain is one env var
(that is ablation C of docs/PLAN.md).

Token accounting (the cost metric of the plan) works the same way in both: ``last_usage``
holds what the last ``generate`` really consumed, ``total_usage`` accumulates it over the
run. The endpoint reads it from the service's ``usage`` field, so the figures are exact;
the local backend counts with its own tokenizer. When neither is available the estimate of
``estimate_tokens`` is the fallback, and ``last_usage["estimated"]`` says so.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from agent_core.config import load_env
from tool_use.parser import parse_tool_calls

load_env()

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 1536
DEFAULT_BACKEND = "hf"
DEFAULT_ENDPOINT_URL = "http://0.0.0.0:8001"
DEFAULT_ENDPOINT_MODEL = "qwen3.8-27b-fp8"
DEFAULT_REQUEST_TIMEOUT_S = 300.0
# Thinking costs tokens and, on the 1.7B, accuracy. Opt in with ARCA_ENABLE_THINKING=1.
DEFAULT_ENABLE_THINKING = False
CHARS_PER_TOKEN = 4  # rough estimate when no tokenizer is available


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable written as 1/0, true/false, yes/no, on/off."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean: use 1/0, true/false, yes/no or on/off"
    )


def enable_thinking_from_env(default: bool = DEFAULT_ENABLE_THINKING) -> bool:
    """Whether the model should emit reasoning traces (``ARCA_ENABLE_THINKING``).

    Off by default: see the module docstring and the 2026-11-19 entry of EXPERIMENTS.md.
    """
    return _env_flag("ARCA_ENABLE_THINKING", default)


def estimate_tokens(text: str) -> int:
    """Backend-independent token estimate, used while no tokenizer is loaded."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def new_usage() -> dict:
    """An empty accumulator of token usage."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def usage_of(prompt_tokens: int, completion_tokens: int, estimated: bool) -> dict:
    """Usage of a single call, in the shape of the OpenAI ``usage`` field."""
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "estimated": estimated,
    }


class _TracksUsage:
    """Shared token accounting: ``last_usage`` for the last call, ``total_usage`` for all."""

    def _reset_usage(self) -> None:
        self.last_usage: dict | None = None
        self.total_usage: dict = new_usage()

    def _record_usage(self, usage: dict) -> dict:
        """Store the usage of one call and add it to the accumulator."""
        self.last_usage = usage
        self.total_usage["prompt_tokens"] += usage["prompt_tokens"]
        self.total_usage["completion_tokens"] += usage["completion_tokens"]
        self.total_usage["total_tokens"] += usage["total_tokens"]
        self.total_usage["calls"] += 1
        return usage


class LLM(_TracksUsage):
    """Qwen3 run locally with transformers. Loads lazily, so importing costs nothing."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        enable_thinking: bool | None = None,
    ):
        self.model_id = model_id or os.environ.get("ARCA_MODEL", DEFAULT_MODEL)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = (
            enable_thinking_from_env() if enable_thinking is None else enable_thinking
        )
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()  # one generation at a time on the shared GPU
        self._reset_usage()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self._model = model.to(device).eval()

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """One greedy generation. ``messages`` is a list of {"role", "content"} dicts.

        ``tools`` is accepted for interface compatibility and ignored: the ReAct system
        prompt already lists the tools and the model answers with ``<tool_call>`` text.
        """
        import torch

        with self._lock:
            self._load()
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[0, prompt_length:]
            self._record_usage(
                usage_of(int(prompt_length), int(new_tokens.shape[0]), estimated=False)
            )
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        """Exact count if the tokenizer is loaded, otherwise a rough estimate."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return estimate_tokens(text)


class EndpointLLM(_TracksUsage):
    """OpenAI-compatible chat-completions client (the DGX's central vLLM service).

    Same interface as ``LLM``, so the agents cannot tell which brain they are talking to.
    Nothing is read from the environment at call time: the configuration is resolved once,
    in the constructor.
    """

    def __init__(
        self,
        url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        enable_thinking: bool | None = None,
        timeout_s: float | None = None,
    ):
        base = url or os.environ.get("ARCA_LLM_URL", DEFAULT_ENDPOINT_URL)
        self.url = self._chat_completions_url(base)
        self.model_id = model_id or os.environ.get("ARCA_LLM_MODEL", DEFAULT_ENDPOINT_MODEL)
        # A secret: it comes from the environment (.env) and is never logged or printed.
        self.api_key = api_key or os.environ.get("ARCA_LLM_API_KEY", "EMPTY")
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = (
            enable_thinking_from_env() if enable_thinking is None else enable_thinking
        )
        self.timeout_s = (
            float(os.environ.get("ARCA_LLM_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S))
            if timeout_s is None
            else timeout_s
        )
        self._reset_usage()

    @staticmethod
    def _chat_completions_url(base: str) -> str:
        """Accept the base URL with or without ``/v1`` and with or without the endpoint."""
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """One generation through the HTTP service. Raises on a transport or HTTP error.

        When ``tools`` is given the call goes out natively (``tools`` in the request,
        ``tool_calls`` in the reply) and the history is translated to the native format,
        because the server's tool-call parser destroys the ``<tool_call>`` text protocol.
        The reply is translated back, so the caller sees the same text either way.
        """
        import requests

        payload = {
            "model": self.model_id,
            "messages": self._native_messages(messages) if tools else messages,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = requests.post(
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        text = self._text_of(body)
        self._record_usage(self._usage_of_reply(body, messages, text))
        return text

    # -- translation between the text protocol and the native tools API ------------------

    @staticmethod
    def _native_messages(messages: list[dict]) -> list[dict]:
        """Rewrite the ReAct history in the native tool-calling format.

        The loop keeps the conversation as text: the assistant's turn holds a
        ``<tool_call>`` block and the observation is a ``tool`` message with the tool's
        name. The API wants the call in ``tool_calls`` with an id, and the observation in a
        ``tool`` message carrying the matching ``tool_call_id``. Without the ids the second
        step of a chain is rejected, or the model loses track of which result is which.

        An observation whose call is no longer in the window (memory compression drops the
        oldest turns) cannot be a ``tool`` message, because the API would reject it with no
        call to attach it to: it is passed on as plain text instead.
        """
        converted: list[dict] = []
        pending_ids: list[str] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "assistant":
                parsed = parse_tool_calls(content)
                valid = [c for c in parsed.tool_calls if c.error is None]
                pending_ids = [f"call_{index}_{n}" for n in range(len(valid))]
                if not valid:
                    converted.append({"role": "assistant", "content": content})
                    continue
                converted.append(
                    {
                        "role": "assistant",
                        "content": parsed.text or None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                            for call_id, call in zip(pending_ids, valid, strict=True)
                        ],
                    }
                )
            elif role == "tool":
                if pending_ids:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": pending_ids.pop(0),
                            "content": content,
                        }
                    )
                else:
                    name = message.get("name") or "a tool"
                    converted.append(
                        {"role": "user", "content": f"Result of {name}: {content}"}
                    )
            else:
                converted.append({"role": role, "content": content})
        return converted

    @staticmethod
    def _tool_call_text(message: dict) -> str:
        """The reply's ``tool_calls``, written back as the ``<tool_call>`` blocks we parse."""
        blocks = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    arguments: Any = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    arguments = raw_arguments  # keep it: the parser reports it as an error
            else:
                arguments = raw_arguments or {}
            payload = {"name": function.get("name"), "arguments": arguments}
            blocks.append(
                f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
            )
        return "\n".join(blocks)

    def _usage_of_reply(self, body: dict, messages: list[dict], text: str) -> dict:
        """The real token usage reported by the service, or an estimate as a fallback.

        vLLM's OpenAI-compatible replies carry ``usage`` with ``prompt_tokens`` and
        ``completion_tokens``. Those are the numbers the cost metric of the report needs,
        so they are preferred over any estimate; a reply without them (or with them empty)
        falls back to ``estimate_tokens`` and is flagged as estimated.
        """
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            return usage_of(prompt_tokens, completion_tokens, estimated=False)
        prompt = "".join(str(m.get("content") or "") for m in messages)
        return usage_of(estimate_tokens(prompt), estimate_tokens(text), estimated=True)

    # Which field carries the thinking depends on the server's reasoning parser: vLLM 0.29
    # of the DGX calls it ``reasoning``, other builds ``reasoning_content``. Both are read.
    REASONING_FIELDS = ("reasoning_content", "reasoning")

    @classmethod
    def _text_of(cls, body: dict) -> str:
        """The assistant's text, keeping the reasoning trace when it is served apart.

        Three things may arrive: the text, the thinking (inline in ``content`` or, when the
        server runs a reasoning parser, in a field of its own) and the native
        ``tool_calls``. They are assembled into the one format the loop understands, so a
        caller cannot tell whether the call came as text or through the API.
        """
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected chat-completions reply: {body!r}") from exc
        content = (message.get("content") or "").strip()
        calls = cls._tool_call_text(message)
        if calls:
            content = f"{content}\n{calls}".strip() if content else calls
        traces = (
            (message.get(field) or "").strip()
            for field in cls.REASONING_FIELDS
            if message.get(field)
        )
        reasoning = next(traces, "")
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
        return content

    def count_tokens(self, text: str) -> int:
        """Estimate: the service exposes no tokenizer endpoint, and we never download one.

        It is only used to size the memory budget. The tokens charged to a run come from
        ``last_usage``, which is exact.
        """
        return estimate_tokens(text)


BACKENDS = {"hf": LLM, "endpoint": EndpointLLM}

# What every agent receives: whichever backend is configured. Annotate the shared brain with
# this instead of with ``LLM``, which is only the local one.
Brain = LLM | EndpointLLM


def build_llm(backend: str | None = None, **kwargs) -> LLM | EndpointLLM:
    """The shared brain, chosen by ``ARCA_LLM_BACKEND`` (``hf`` or ``endpoint``).

    Args:
        backend: Override for the environment variable.
        **kwargs: Passed to the backend's constructor (``max_new_tokens``,
            ``enable_thinking``, and the endpoint's ``url``/``model_id``/``api_key``).

    Raises:
        ValueError: When the backend name is unknown.
    """
    name = (backend or os.environ.get("ARCA_LLM_BACKEND", DEFAULT_BACKEND)).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"ARCA_LLM_BACKEND={name!r} is unknown: use one of {sorted(BACKENDS)}"
        )
    return BACKENDS[name](**kwargs)


if __name__ == "__main__":
    # `uv run python -m agent_core.llm` probes whichever backend ARCA_LLM_BACKEND selects:
    #   ARCA_LLM_BACKEND=hf        -> downloads and runs Qwen3-1.7B locally
    #   ARCA_LLM_BACKEND=endpoint  -> one request to the central service (inside the DGX)
    llm = build_llm()
    print(f"backend: {type(llm).__name__} · model: {llm.model_id}")
    print(f"thinking: {llm.enable_thinking}")
    print(
        llm.generate(
            [{"role": "user", "content": "¿Cuál es la capital de España?"}], max_new_tokens=64
        )
    )
    print(f"usage: {llm.last_usage}")
