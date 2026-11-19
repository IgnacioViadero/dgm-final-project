"""Helpers that clean and read what the model writes, and that guard tool execution.

They are pure functions: no model, no GPU, so they can be tested on their own.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any


def call_with_timeout(function: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``function`` in a daemon thread; raise ``TimeoutError`` if it takes too long."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"no result after {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def call_key(name: str, arguments: dict) -> str:
    """A stable key for a tool call, used to detect repeated calls."""
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


_FABRICATED_OBSERVATION = re.compile(r"^\s*(?:Observation|Observación)\s*:", re.MULTILINE)
_ACTION_LINE = re.compile(r"^[ \t]*Action:[ \t]*([A-Za-z_]\w*)?[ \t]*\n?[ \t]*(?=\{)", re.MULTILINE)
_SIMPLE_SUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)(?![\w.])")


def _repair_sums(body: str) -> str:
    """``"annual_rate_pct": 2.5 + 3`` is not JSON; the model means 5.5. Add them up."""

    def add(match: re.Match) -> str:
        a, sign, b = float(match.group(1)), match.group(2), float(match.group(3))
        return f"{round(a + b if sign == '+' else a - b, 6):g}"

    return _SIMPLE_SUM.sub(add, body)


def clean_model_output(raw: str) -> str:
    """Make a raw completion parseable without changing what the model decided to do.

    Small models imitate the ReAct transcript and write the *observation* too, then carry on
    as if they had it: everything after the first ``Observation:`` is invented, so it is cut.
    ``Action: {json}`` (and ``Action: name {args}``) is rewritten as a ``<tool_call>`` block,
    and a sum inside the JSON (``2.5 + 3``) is added up. A tool call that is still invalid
    reaches the loop as an error the model can read.
    """
    head, sep, tail = raw.rpartition("</think>")
    match = _FABRICATED_OBSERVATION.search(tail)
    if match:
        tail = tail[: match.start()]
    text = head + sep + tail
    if "<tool_call>" not in text:
        decoder = json.JSONDecoder()
        out, last = [], 0
        for found in _ACTION_LINE.finditer(text):
            if found.start() < last:
                continue
            try:
                payload, end = decoder.raw_decode(text, found.end())
            except json.JSONDecodeError:
                end = text.find("\n", found.end())
                end = len(text) if end < 0 else end
                payload = None
            body = text[found.end() : end]
            if payload is not None and found.group(1):
                body = json.dumps({"name": found.group(1), "arguments": payload})
            out += [text[last : found.start()], f"<tool_call>\n{body}\n</tool_call>"]
            last = end
        text = "".join(out) + text[last:]

    def repair(block: re.Match) -> str:
        body = block.group(1)
        try:
            json.loads(body)
        except json.JSONDecodeError:
            body = _repair_sums(body)
        return f"<tool_call>{body}</tool_call>"

    return re.sub(r"<tool_call>(.*?)</tool_call>", repair, text, flags=re.DOTALL)


def thought(parsed) -> str:
    """The reasoning text of a parsed message, without the ``Thought:`` prefix."""
    text = (parsed.thinking or parsed.text).strip()
    for prefix in ("Thought:", "Pensamiento:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def prose_answer(parsed) -> str:
    """The plain-text answer of a message without tool calls ('' if it is a cut-off think)."""
    text = parsed.text.strip()
    if "<think>" in text:  # an unterminated think block is not an answer
        return ""
    for prefix in ("Thought:", "Pensamiento:", "Final answer:", "Respuesta:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text