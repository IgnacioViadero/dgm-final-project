"""The native tools API of the endpoint backend, with the server mocked: no network.

Why it exists: the DGX's vLLM runs a tool-call parser that strips ``<tool_call>`` blocks
out of ``content``, and with no ``tools`` in the request it drops them altogether -- the
model generates the call and the reply arrives empty (measured on 2026-11-19, 0/8 problems
and zero tool calls). Sent natively the call comes back in ``tool_calls``, and these tests
pin the translation in both directions so the parser, the ReAct loop and the agents never
learn which protocol was used.
"""

import json

import pytest

from agent_core.llm import EndpointLLM
from agents import cuota_tae
from tool_use.parser import parse_tool_calls


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def tool_call_reply(name, arguments, content=None, call_id="chatcmpl-tool-1"):
    """A reply shaped like the DGX's vLLM: content null, the call in tool_calls."""
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def text_reply(content):
    return {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }


@pytest.fixture
def server(monkeypatch):
    """A fake service that answers from a queue and records every request."""
    state = {"replies": [], "requests": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["requests"].append(json)
        return FakeResponse(state["replies"].pop(0))

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    return state


TOOLS = [
    {
        "type": "function",
        "function": {"name": "calculate_monthly_payment", "description": "x", "parameters": {}},
    }
]


# -- one call --------------------------------------------------------------------------


def test_tools_are_sent_and_the_call_comes_back_as_text(server):
    server["replies"] = [
        tool_call_reply(
            "calculate_monthly_payment",
            {"principal": 3708.23, "annual_rate_pct": 11.73, "months": 24},
        )
    ]
    llm = EndpointLLM(url="http://h:8001", api_key="k")
    raw = llm.generate([{"role": "user", "content": "cuota"}], tools=TOOLS)

    sent = server["requests"][0]
    assert sent["tools"] == TOOLS
    assert sent["tool_choice"] == "auto"

    # What the loop receives is the text protocol it already parses.
    parsed = parse_tool_calls(raw)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.error is None
    assert call.name == "calculate_monthly_payment"
    assert call.arguments == {"principal": 3708.23, "annual_rate_pct": 11.73, "months": 24}


def test_without_tools_nothing_native_is_sent(server):
    server["replies"] = [text_reply("READY")]
    EndpointLLM(url="http://h:8001").generate([{"role": "user", "content": "hi"}])
    sent = server["requests"][0]
    assert "tools" not in sent and "tool_choice" not in sent
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_a_reply_without_tool_calls_is_plain_text(server):
    server["replies"] = [text_reply("La cuota es de 174.09 euros.")]
    raw = EndpointLLM(url="http://h:8001").generate([], tools=TOOLS)
    assert raw == "La cuota es de 174.09 euros."
    assert parse_tool_calls(raw).tool_calls == []


def test_content_and_a_call_in_the_same_reply_both_survive(server):
    server["replies"] = [tool_call_reply("final_answer", {"answer": "7.48"}, content="Ya lo tengo")]
    raw = EndpointLLM(url="http://h:8001").generate([], tools=TOOLS)
    assert "Ya lo tengo" in raw
    assert parse_tool_calls(raw).tool_calls[0].name == "final_answer"


def test_reasoning_and_a_call_in_the_same_reply(server):
    reply = tool_call_reply("final_answer", {"answer": "1"})
    reply["choices"][0]["message"]["reasoning"] = "lo pienso"
    server["replies"] = [reply]
    raw = EndpointLLM(url="http://h:8001").generate([], tools=TOOLS)
    parsed = parse_tool_calls(raw)
    assert parsed.thinking == "lo pienso"
    assert parsed.tool_calls[0].name == "final_answer"


def test_broken_arguments_are_handed_to_the_parser_not_swallowed(server):
    reply = tool_call_reply("calculate_apr", {})
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    server["replies"] = [reply]
    raw = EndpointLLM(url="http://h:8001").generate([], tools=TOOLS)
    assert "calculate_apr" in raw  # the loop decides what to do with it


# -- the history, translated to the native format --------------------------------------


def test_the_history_carries_tool_call_ids(server):
    """The second step must tell the server which call each observation belongs to."""
    history = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Euríbor + 1,5 % a 60 meses, 15000 euros"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "get_variable_rate", '
            '"arguments": {"tenor": "12M", "spread_pct": 1.5}}\n</tool_call>',
        },
        {"role": "tool", "content": '{"result": 3.913}', "name": "get_variable_rate"},
    ]
    server["replies"] = [
        tool_call_reply(
            "calculate_monthly_payment",
            {"principal": 15000, "annual_rate_pct": 3.913, "months": 60},
        )
    ]
    EndpointLLM(url="http://h:8001").generate(history, tools=TOOLS)

    sent = server["requests"][0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assistant, observation = sent[2], sent[3]
    assert assistant["content"] is None  # only the call, no prose
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["function"]["name"] == "get_variable_rate"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "tenor": "12M",
        "spread_pct": 1.5,
    }
    # The observation points at that call's id.
    assert observation["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert observation["content"] == '{"result": 3.913}'


def test_prose_around_a_call_is_kept_as_content(server):
    history = [
        {
            "role": "assistant",
            "content": 'Voy a calcularlo.\n<tool_call>{"name": "f", "arguments": {}}</tool_call>',
        }
    ]
    server["replies"] = [text_reply("ok")]
    EndpointLLM(url="http://h:8001").generate(history, tools=TOOLS)
    assert server["requests"][0]["messages"][0]["content"] == "Voy a calcularlo."


def test_an_assistant_turn_without_a_call_stays_plain(server):
    server["replies"] = [text_reply("ok")]
    EndpointLLM(url="http://h:8001").generate(
        [{"role": "assistant", "content": "No puedo responder."}], tools=TOOLS
    )
    sent = server["requests"][0]["messages"][0]
    assert sent == {"role": "assistant", "content": "No puedo responder."}


def test_an_orphan_observation_is_not_sent_as_a_tool_message(server):
    """Memory compression can drop the call: a tool message with no id would be rejected."""
    server["replies"] = [text_reply("ok")]
    EndpointLLM(url="http://h:8001").generate(
        [{"role": "tool", "content": '{"result": 1}', "name": "calculate_apr"}], tools=TOOLS
    )
    sent = server["requests"][0]["messages"][0]
    assert sent["role"] == "user"
    assert "calculate_apr" in sent["content"] and '{"result": 1}' in sent["content"]


def test_a_broken_tool_call_block_in_the_history_is_sent_as_text(server):
    server["replies"] = [text_reply("ok")]
    EndpointLLM(url="http://h:8001").generate(
        [{"role": "assistant", "content": "<tool_call>{not json}</tool_call>"}], tools=TOOLS
    )
    sent = server["requests"][0]["messages"][0]
    assert "tool_calls" not in sent
    assert "not json" in sent["content"]


# -- two chained steps, through the real agent -----------------------------------------


def test_two_chained_steps_through_the_agent(server):
    """get_variable_rate -> calculate_monthly_payment, with the ECB mocked too."""
    server["replies"] = [
        tool_call_reply("get_variable_rate", {"tenor": "12M", "spread_pct": 1.5}, call_id="c1"),
        tool_call_reply(
            "calculate_monthly_payment",
            {"principal": 15000, "annual_rate_pct": 3.913, "months": 60},
            call_id="c2",
        ),
        tool_call_reply("final_answer", {"answer": "275.66"}, call_id="c3"),
    ]

    import requests

    ecb_body = {
        "dataSets": [{"series": {"0": {"observations": {"0": [2.413]}}}}],
        "structure": {"dimensions": {"observation": [{"values": [{"id": "2026-10"}]}]}},
    }
    original_get = requests.get
    requests.get = lambda *a, **k: FakeResponse(ecb_body)
    try:
        llm = EndpointLLM(url="http://h:8001", api_key="k")
        response = cuota_tae.build(llm).run(
            "Euríbor + 1,5 %: cuota de 15000 euros a 60 meses", max_steps=5
        )
    finally:
        requests.get = original_get

    assert response.succeeded
    assert response.final_answer == "275.66"
    assert [s.action.name for s in response.steps] == [
        "get_variable_rate",
        "calculate_monthly_payment",
        "final_answer",
    ]
    assert json.loads(response.steps[0].observation)["result"] == 3.913

    # The second request carried the first call and its observation, with matching ids.
    second = server["requests"][1]["messages"]
    assistant = next(m for m in second if m.get("tool_calls"))
    observation = next(m for m in second if m["role"] == "tool")
    assert assistant["tool_calls"][0]["function"]["name"] == "get_variable_rate"
    assert observation["tool_call_id"] == assistant["tool_calls"][0]["id"]
    # And every request declared the agent's tools.
    for request in server["requests"]:
        assert {t["function"]["name"] for t in request["tools"]} == {
            "calculate_monthly_payment",
            "calculate_apr",
            "get_variable_rate",
            "get_euribor",
            "final_answer",
        }
    assert response.tokens_used == 360  # 3 generations × 120 real tokens
