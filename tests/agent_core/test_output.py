import pytest

from agent_core.output import call_key, call_with_timeout, clean_model_output


def test_sum_inside_json_is_added():
    raw = '<tool_call>{"name": "x", "arguments": {"rate": 2.5 + 3}}</tool_call>'
    assert '"rate": 5.5' in clean_model_output(raw)


def test_fabricated_observation_is_cut():
    raw = '</think><tool_call>{"name": "x", "arguments": {}}</tool_call>\nObservation: 99'
    assert "99" not in clean_model_output(raw)


def test_observation_inside_thinking_is_kept():
    raw = "<think>\nObservation: pensando</think><tool_call>{}</tool_call>"
    assert "pensando" in clean_model_output(raw)


def test_action_line_becomes_tool_call():
    assert "<tool_call>" in clean_model_output('Action: {"name": "x", "arguments": {}}')


def test_valid_tool_call_is_unchanged():
    raw = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
    assert clean_model_output(raw) == raw


def test_call_key_ignores_argument_order():
    assert call_key("f", {"a": 1, "b": 2}) == call_key("f", {"b": 2, "a": 1})


def test_call_with_timeout_returns_value():
    assert call_with_timeout(lambda: 7, 1) == 7


def test_call_with_timeout_raises_on_slow_function():
    import time

    with pytest.raises(TimeoutError):
        call_with_timeout(lambda: time.sleep(1), 0.05)


def test_call_with_timeout_propagates_errors():
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        call_with_timeout(boom, 1)