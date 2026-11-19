import time
from dataclasses import replace

from agent_core.react_agent import NUDGE, ReActAgent
from tool_use.registry import ToolRegistry, tool

RULES = "Additional rules:\n- Use the tools."


def call(name: str, **arguments) -> str:
    import json

    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def make_agent(llm, calls=None, **kwargs) -> ReActAgent:
    executed = calls if calls is not None else []

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        executed.append((a, b))
        return a + b

    @tool
    def slow() -> str:
        """Take too long.

        Args:
        """
        time.sleep(2)
        return "late"

    registry = ToolRegistry()
    registry.register(add, slow)
    return ReActAgent("test", llm, registry, RULES, **kwargs)


def test_final_answer_is_registered_automatically(fake_llm):
    agent = make_agent(fake_llm([]))
    assert "final_answer" in agent.registry.tools


def test_system_prompt_has_rules_and_tools(fake_llm):
    prompt = make_agent(fake_llm([])).system_prompt(max_steps=5)
    assert RULES in prompt
    assert "add" in prompt


def test_tool_then_final_answer(fake_llm):
    llm = fake_llm([call("add", a=2, b=3), call("final_answer", answer="5")])
    response = make_agent(llm).run("2 + 3")
    assert response.succeeded
    assert response.final_answer == "5"
    assert response.n_steps == 2
    assert response.tokens_used > 0


def test_prose_after_a_tool_is_accepted(fake_llm):
    llm = fake_llm([call("add", a=1, b=1), "El resultado es 2."])
    response = make_agent(llm).run("1 + 1")
    assert response.succeeded
    assert response.final_answer == "El resultado es 2."


def test_prose_before_any_tool_gets_one_reminder(fake_llm):
    llm = fake_llm(["Es 5, seguro.", call("add", a=2, b=3), call("final_answer", answer="5")])
    response = make_agent(llm).run("2 + 3")
    assert response.succeeded
    assert NUDGE in response.steps[0].observation


def test_malformed_json_becomes_an_observation(fake_llm):
    llm = fake_llm(["<tool_call>{not json</tool_call>", call("final_answer", answer="ok")])
    response = make_agent(llm).run("x")
    assert response.steps[0].observation.startswith("ERROR")
    assert response.succeeded


def test_unknown_tool_becomes_an_observation(fake_llm):
    llm = fake_llm([call("does_not_exist"), call("final_answer", answer="ok")])
    response = make_agent(llm).run("x")
    assert response.steps[0].observation.startswith("ERROR")
    assert response.succeeded


def test_only_the_first_of_several_calls_runs(fake_llm):
    executed: list = []
    llm = fake_llm(
        [call("add", a=1, b=2) + call("add", a=10, b=20), call("final_answer", answer="3")]
    )
    response = make_agent(llm, executed).run("x")
    assert executed == [(1, 2)]
    assert "Only your first tool call" in response.steps[0].observation


def test_duplicate_call_is_not_executed_and_third_ends_the_run(fake_llm):
    executed: list = []
    same = call("add", a=1, b=2)
    llm = fake_llm([same, same, same])
    response = make_agent(llm, executed).run("x")
    assert executed == [(1, 2)]
    assert "duplicate call" in response.steps[1].observation
    assert not response.succeeded
    assert response.n_steps == 3


def test_run_without_final_answer_fails_at_max_steps(fake_llm):
    llm = fake_llm([call("add", a=i, b=1) for i in range(4)])
    response = make_agent(llm).run("x", max_steps=3)
    assert not response.succeeded
    assert response.n_steps == 3
    assert response.final_answer == ""


def test_confirmation_denied_blocks_the_tool(fake_llm):
    executed: list = []
    llm = fake_llm([call("add", a=1, b=2), call("final_answer", answer="no")])
    agent = make_agent(llm, executed, confirm=lambda name, args: False)
    agent.registry.tools["add"] = replace(agent.registry.tools["add"], requires_confirmation=True)
    response = agent.run("x")
    assert executed == []
    assert "confirmation required" in response.steps[0].observation


def test_tool_timeout_is_an_observation(fake_llm):
    llm = fake_llm([call("slow"), call("final_answer", answer="ok")])
    agent = make_agent(llm, timeouts={"slow": 0.1})
    response = agent.run("x")
    assert "timeout" in response.steps[0].observation
    assert response.succeeded


def test_model_name_includes_agent_and_model(fake_llm):
    llm = fake_llm([call("final_answer", answer="ok")])
    assert make_agent(llm).run("x").model == "test:fake"