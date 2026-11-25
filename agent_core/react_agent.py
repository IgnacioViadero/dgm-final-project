"""The ReAct agent: Thought -> Action -> Observation, in a loop.

One class serves every specialist. What changes between agents is the tool registry and the
rules they receive; the model (``LLM``) is shared and the loop is the same.

* One action per step: if the model emits several <tool_call> blocks, only the first runs.
* Everything that goes wrong is an observation, never an exception.
* Per-tool timeout; a stuck tool cannot block the agent.
* Loop guard: the second identical call is not executed, the third ends the run as failed.
* Prose after using a tool is accepted as the final answer; before any tool, one reminder.
* Tools with ``requires_confirmation`` go through ``self.confirm``.
* The tool definitions are handed to the brain on every call. The local backend ignores
  them (it reads them from the system prompt); the endpoint needs them to call natively.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_core.llm import LLM
from agent_core.memory import ConversationMemory, clip, extractive_summary
from agent_core.output import (
    call_key,
    call_with_timeout,
    clean_model_output,
    prose_answer,
    thought,
)
from agent_core.prompts import REACT_SYSTEM_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.executor import PARSE_ERROR_HINT, confirm_from_env
from tool_use.parser import ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry, tool

log = logging.getLogger("arca.agent")

NUDGE = (
    "Reminder: you have not used any tool yet. Reply with exactly one <tool_call> block. "
    "Use the tools to get the data or do the calculation, and finish by calling "
    "final_answer. If the task cannot be done with the tools, call final_answer and say so."
)


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer


@dataclass
class ReActAgent:
    name: str
    llm: LLM
    registry: ToolRegistry
    rules: str  # specialist-specific rules appended to the system prompt
    token_budget: int = 6000
    tool_timeout_s: float = 20.0
    timeouts: dict[str, float] = field(default_factory=dict)  # per-tool overrides
    max_observation_chars: int = 2000
    confirm: Callable[[str, dict], bool] = field(default_factory=confirm_from_env)
    steps: list[AgentStep] = field(default_factory=list)  # steps of the last run

    def __post_init__(self) -> None:
        if "final_answer" not in self.registry.tools:
            self.registry.register(final_answer)

    def system_prompt(self, max_steps: int) -> str:
        return (
            REACT_SYSTEM_PROMPT.format(
                max_steps=max_steps, tools_description=self.registry.render_for_prompt()
            )
            + "\n"
            + self.rules
        )

    def _execute(self, call: ToolCall, seen: dict[str, int]) -> ToolCallRecord:
        """Run one tool call with every guard. Never raises."""
        if call.error is not None:
            return ToolCallRecord(
                name=call.name or "<invalid>",
                arguments={},
                ok=False,
                error=call.error + PARSE_ERROR_HINT,
            )
        key = call_key(call.name, call.arguments)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=(
                    f"duplicate call: you already called {call.name} with these arguments "
                    "and got its result. Use that result, or call final_answer."
                ),
            )
        spec = self.registry.tools.get(call.name)
        if spec is not None and spec.requires_confirmation:
            approved = self.confirm(call.name, call.arguments)
            log.info("confirmation for %s %s: %s", call.name, call.arguments, approved)
            if not approved:
                return ToolCallRecord(
                    name=call.name,
                    arguments=call.arguments,
                    ok=False,
                    error=(
                        "confirmation required: the user has not approved this action, so it "
                        "was NOT executed. Tell the user what you would do and ask them to "
                        "confirm."
                    ),
                )
        timeout = self.timeouts.get(call.name, self.tool_timeout_s)
        try:
            return call_with_timeout(lambda: self.registry.call(call.name, call.arguments), timeout)
        except TimeoutError as exc:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=f"timeout: {call.name} gave {exc}. Try other arguments or another tool.",
            )

    def _tokens_of_last_call(self, messages: list[dict], raw: str) -> int:
        """Tokens charged for one generation: the real figure when the brain reports it.

        ``EndpointLLM`` fills ``last_usage`` from the service's ``usage`` field and the
        local backend from its own tokenizer, so this is exact for both. Brains that do not
        track usage (the ``FakeLLM`` of the tests, or a future one) fall back to counting
        the prompt and the reply with ``count_tokens``.
        """
        usage = getattr(self.llm, "last_usage", None)
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            return usage["total_tokens"]
        prompt = "".join(str(m.get("content") or "") for m in messages)
        return self.llm.count_tokens(prompt) + self.llm.count_tokens(raw)

    def run(self, task: str, max_steps: int = 6) -> AgentResponse:
        """The ReAct loop: think, act, observe, until ``final_answer`` or no steps left."""
        memory = ConversationMemory(
            self.system_prompt(max_steps),
            self.llm.count_tokens,
            token_budget=self.token_budget,
            pin_first=True,
            max_observation_chars=300,
        )
        memory.add("user", task)
        steps: list[AgentStep] = []
        seen: dict[str, int] = {}
        tokens = 0
        used_a_tool = False
        reminded = False

        def finish(answer: str, ok: bool, n_steps: int) -> AgentResponse:
            self.steps = steps
            return AgentResponse(
                final_answer=answer,
                steps=steps,
                n_steps=n_steps,
                succeeded=ok,
                model=f"{self.name}:{self.llm.model_id}",
                tokens_used=tokens,
            )

        for step_number in range(1, max_steps + 1):
            messages = memory.render()
            raw = self.llm.generate(messages, tools=self.registry.definitions())
            tokens += self._tokens_of_last_call(messages, raw)
            raw = clean_model_output(raw)
            parsed = parse_tool_calls(raw)
            thinking = thought(parsed)
            valid = [c for c in parsed.tool_calls if c.error is None]
            first_is_broken = bool(parsed.tool_calls) and parsed.tool_calls[0].error is not None

            if not parsed.tool_calls:  # the model wrote prose instead of an action
                prose = prose_answer(parsed)
                if prose and (used_a_tool or reminded):
                    steps.append(
                        AgentStep(
                            thought="(no final_answer call; the text is taken as the answer) "
                            + thinking,
                            action=AgentAction(name="final_answer", arguments={"answer": prose}),
                            observation=prose,
                        )
                    )
                    return finish(prose, True, step_number)
                steps.append(AgentStep(thought=thinking, observation=f"[reminder] {NUDGE}"))
                memory.add("assistant", raw)
                memory.add("user", NUDGE)
                reminded = True
                continue

            if first_is_broken or not valid:  # tell the model what is wrong, run nothing later
                record = self._execute(parsed.tool_calls[0], seen)
                observation = f"ERROR: {record.error}"
                steps.append(AgentStep(thought=thinking, observation=clip(observation, 4000)))
                memory.add("assistant", raw)
                memory.add("tool", observation, name="error")
                memory.compress(extractive_summary)
                continue

            call = valid[0]
            extra = (
                " Only your first tool call was executed; wait for results before the next."
                if len(valid) > 1
                else ""
            )
            action = AgentAction(name=call.name, arguments=call.arguments)

            if call.name == "final_answer":
                record = self.registry.call("final_answer", call.arguments)
                if record.ok:
                    answer = str(record.result)
                    steps.append(AgentStep(thought=thinking, action=action, observation=answer))
                    return finish(answer, True, step_number)
            else:
                record = self._execute(call, seen)

            if record.ok:
                used_a_tool = True
                payload = format_tool_result(record.name, record.result)
            else:
                payload = f"ERROR: {record.error}"
            payload = clip(payload, self.max_observation_chars) + extra
            steps.append(AgentStep(thought=thinking, action=action, observation=payload))
            memory.add("assistant", raw)
            memory.add("tool", payload, name=call.name)
            memory.compress(extractive_summary)

            if seen.get(call_key(call.name, call.arguments), 0) >= 3:
                log.warning("loop detected: %s called 3 times with the same arguments", call.name)
                return finish("", False, step_number)

        return finish("", False, max_steps)