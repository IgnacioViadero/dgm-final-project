"""Entry point of the multi-agent orchestrator (phase P3 of ``docs/PLAN.md``).

The orchestrator itself is not implemented yet. ``build_orchestrator`` is already the
factory everything else goes through, and it raises ``NotImplementedError`` so that the
API starts normally and ``/health`` reports the agent phase as ``pending`` instead of
crashing on import.

When the orchestrator lands, this factory must return an object with::

    run(task: str, max_steps: int = 6, brain: str | None = None) -> api.schemas.AgentResponse

which is exactly what ``/agent`` calls.
"""

from __future__ import annotations

from typing import Any

PENDING_MESSAGE = (
    "The orchestrator (router/) is not implemented yet: no RouteDecision, no specialists "
    "dispatch and no critic. See phase P3 in docs/PLAN.md."
)


def build_orchestrator(llm: Any | None = None) -> Any:
    """Build the orchestrator the API and the evaluation share.

    Args:
        llm: The shared brain. When omitted, the backend configured by
            ``ARCA_LLM_BACKEND`` will be built by the orchestrator itself.

    Raises:
        NotImplementedError: Always, until phase P3 is done.
    """
    raise NotImplementedError(PENDING_MESSAGE)
