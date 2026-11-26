from __future__ import annotations

import logging

from agent_core.llm import LLM, build_llm
from agents import coste_ratio, cuota_tae, rag
from api.schemas import AgentResponse, AgentStep
from router.router import route

log = logging.getLogger("dgm-project.orchestrator")


class Orchestrator:
    """Routes a task to one specialist and returns its AgentResponse."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.agents = {
            cuota_tae.NAME: cuota_tae.build(llm),
            coste_ratio.NAME: coste_ratio.build(llm),
            rag.NAME: rag.build(llm)
        }
        self.last_decision = None

    def run(self, task: str, max_steps: int = 6, brain: str | None = None) -> AgentResponse:
        try:
            decision = route(self.llm, task)
            self.last_decision = decision
        except Exception as exc:  # the router must never take the API down
            log.exception("router failed")
            return AgentResponse(
                final_answer="", steps=[], n_steps=0, succeeded=False,
                model=f"orchestrator:{self.llm.model_id}",
                tokens_used=0,
            )
        log.info("routed to %s (%s)", decision.agent, decision.reason)

        response = self.agents[decision.agent].run(task, max_steps=max_steps)

        route_step = AgentStep(
            thought=f"Router -> {decision.agent}. {decision.reason}".strip(),
        )
        response.steps = [route_step, *response.steps]
        response.n_steps = len(response.steps)
        response.model = f"orchestrator>{response.model}"
        return response


def build_orchestrator(llm=None) -> Orchestrator:
    if llm is None:
        llm = build_llm()  # your real factory, driven by ARCA_LLM_BACKEND
    return Orchestrator(llm)