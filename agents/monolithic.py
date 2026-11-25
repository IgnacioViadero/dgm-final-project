"""The ARCA baseline: one agent, one big tool. The 'before' of the comparison.

This is the architecture of phase 4 of ARCA, kept unchanged on purpose so the multi-agent
system has something to be measured against (ablation A of docs/PLAN.md). It is a single
ReActAgent with ``loan_metrics`` -- seven arguments, most of them optional, every metric in
one result -- plus ``get_euribor``, whose spread the model has to add by hand.

Its known failure modes are the point of the comparison, so do NOT fix them here and do NOT
touch ``loan_metrics``:

* the model omits ``upfront_fee`` and the APR comes out wrong (10.36% instead of 12.60% in
  the documented case);
* with 'Euribor plus a spread' it has to do the sum itself, and often does it in the prompt;
* one tool for four metrics means no tool name signals which metric is wanted.

``save_credit_report`` is left out: it is an action with side effects, not a metric, and the
specialists do not have it either, so including it would make the comparison unfair.
"""

from __future__ import annotations

from agent_core.llm import Brain
from agent_core.react_agent import ReActAgent
from tool_use.registry import ToolRegistry
from tool_use.tools import get_euribor, loan_metrics

NAME = "monolithic"

RULES = """Additional rules:
- Reply in the same language as the user.
- Emit ONE tool call per message, inside <tool_call></tool_call>, and stop: the observation
  is written by the system, never by you. Arguments are plain JSON values (write 5.5, not 2.5 + 3).
- Never calculate loan figures by hand: always call loan_metrics.
- loan_metrics returns monthly_payment, total_cost (interest plus fees), apr_pct (TAE) and
  debt_ratio_pct in one result. Call it once and read the field the user asked for.
- debt_ratio_pct is null unless you pass monthly_income. If the user asks for the ratio and
  did not give their income, do not guess: call final_answer asking for it.
- If the rate is 'Euribor plus a spread', call get_euribor first and pass the Euribor plus
  the spread as annual_rate_pct.
- annual_rate_pct is a percentage: 8.5 means 8.5%, not 0.085. If the user gives no numeric
  rate, say so in final_answer instead of inventing one.
- Finish by calling final_answer with the number the user asked for, rounded to 2 decimals.
"""


def build(llm: Brain) -> ReActAgent:
    """The single-agent baseline, sharing the given model with the other agents."""
    registry = ToolRegistry().register(loan_metrics, get_euribor)
    return ReActAgent(NAME, llm, registry, RULES)
