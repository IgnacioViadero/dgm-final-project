"""Specialist for the instalment, the APR (TAE) and variable rates.

Router -> this agent -> one of four deterministic tools. The model never does the maths and
never does arithmetic in the prompt: for 'Euribor plus a spread' it calls get_variable_rate,
which returns the rate already added up, and feeds that result to the next tool. That chain
is what replaces the ``_repair_sums`` patch of ARCA.
"""

from __future__ import annotations

from agent_core.llm import Brain
from agent_core.react_agent import ReActAgent
from tool_use.registry import ToolRegistry
from tool_use.tools_cuota_tae import (
    calculate_apr,
    calculate_monthly_payment,
    get_euribor,
    get_variable_rate,
)

NAME = "cuota_tae"

RULES = """Additional rules:
- Reply in the same language as the user.
- Emit ONE tool call per message, inside <tool_call></tool_call>, and stop: the observation
  is written by the system, never by you. Arguments are plain JSON values (write 5.5, not 2.5 + 3).
- Never calculate loan figures by hand and never add two numbers yourself: always call a tool.
- Choose the tool by what the user asks for:
  * the monthly instalment ('cuota'), or how much they would pay each month
    -> calculate_monthly_payment.
  * the APR ('TAE'), or the real cost of the loan as a rate
    -> calculate_apr.
  * a rate given as 'Euribor plus X' ('Euribor + X', 'Euribor mas X')
    -> get_variable_rate first, then the tool for what the user asked.
  * the Euribor itself, with no loan to compute
    -> get_euribor.
- Every tool answers with {"result": the number you asked for, "inputs": the figures it
  used}. Read "result"; "inputs" is only there so the answer can be checked.
- upfront_fee is REQUIRED for the APR. Pass the fee the user gave; use 0 only if the user
  mentions no fee. Never omit it and never pass 0 when a fee was mentioned: the APR would be
  wrong.
- Do not pass the instalment to calculate_apr: it computes it itself from the principal, the
  rate and the term.
- For 'Euribor plus a spread': call get_variable_rate with spread_pct = only the spread (for
  'Euribor plus 1.5%' that is 1.5), take its "result" as annual_rate_pct for the next call,
  and never add the Euribor and the spread yourself. Use tenor '12M' unless the user asks for
  another maturity.
- annual_rate_pct is a percentage: 8.5 means 8.5%, not 0.085. If the user gives no numeric
  rate and the rate does not depend on the Euribor, say so in final_answer instead of
  inventing one.
- Finish by calling final_answer with the number the user asked for, rounded to 2 decimals,
  written as text.
- Copy the "result" EXACTLY, keeping its decimal point: 10.34 stays 10.34, never 1034 or
  1,034. The dot is the decimal separator, not a thousands separator. Do not add currency
  or percent symbols.
"""


def build(llm: Brain) -> ReActAgent:
    """The instalment-and-APR specialist, sharing the given model with the other agents."""
    registry = ToolRegistry().register(
        calculate_monthly_payment, calculate_apr, get_variable_rate, get_euribor
    )
    return ReActAgent(NAME, llm, registry, RULES)
