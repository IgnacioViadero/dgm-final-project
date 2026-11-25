"""Specialist for the total cost of credit and the debt ratio.

Router -> this agent -> one of two deterministic tools. The model never does the maths: it
picks the tool, fills in the arguments from the user's question and reads the result.
"""

from __future__ import annotations

from agent_core.llm import Brain
from agent_core.react_agent import ReActAgent
from tool_use.registry import ToolRegistry
from tool_use.tools_coste_ratio import calculate_debt_ratio, calculate_total_cost

NAME = "coste_ratio"

RULES = """Additional rules:
- Reply in the same language as the user.
- Emit ONE tool call per message, inside <tool_call></tool_call>, and stop: the observation
  is written by the system, never by you. Arguments are plain JSON values (write 5.5, not 2.5 + 3).
- Never calculate loan figures by hand: always call a tool.
- Choose the tool by what the user asks for:
  * total cost of the credit, total interest, or how much is paid in total
    -> calculate_total_cost.
  * debt ratio, or what percentage of the income goes to the instalment
    -> calculate_debt_ratio.
- Every tool answers with {"result": the number you asked for, "inputs": the figures it
  used}. Read "result"; "inputs" is only there so the answer can be checked.
- calculate_total_cost returns the interest plus the fees (everything paid minus the capital
  received). If the user asks how much they will repay in total, add the principal to it.
- upfront_fee is required for the total cost: use 0 if the user mentions no fee.
- monthly_income is required for the debt ratio. If the user did not give it, do not guess:
  call final_answer asking for it. other_monthly_debts is 0 if the user mentions none.
- annual_rate_pct is a percentage: 8.5 means 8.5%, not 0.085. If the user gives no numeric
  rate, say so in final_answer instead of inventing one.
- Finish by calling final_answer with the number the user asked for, rounded to 2 decimals.
"""


def build(llm: Brain) -> ReActAgent:
    """The cost-and-ratio specialist, sharing the given model with the other agents."""
    registry = ToolRegistry().register(calculate_total_cost, calculate_debt_ratio)
    return ReActAgent(NAME, llm, registry, RULES)