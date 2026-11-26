import json
import re

from router.decision import RouteDecision

ROUTER_PROMPT = """You route a credit question to ONE specialist. Reply with JSON only.
- "cuota_tae": calculate the monthly payment (cuota), the APR (TAE), or a rate given as
  Euribor plus a spread.
- "coste_ratio": calculate the total cost of the credit, total interest, total amount repaid,
  debt ratio, or percentage of income going to the instalment.
- "rag": questions with NO figures to calculate: what a concept means (for example what the
  APR is), what the law says, rights and duties of the consumer or the lender, required
  documents or information, withdrawal right. Anything that is not a calculation also goes
  here.
If the question gives amounts, rates or terms to compute, never choose "rag".
Format: {"agent": "cuota_tae" | "coste_ratio" | "rag", "reason": "<short>"}"""

DEFAULT_AGENT = "cuota_tae"


def route(llm, task: str) -> RouteDecision:
    text = llm.generate(
        [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": task}],
        max_new_tokens=64,
    )
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    agent = data.get("agent")
    if agent not in ("cuota_tae", "coste_ratio", "rag"):
        return RouteDecision(DEFAULT_AGENT, "fallback: unparseable router output")
    return RouteDecision(agent, str(data.get("reason", "")))