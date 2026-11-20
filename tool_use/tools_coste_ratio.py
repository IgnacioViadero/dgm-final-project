"""Tools of the ``coste_ratio`` specialist: total cost of the credit and debt ratio.

Design (section 1.2 of ``docs/PLAN.md``): small tools, one metric each, with **required**
arguments wherever a silent default would produce a wrong answer. Each tool does its own
intermediate arithmetic (it computes the instalment internally, so the agent is
self-sufficient) and returns ``{"result": ..., "inputs": {...}}`` so the critic can check
which numbers were actually used.

The maths lives in ``tool_use.credit_math`` and the validation in ``tool_use.guards``;
neither is repeated here. The convention of the domain is kept: the instalment is rounded
to cents before being used in anything else.

No ``from __future__ import annotations`` on purpose: the registry builds the Pydantic
schema from the annotations and needs the real objects, not strings.
"""

from typing import Any

from tool_use import credit_math, guards
from tool_use.registry import tool


@tool
def calculate_total_cost(
    principal: float,
    annual_rate_pct: float,
    months: int,
    upfront_fee: float,
    monthly_payment: float | None = None,
) -> dict:
    """Total cost of a consumer loan: the interest plus the fees, in euros.

    The result is everything paid minus the capital received, that is
    ``monthly_payment * months + upfront_fee - principal``. If the user asks how much they
    will repay in total, add the principal to this result. The instalment is computed here
    from the principal, the rate and the term, so never calculate it by hand.

    Args:
        principal: Loan amount in euros, for example 12000.
        annual_rate_pct: Nominal annual interest rate in percent: 5.5 means 5.5%, not 0.055.
            For 'Euribor plus a spread' use the already added-up rate from the rate tool;
            never guess the Euribor.
        months: Term of the loan in months.
        upfront_fee: Opening fee in euros, paid at the start. Required: use 0 only when the
            user mentions no fee.
        monthly_payment: Instalment in euros if the user already gave it. Omit it to have it
            computed from the principal, the rate and the term.
    """
    payment = guards.resolve_payment(principal, annual_rate_pct, months, monthly_payment)
    guards.check_fee(upfront_fee, principal)
    cost = credit_math.total_cost(payment, months, upfront_fee, principal)
    inputs: dict[str, Any] = {
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "months": months,
        "upfront_fee": upfront_fee,
        "monthly_payment": payment,
    }
    return {"result": round(cost, 2), "inputs": inputs}


@tool
def calculate_debt_ratio(
    monthly_income: float,
    principal: float,
    annual_rate_pct: float,
    months: int,
    other_monthly_debts: float = 0.0,
    monthly_payment: float | None = None,
) -> dict:
    """Debt ratio in percent: the share of net monthly income that goes to debt service.

    The result is ``(monthly_payment + other_monthly_debts) / monthly_income * 100``. The
    instalment of this loan is computed here, so never calculate it by hand.

    Args:
        monthly_income: Net monthly income in euros. Required: if the user did not give it,
            do not guess, ask for it with final_answer.
        principal: Loan amount in euros, for example 9000.
        annual_rate_pct: Nominal annual interest rate in percent: 11.2 means 11.2%.
        months: Term of the loan in months.
        other_monthly_debts: Other monthly debt payments in euros. 0 when the user mentions
            none.
        monthly_payment: Instalment in euros if the user already gave it. Omit it to have it
            computed from the principal, the rate and the term.
    """
    payment = guards.resolve_payment(principal, annual_rate_pct, months, monthly_payment)
    guards.check_other_debts(other_monthly_debts)
    ratio = credit_math.debt_ratio_pct(payment, other_monthly_debts, monthly_income)
    inputs: dict[str, Any] = {
        "monthly_income": monthly_income,
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "months": months,
        "other_monthly_debts": other_monthly_debts,
        "monthly_payment": payment,
    }
    return {"result": round(ratio, 2), "inputs": inputs}
