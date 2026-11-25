"""Tools of the ``cuota_tae`` specialist: instalment, APR (TAE) and variable rates.

Design (section 1.2 of ``docs/PLAN.md``), written against the failure documented in ARCA:
the maths was never wrong, the tool interface was. ``loan_metrics`` had seven arguments,
most of them optional, and the model silently omitted ``upfront_fee``, so the APR came out
at 10.36% instead of 12.60%. Hence:

* one tool per metric, so there is one obvious choice;
* ``upfront_fee`` is **required** in ``calculate_apr``: a default of 0 is exactly the silent
  wrong answer we are trying to prevent;
* the instalment the APR needs is computed inside the tool, rounded to cents as a bank
  charges it, so the model never has to carry a number between two calls;
* ``get_variable_rate`` adds the Euribor and the spread itself, so the model never does
  arithmetic in the prompt (that was the ``_repair_sums`` patch of ARCA);
* every tool returns ``{"result": ..., "inputs": {...}}``, so the critic can check which
  numbers were actually used.

The maths lives in ``tool_use.credit_math`` and the validation in ``tool_use.guards``.
``get_euribor`` is re-exported from ``tool_use.tools`` so the specialist registers all its
tools from one module.

No ``from __future__ import annotations`` on purpose: the registry builds the Pydantic
schema from the annotations and needs the real ``Literal``, not a string.
"""

from typing import Any, Literal

from tool_use import credit_math, guards
from tool_use.registry import tool
from tool_use.tools import get_euribor

__all__ = [
    "calculate_apr",
    "calculate_monthly_payment",
    "get_euribor",
    "get_variable_rate",
]

@tool
def calculate_monthly_payment(principal: float, annual_rate_pct: float, months: int) -> dict:
    """Monthly instalment of a consumer loan with French amortisation, in euros.

    The instalment is constant and is returned rounded to cents, which is the figure the
    bank charges and the one every other calculation uses. Always use this tool instead of
    computing the instalment by hand.

    Args:
        principal: Loan amount in euros, for example 12000.
        annual_rate_pct: Nominal annual interest rate in percent: 8.5 means 8.5%, not 0.085.
            For 'Euribor plus a spread', call get_variable_rate first and use its result;
            never add the spread yourself and never guess the Euribor.
        months: Term of the loan in months.
    """
    payment = guards.rounded_payment(principal, annual_rate_pct, months)
    inputs: dict[str, Any] = {
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "months": months,
    }
    return {"result": payment, "inputs": inputs}


@tool
def calculate_apr(
    principal: float, annual_rate_pct: float, months: int, upfront_fee: float
) -> dict:
    """APR (TAE) of a consumer loan, in percent.

    The APR is the rate that makes the instalments equal the money actually received, so it
    includes the opening fee and is higher than the nominal rate whenever there is one. The
    instalment is computed inside this tool, so do not pass it and do not compute it first.

    Args:
        principal: Loan amount in euros, for example 10326.53.
        annual_rate_pct: Nominal annual interest rate in percent: 6.8 means 6.8%, not 0.068.
            For 'Euribor plus a spread', call get_variable_rate first and use its result.
        months: Term of the loan in months.
        upfront_fee: Opening fee in euros, paid at the start. REQUIRED: the APR is wrong
            without it. Use 0 only when the user mentions no fee; if the user gives a fee,
            pass that figure and never 0.
    """
    guards.check_fee(upfront_fee, principal)
    payment = guards.rounded_payment(principal, annual_rate_pct, months)
    apr = credit_math.apr_pct(principal, upfront_fee, payment, months)
    inputs: dict[str, Any] = {
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "months": months,
        "upfront_fee": upfront_fee,
        "monthly_payment": payment,
    }
    return {"result": round(apr, 2), "inputs": inputs}


@tool
def get_variable_rate(tenor: Literal["1M", "3M", "6M", "12M"], spread_pct: float) -> dict:
    """Nominal annual rate of a variable-rate loan: the current Euribor plus the spread.

    Use it whenever the rate is given as 'Euribor plus X' (also 'Euribor más X' or
    'Euribor + X'). It queries the ECB for the Euribor and returns the rate ALREADY ADDED
    UP, in percent: pass that result straight to the other tools as annual_rate_pct. Never
    add the two numbers yourself.

    Args:
        tenor: Euribor maturity: '1M' one month (mensual), '3M' three months (trimestral),
            '6M' six months (semestral), '12M' one year (anual). Consumer loans and
            mortgages usually reference '12M'; use '12M' when the user does not say.
        spread_pct: The spread over the Euribor, in percentage points: for 'Euribor plus
            1.5%' it is 1.5. It is the spread alone, never the whole rate.
    """
    guards.check_spread(spread_pct)
    euribor = get_euribor(tenor=tenor)
    rate = round(euribor["rate_pct"] + spread_pct, 3)
    inputs: dict[str, Any] = {
        "tenor": tenor,
        "spread_pct": spread_pct,
        "euribor_pct": euribor["rate_pct"],
        "euribor_period": euribor["period"],
        "source": euribor["source"],
    }
    return {"result": rate, "inputs": inputs}
