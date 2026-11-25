"""Validation shared by the specialists' tools, and the rounding convention of the domain.

Two reasons for this module. First, a single source of truth: the plausibility limits and
the "instalment rounded to cents before it is used in anything else" rule must be identical
in every tool, or the same question answered by two agents would give two numbers. Second,
the error messages are part of the product: the registry turns a ``ValueError`` into an
observation, and that message is the agent's only chance to correct its own call, so it is
written for the model, not for a developer.

``tool_use/tools.py`` (the monolithic ARCA baseline) deliberately keeps its own copy of
these checks: it is the "before" of the comparison and must not move.

These are plain helpers, not tools, so the registry never reads their annotations.
"""

from tool_use import credit_math

# Nominal rates above this are almost surely a mistake (for example a rate guessed before
# the Euribor tool returned). Rejecting them turns a silent wrong answer into an observation.
MAX_PLAUSIBLE_RATE_PCT = 60.0
# A spread over the Euribor is a couple of points, not a rate. A spread this large is almost
# always the model passing the whole rate, or the Euribor itself, as the spread.
MAX_PLAUSIBLE_SPREAD_PCT = 20.0


def check_rate(annual_rate_pct: float) -> float:
    """Reject a rate that cannot be a nominal annual percentage for a consumer loan."""
    if annual_rate_pct > MAX_PLAUSIBLE_RATE_PCT:
        raise ValueError(
            f"annual_rate_pct={annual_rate_pct:g} is implausible (expected 0 to "
            f"{MAX_PLAUSIBLE_RATE_PCT:g}, in percent: 8.5 means 8.5%, not 0.085). If the rate "
            "depends on another tool's result (for example the Euribor), call get_variable_rate "
            "first, wait for its result, and then call this tool again."
        )
    return annual_rate_pct


def check_spread(spread_pct: float) -> float:
    """Reject a spread over the Euribor that is really a whole rate."""
    if spread_pct < 0:
        raise ValueError("spread_pct must be >= 0: a loan is not priced below the Euribor")
    if spread_pct > MAX_PLAUSIBLE_SPREAD_PCT:
        raise ValueError(
            f"spread_pct={spread_pct:g} is implausible for a spread over the Euribor (expected "
            f"0 to {MAX_PLAUSIBLE_SPREAD_PCT:g} percentage points). Pass only the spread, not "
            "the whole rate and not the Euribor."
        )
    return spread_pct


def check_fee(upfront_fee: float, principal: float) -> float:
    """Reject an opening fee that is negative or that is really the loan amount."""
    if upfront_fee < 0:
        raise ValueError("upfront_fee must be >= 0 (use 0 only when the user mentions no fee)")
    if upfront_fee >= principal:
        raise ValueError(
            f"upfront_fee={upfront_fee:g} must be smaller than principal={principal:g}: check "
            "which number is the fee and which is the loan amount"
        )
    return upfront_fee


def check_other_debts(other_monthly_debts: float) -> float:
    """Reject negative monthly debts."""
    if other_monthly_debts < 0:
        raise ValueError("other_monthly_debts must be >= 0 (use 0 when there are none)")
    return other_monthly_debts


def rounded_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """The instalment in euros, rounded to cents: the figure every other metric uses.

    This is the convention of the phase-1 dataset and of ``credit_math``: a bank charges
    whole cents, so the total cost and the APR are computed from the rounded instalment.
    """
    check_rate(annual_rate_pct)
    return round(credit_math.monthly_payment(principal, annual_rate_pct, months), 2)


def resolve_payment(
    principal: float, annual_rate_pct: float, months: int, monthly_payment: float | None
) -> float:
    """The instalment: the one the user gave, rounded to cents, or the one computed here."""
    if monthly_payment is None:
        return rounded_payment(principal, annual_rate_pct, months)
    check_rate(annual_rate_pct)
    if monthly_payment <= 0:
        raise ValueError("monthly_payment must be > 0, or omitted so it is computed here")
    return round(monthly_payment, 2)
