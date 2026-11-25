"""Validation errors and the variable-rate tool, with ``requests`` mocked: no network.

What matters here is not only that a bad call raises, but that it raises a ``ValueError``
whose message tells the model how to fix the call: the registry turns it into an
observation, and that is the agent's only chance to correct itself.
"""

import pytest

from tool_use.guards import MAX_PLAUSIBLE_RATE_PCT
from tool_use.registry import ToolRegistry
from tool_use.tools_coste_ratio import calculate_debt_ratio, calculate_total_cost
from tool_use.tools_cuota_tae import (
    calculate_apr,
    calculate_monthly_payment,
    get_euribor,
    get_variable_rate,
)

ECB_REPLY = {
    "dataSets": [{"series": {"0:0:0:0:0:0:0": {"observations": {"0": [2.413]}}}}],
    "structure": {"dimensions": {"observation": [{"values": [{"id": "2026-10"}]}]}},
}


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def ecb(monkeypatch):
    """Patch ``requests.get`` with a canned ECB reply and record the calls."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params})
        return FakeResponse(ECB_REPLY)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# -- get_variable_rate: the Euribor and the spread, already added up ------------------


def test_variable_rate_adds_the_spread_itself(ecb):
    got = get_variable_rate(tenor="12M", spread_pct=1.5)
    assert got["result"] == 3.913  # 2.413 + 1.5, computed by the tool, never by the model
    assert got["inputs"] == {
        "tenor": "12M",
        "spread_pct": 1.5,
        "euribor_pct": 2.413,
        "euribor_period": "2026-10",
        "source": "ECB",
    }
    assert len(ecb) == 1


def test_variable_rate_queries_the_requested_tenor(ecb):
    get_variable_rate(tenor="3M", spread_pct=0)
    assert "EURIBOR3MD_" in ecb[0]["url"]
    assert ecb[0]["params"]["lastNObservations"] == 1


def test_variable_rate_with_a_zero_spread_is_the_euribor(ecb):
    assert get_variable_rate(tenor="12M", spread_pct=0)["result"] == 2.413


def test_variable_rate_result_feeds_the_other_tools(ecb):
    """The point of the tool: its result goes straight in as annual_rate_pct."""
    rate = get_variable_rate(tenor="12M", spread_pct=1.5)["result"]
    payment = calculate_monthly_payment(principal=15000, annual_rate_pct=rate, months=60)
    assert payment["inputs"]["annual_rate_pct"] == 3.913
    assert payment["result"] > 0


def test_variable_rate_rejects_a_negative_spread(ecb):
    with pytest.raises(ValueError, match="spread_pct must be >= 0"):
        get_variable_rate(tenor="12M", spread_pct=-0.5)
    assert ecb == []  # rejected before spending a request


def test_variable_rate_rejects_a_whole_rate_as_the_spread(ecb):
    with pytest.raises(ValueError, match="only the spread"):
        get_variable_rate(tenor="12M", spread_pct=24.5)
    assert ecb == []


def test_variable_rate_needs_an_explicit_tenor():
    """No silent default: a tenor the user did not give must be the model's decision."""
    tool_def = get_variable_rate.definition["function"]["parameters"]
    assert set(tool_def["required"]) == {"tenor", "spread_pct"}


def test_unknown_tenor_is_an_observation_not_a_crash(ecb):
    record = ToolRegistry().register(get_variable_rate).call(
        "get_variable_rate", {"tenor": "9M", "spread_pct": 1.0}
    )
    assert not record.ok
    assert "invalid arguments" in record.error
    assert ecb == []


def test_euribor_is_reexported():
    assert get_euribor.name == "get_euribor"


# -- calculate_apr: the ARCA bug, made impossible -------------------------------------


def test_apr_requires_the_upfront_fee():
    """The documented ARCA failure: a missing fee gave 10.36% instead of 12.60%."""
    record = ToolRegistry().register(calculate_apr).call(
        "calculate_apr", {"principal": 12000, "annual_rate_pct": 12.0, "months": 36}
    )
    assert not record.ok
    assert "upfront_fee" in record.error
    assert set(calculate_apr.definition["function"]["parameters"]["required"]) == {
        "principal",
        "annual_rate_pct",
        "months",
        "upfront_fee",
    }


def test_apr_with_a_fee_is_above_the_nominal_rate():
    with_fee = calculate_apr(principal=12000, annual_rate_pct=12.0, months=36, upfront_fee=300)
    without = calculate_apr(principal=12000, annual_rate_pct=12.0, months=36, upfront_fee=0)
    assert with_fee["result"] > without["result"] > 12.0  # monthly compounding alone


def test_apr_does_not_take_the_instalment_from_the_model():
    """The instalment is computed inside, so the model cannot pass a wrong one."""
    assert "monthly_payment" not in calculate_apr.definition["function"]["parameters"][
        "properties"
    ]


def test_apr_rejects_a_fee_that_is_not_a_fee():
    with pytest.raises(ValueError, match="smaller than principal"):
        calculate_apr(principal=300, annual_rate_pct=12.0, months=36, upfront_fee=12000)


def test_apr_rejects_a_negative_fee():
    with pytest.raises(ValueError, match="upfront_fee must be >= 0"):
        calculate_apr(principal=12000, annual_rate_pct=12.0, months=36, upfront_fee=-1)


# -- plausibility guards --------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda rate: calculate_monthly_payment(principal=12000, annual_rate_pct=rate, months=36),
        lambda rate: calculate_apr(
            principal=12000, annual_rate_pct=rate, months=36, upfront_fee=0
        ),
        lambda rate: calculate_total_cost(
            principal=12000, annual_rate_pct=rate, months=36, upfront_fee=0
        ),
        lambda rate: calculate_debt_ratio(
            monthly_income=1800, principal=12000, annual_rate_pct=rate, months=36
        ),
    ],
    ids=["payment", "apr", "total_cost", "debt_ratio"],
)
def test_an_implausible_rate_is_rejected_everywhere(call):
    with pytest.raises(ValueError, match="implausible"):
        call(MAX_PLAUSIBLE_RATE_PCT + 1)
    assert call(MAX_PLAUSIBLE_RATE_PCT)["result"] > 0  # the boundary is accepted


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"principal": 0, "annual_rate_pct": 5.5, "months": 36}, "principal must be > 0"),
        ({"principal": 12000, "annual_rate_pct": 5.5, "months": 0}, "months must be >= 1"),
        ({"principal": 12000, "annual_rate_pct": -1, "months": 36}, "must be >= 0"),
    ],
)
def test_credit_math_errors_reach_the_model(arguments, message):
    with pytest.raises(ValueError, match=message):
        calculate_monthly_payment(**arguments)


def test_a_zero_rate_loan_is_the_principal_split_in_instalments():
    got = calculate_monthly_payment(principal=1200, annual_rate_pct=0, months=12)
    assert got["result"] == 100.0


def test_every_tool_has_a_docstring_the_model_can_read():
    for spec in (calculate_monthly_payment, calculate_apr, get_variable_rate):
        description = spec.definition["function"]["description"]
        assert len(description) > 100
        for name in spec.definition["function"]["parameters"]["properties"]:
            assert name in description  # every argument is explained
