"""The instalment-and-APR specialist, driven by FakeLLM: no GPU, no network.

The scripted replies are what we want the real model to do. The tests that matter are the
last two: the Euribor chain (the tool adds the spread, not the model) and the missing
``upfront_fee`` becoming an observation the agent can recover from. Those are the two ARCA
failures this specialist exists to prevent.
"""

import json

import pytest

from agents import cuota_tae

ECB_REPLY = {
    "dataSets": [{"series": {"0:0:0:0:0:0:0": {"observations": {"0": [2.413]}}}}],
    "structure": {"dimensions": {"observation": [{"values": [{"id": "2026-10"}]}]}},
}


class FakeResponse:
    def __init__(self, body):
        self._body = body
        self.status_code = 200

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


@pytest.fixture
def ecb(monkeypatch):
    """Patch ``requests.get`` with a canned ECB reply and record the calls."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse(ECB_REPLY)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def observation(response, index: int) -> dict:
    """The tool result of one step, parsed back from the observation text."""
    return json.loads(response.steps[index].observation)


# -- wiring ---------------------------------------------------------------------------


def test_registry_has_only_its_own_tools(fake_llm):
    agent = cuota_tae.build(fake_llm([]))
    assert set(agent.registry.tools) == {
        "calculate_monthly_payment",
        "calculate_apr",
        "get_variable_rate",
        "get_euribor",
        "final_answer",
    }


def test_the_module_follows_the_specialist_contract():
    assert cuota_tae.NAME == "cuota_tae"
    assert cuota_tae.RULES.startswith("Additional rules:")
    assert callable(cuota_tae.build)


def test_the_rules_reach_the_system_prompt(fake_llm):
    prompt = cuota_tae.build(fake_llm([])).system_prompt(max_steps=6)
    assert "get_variable_rate" in prompt
    assert "upfront_fee is REQUIRED for the APR" in prompt


# -- the four happy flows -------------------------------------------------------------


def test_monthly_payment_flow(fake_llm):
    llm = fake_llm(
        [
            call(
                "calculate_monthly_payment", principal=3708.23, annual_rate_pct=11.73, months=24
            ),
            call("final_answer", answer="174.09"),
        ]
    )
    response = cuota_tae.build(llm).run(
        "Cuota mensual de 3708.23 euros a 24 meses al 11,73 %"
    )
    assert response.succeeded
    assert observation(response, 0)["result"] == 174.09
    assert response.final_answer == "174.09"


def test_apr_flow_with_a_fee(fake_llm):
    llm = fake_llm(
        [
            call(
                "calculate_apr",
                principal=10326.53,
                annual_rate_pct=6.8,
                months=36,
                upfront_fee=65.86,
            ),
            call("final_answer", answer="7.48"),
        ]
    )
    response = cuota_tae.build(llm).run(
        "TAE de 10326.53 euros a 36 meses al 6,8 % con 65,86 euros de comisión de apertura"
    )
    assert response.succeeded
    assert observation(response, 0)["result"] == 7.48
    assert observation(response, 0)["inputs"]["upfront_fee"] == 65.86
    assert response.final_answer == "7.48"


def test_apr_flow_without_a_fee_passes_zero_explicitly(fake_llm):
    """No fee mentioned is upfront_fee=0, stated by the model, not defaulted silently."""
    llm = fake_llm(
        [
            call("calculate_apr", principal=12000, annual_rate_pct=12.0, months=36, upfront_fee=0),
            call("final_answer", answer="12.68"),
        ]
    )
    response = cuota_tae.build(llm).run("TAE de 12000 euros a 36 meses al 12 %, sin comisiones")
    assert response.succeeded
    result = observation(response, 0)
    assert result["inputs"]["upfront_fee"] == 0
    assert result["result"] == pytest.approx(12.68, abs=0.01)  # above the nominal 12%


def test_euribor_alone_when_there_is_no_loan(fake_llm, ecb):
    llm = fake_llm(
        [
            call("get_euribor", tenor="12M"),
            call("final_answer", answer="El Euríbor 12M es 2,413 %."),
        ]
    )
    response = cuota_tae.build(llm).run("¿Cuánto está el Euríbor a 12 meses?")
    assert response.succeeded
    assert observation(response, 0)["rate_pct"] == 2.413
    assert len(ecb) == 1


# -- the Euribor chain: the tool adds the spread, never the model ---------------------


def test_variable_rate_chains_into_the_payment(fake_llm, ecb):
    llm = fake_llm(
        [
            call("get_variable_rate", tenor="12M", spread_pct=1.5),
            call("calculate_monthly_payment", principal=15000, annual_rate_pct=3.913, months=60),
            call("final_answer", answer="275.66"),
        ]
    )
    response = cuota_tae.build(llm).run(
        "Euríbor + 1,5 %: ¿qué cuota pago por 15.000 euros a 60 meses?"
    )
    assert response.succeeded
    assert response.n_steps == 3

    rate = observation(response, 0)
    assert rate["result"] == 3.913  # 2.413 + 1.5, added up by the tool
    assert rate["inputs"]["euribor_pct"] == 2.413
    assert rate["inputs"]["spread_pct"] == 1.5

    payment = observation(response, 1)
    assert payment["inputs"]["annual_rate_pct"] == 3.913  # the rate came from the observation
    assert payment["result"] == pytest.approx(float(response.final_answer), abs=0.01)
    assert len(ecb) == 1


def test_an_implausible_spread_is_an_observation_and_is_recovered(fake_llm, ecb):
    """A spread that is really a whole rate must not reach the ECB."""
    llm = fake_llm(
        [
            call("get_variable_rate", tenor="12M", spread_pct=24.5),
            call("get_variable_rate", tenor="12M", spread_pct=1.5),
            call("final_answer", answer="3.91"),
        ]
    )
    response = cuota_tae.build(llm).run("Euríbor + 1,5 %")
    assert response.succeeded
    assert response.steps[0].observation.startswith("ERROR")
    assert "only the spread" in response.steps[0].observation
    assert observation(response, 1)["result"] == 3.913
    assert len(ecb) == 1  # the rejected call spent no request


# -- the ARCA bug: a missing required argument ----------------------------------------


def test_missing_upfront_fee_becomes_an_observation_and_is_recovered(fake_llm):
    llm = fake_llm(
        [
            call("calculate_apr", principal=12000, annual_rate_pct=12.0, months=36),
            call(
                "calculate_apr",
                principal=12000,
                annual_rate_pct=12.0,
                months=36,
                upfront_fee=300,
            ),
            call("final_answer", answer="14.67"),
        ]
    )
    response = cuota_tae.build(llm).run(
        "TAE de 12000 euros a 36 meses al 12 % con 300 euros de comisión"
    )
    assert response.steps[0].observation.startswith("ERROR")
    assert "upfront_fee" in response.steps[0].observation
    assert response.succeeded
    assert observation(response, 1)["inputs"]["upfront_fee"] == 300
