"""The ARCA baseline, driven by FakeLLM. It exists to be compared, not to be improved.

These tests pin the baseline's shape so ablation A stays honest: if someone later gives the
monolithic agent the specialists' small tools, these tests fail and the comparison is no
longer 'one big tool vs several small ones'.
"""

import json

from agents import monolithic


def call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def test_registry_has_only_its_own_tools(fake_llm):
    agent = monolithic.build(fake_llm([]))
    assert set(agent.registry.tools) == {"loan_metrics", "get_euribor", "final_answer"}


def test_the_module_follows_the_specialist_contract():
    assert monolithic.NAME == "monolithic"
    assert monolithic.RULES.startswith("Additional rules:")
    assert callable(monolithic.build)


def test_loan_metrics_keeps_its_seven_arguments(fake_llm):
    """The baseline's weakness is its interface: many arguments, mostly optional."""
    agent = monolithic.build(fake_llm([]))
    parameters = agent.registry.tools["loan_metrics"].definition["function"]["parameters"]
    assert len(parameters["properties"]) == 7
    # upfront_fee is optional here: that is the documented cause of the wrong APR.
    assert set(parameters.get("required", [])) == {"principal", "annual_rate_pct", "months"}


def test_happy_flow_reads_one_field_of_the_single_result(fake_llm):
    llm = fake_llm(
        [
            call(
                "loan_metrics",
                principal=12000,
                annual_rate_pct=5.5,
                months=36,
                upfront_fee=0,
                monthly_income=1800,
            ),
            call("final_answer", answer="362.35"),
        ]
    )
    response = monolithic.build(llm).run(
        "Cuota de 12000 euros a 36 meses al 5,5 % ganando 1800 euros"
    )
    assert response.succeeded
    assert response.n_steps == 2
    result = json.loads(response.steps[0].observation)
    # One call, every metric: that is the baseline's design.
    assert result == {
        "monthly_payment": 362.35,
        "total_cost": 1044.6,
        "apr_pct": 5.64,
        "debt_ratio_pct": 20.13,
    }
    assert response.final_answer == "362.35"
    assert response.tokens_used > 0


def test_the_documented_arca_bug_still_reproduces(fake_llm):
    """Omitting upfront_fee gives a wrong APR silently: no error, just a bad number.

    This is the behaviour the specialists' required argument prevents. It is asserted here
    so the report can point at a test instead of at an anecdote.
    """
    agent = monolithic.build(fake_llm([]))
    without_fee = agent.registry.call(
        "loan_metrics", {"principal": 12000, "annual_rate_pct": 12.0, "months": 36}
    )
    with_fee = agent.registry.call(
        "loan_metrics",
        {"principal": 12000, "annual_rate_pct": 12.0, "months": 36, "upfront_fee": 300},
    )
    assert without_fee.ok and with_fee.ok  # neither call complains
    assert without_fee.result["apr_pct"] != with_fee.result["apr_pct"]
    assert with_fee.result["apr_pct"] > without_fee.result["apr_pct"]
