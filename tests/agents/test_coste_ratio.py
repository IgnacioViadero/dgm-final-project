import json

from agents import coste_ratio


def call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def test_registry_has_only_its_own_tools(fake_llm):
    agent = coste_ratio.build(fake_llm([]))
    assert set(agent.registry.tools) == {
        "calculate_total_cost",
        "calculate_debt_ratio",
        "final_answer",
    }


def test_total_cost_flow(fake_llm):
    llm = fake_llm(
        [
            call("calculate_total_cost", principal=12000, annual_rate_pct=5.5, months=36, upfront_fee=0),
            call("final_answer", answer="1044.60"),
        ]
    )
    response = coste_ratio.build(llm).run("Coste total de 12000 euros a 36 meses al 5,5 %")
    assert response.succeeded
    assert "1044.6" in response.steps[0].observation
    assert response.final_answer == "1044.60"


def test_missing_required_argument_becomes_observation(fake_llm):
    llm = fake_llm(
        [
            call("calculate_debt_ratio", principal=9000, annual_rate_pct=11.2, months=48),
            call("final_answer", answer="Necesito tus ingresos netos mensuales."),
        ]
    )
    response = coste_ratio.build(llm).run("¿Qué ratio tendría con 9000 euros a 48 meses al 11,2 %?")
    assert response.steps[0].observation.startswith("ERROR")
    assert response.succeeded