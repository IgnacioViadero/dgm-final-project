"""The four metrics, from the tools, against the 90 problems of ``data/rlm_data_test.jsonl``.

This is the guarantee that the specialists' tools agree with the phase-1 dataset: if a tool
ever drifts from ``credit_math`` or from the rounding convention (instalment rounded to
cents before it is used in anything else), these tests fail. Tolerance is 0.01, the
precision the dataset itself asks the model for.

No network and no GPU: the tools used here are pure arithmetic.
"""

import json
from pathlib import Path

import pytest

from tool_use.tools_coste_ratio import calculate_debt_ratio, calculate_total_cost
from tool_use.tools_cuota_tae import calculate_apr, calculate_monthly_payment

TOLERANCE = 0.01
DATASET = Path(__file__).resolve().parents[2] / "data" / "rlm_data_test.jsonl"
# The dataset's Spanish field names, mapped to the tools' arguments in the helpers below.
TYPES = ("cuota", "coste_total", "tae", "ratio_endeudamiento")


def load(kind: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r["tipo"] == kind]


def ids(rows: list[dict]) -> list[str]:
    return [r["id"] for r in rows]


def test_the_dataset_has_the_ninety_problems_of_the_four_types():
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 90
    assert {r["tipo"] for r in rows} == set(TYPES)


CUOTA = load("cuota")
COSTE = load("coste_total")
TAE = load("tae")
RATIO = load("ratio_endeudamiento")


@pytest.mark.parametrize("row", CUOTA, ids=ids(CUOTA))
def test_monthly_payment_matches_the_dataset(row):
    loan = row["_debug_prestamo"]
    got = calculate_monthly_payment(
        principal=loan["principal"],
        annual_rate_pct=loan["tipo_nominal"],
        months=loan["meses"],
    )
    assert got["result"] == pytest.approx(float(row["answer"]), abs=TOLERANCE)
    assert got["inputs"]["principal"] == loan["principal"]


@pytest.mark.parametrize("row", COSTE, ids=ids(COSTE))
def test_total_cost_matches_the_dataset(row):
    loan = row["_debug_prestamo"]
    got = calculate_total_cost(
        principal=loan["principal"],
        annual_rate_pct=loan["tipo_nominal"],
        months=loan["meses"],
        upfront_fee=loan["comision_apertura"],
    )
    assert got["result"] == pytest.approx(float(row["answer"]), abs=TOLERANCE)
    # The instalment the tool computed internally is the one the dataset used.
    assert got["inputs"]["monthly_payment"] == pytest.approx(row["_debug_cuota"], abs=TOLERANCE)


@pytest.mark.parametrize("row", TAE, ids=ids(TAE))
def test_apr_matches_the_dataset(row):
    loan = row["_debug_prestamo"]
    got = calculate_apr(
        principal=loan["principal"],
        annual_rate_pct=loan["tipo_nominal"],
        months=loan["meses"],
        upfront_fee=loan["comision_apertura"],
    )
    assert got["result"] == pytest.approx(float(row["answer"]), abs=TOLERANCE)
    assert got["inputs"]["monthly_payment"] == pytest.approx(row["_debug_cuota"], abs=TOLERANCE)


@pytest.mark.parametrize("row", RATIO, ids=ids(RATIO))
def test_debt_ratio_matches_the_dataset(row):
    loan = row["_debug_prestamo"]
    got = calculate_debt_ratio(
        monthly_income=loan["ingresos_netos"],
        principal=loan["principal"],
        annual_rate_pct=loan["tipo_nominal"],
        months=loan["meses"],
        other_monthly_debts=loan["otras_deudas"],
    )
    assert got["result"] == pytest.approx(float(row["answer"]), abs=TOLERANCE)
    assert got["inputs"]["monthly_payment"] == pytest.approx(row["_debug_cuota"], abs=TOLERANCE)


# -- the instalment given by the user must give the same answer -----------------------


@pytest.mark.parametrize("row", COSTE + RATIO, ids=ids(COSTE + RATIO))
def test_passing_the_known_instalment_gives_the_same_result(row):
    """These problems state the instalment: passing it must not change the answer."""
    loan, payment = row["_debug_prestamo"], row["_debug_cuota"]
    common = {
        "principal": loan["principal"],
        "annual_rate_pct": loan["tipo_nominal"],
        "months": loan["meses"],
    }
    if row["tipo"] == "coste_total":
        got = calculate_total_cost(
            **common, upfront_fee=loan["comision_apertura"], monthly_payment=payment
        )
    else:
        got = calculate_debt_ratio(
            **common,
            monthly_income=loan["ingresos_netos"],
            other_monthly_debts=loan["otras_deudas"],
            monthly_payment=payment,
        )
    assert got["result"] == pytest.approx(float(row["answer"]), abs=TOLERANCE)


# -- every tool reports the inputs it used --------------------------------------------


def test_every_tool_returns_result_and_inputs():
    results = [
        calculate_monthly_payment(principal=12000, annual_rate_pct=5.5, months=36),
        calculate_apr(principal=12000, annual_rate_pct=5.5, months=36, upfront_fee=150),
        calculate_total_cost(principal=12000, annual_rate_pct=5.5, months=36, upfront_fee=150),
        calculate_debt_ratio(
            monthly_income=1800, principal=12000, annual_rate_pct=5.5, months=36
        ),
    ]
    for got in results:
        assert set(got) == {"result", "inputs"}
        assert isinstance(got["result"], float)
        assert got["inputs"]["principal"] == 12000
