"""The statistics and the failure classifier, on hand-made rows: no GPU, no network.

These are the numbers the report quotes, so what is pinned here is that they are the right
numbers: Wilson against values computed by hand, McNemar against the cases where the answer
is obvious (no discordant pairs -> p=1), and the failure classifier against one row per
mechanism, taken from the shapes the real traces have.
"""

import json

import pytest

from eval import stats_numeric as stats


def row(
    pid="cuota_00001",
    tipo="cuota",
    agent="cuota_tae",
    correct=True,
    expected="100.00",
    answer="100.00",
    split="test",
    tools=("calculate_monthly_payment", "final_answer"),
    trace=None,
    error=None,
):
    return {
        "id": pid,
        "tipo": tipo,
        "agent": agent,
        "correct": correct,
        "expected": expected,
        "answer": answer,
        "split": split,
        "tools": list(tools),
        "trace": trace if trace is not None else [],
        "error": error,
        "n_steps": 2,
        "tokens": 1,
        "latency_s": 0.1,
    }


def step(tool, arguments, observation):
    return {"tool": tool, "arguments": arguments, "observation": observation}


# -- Wilson ----------------------------------------------------------------------------


def test_wilson_of_a_perfect_score_has_width_and_does_not_reach_one():
    interval = stats.wilson_interval(600, 600)
    assert interval["point"] == 1.0
    assert interval["high"] == 1.0
    assert 0.99 < interval["low"] < 1.0  # the normal approximation would say 1.0


def test_wilson_matches_the_textbook_value():
    interval = stats.wilson_interval(50, 100)
    assert interval["low"] == pytest.approx(0.4038, abs=1e-3)
    assert interval["high"] == pytest.approx(0.5962, abs=1e-3)


def test_wilson_of_nothing_is_the_whole_range():
    assert stats.wilson_interval(0, 0) == {
        "hits": 0,
        "n": 0,
        "point": 0.0,
        "low": 0.0,
        "high": 1.0,
    }


# -- McNemar ---------------------------------------------------------------------------


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    rows = [
        row(pid="a", agent="cuota_tae", correct=True),
        row(pid="a", agent="monolithic", correct=True),
        row(pid="b", agent="cuota_tae", correct=False),
        row(pid="b", agent="monolithic", correct=False),
    ]
    result = stats.mcnemar(rows)
    assert result["n_paired"] == 2
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_counts_each_direction_and_one_discordant_pair_is_not_significant():
    rows = [
        row(pid="a", agent="cuota_tae", correct=True),
        row(pid="a", agent="monolithic", correct=False),
    ]
    result = stats.mcnemar(rows)
    assert (result["only_specialists"], result["only_monolithic"]) == (1, 0)
    assert result["p_value"] == 1.0


def test_mcnemar_with_ten_failures_on_one_side_only_is_significant():
    rows = []
    for i in range(10):
        rows.append(row(pid=f"p{i}", agent="cuota_tae", correct=True))
        rows.append(row(pid=f"p{i}", agent="monolithic", correct=False))
    result = stats.mcnemar(rows)
    assert result["discordant"] == 10
    assert result["p_value"] == pytest.approx(2 / 2**10)


def test_mcnemar_ignores_problems_only_one_architecture_answered():
    rows = [
        row(pid="a", agent="cuota_tae", correct=True),
        row(pid="a", agent="monolithic", correct=False),
        row(pid="b", agent="cuota_tae", correct=True),  # the other process got no further
    ]
    assert stats.mcnemar(rows)["n_paired"] == 1


def test_mcnemar_can_be_asked_for_one_split():
    rows = [
        row(pid="a", agent="cuota_tae", correct=True, split="test"),
        row(pid="a", agent="monolithic", correct=False, split="test"),
        row(pid="b", agent="cuota_tae", correct=True, split="train"),
        row(pid="b", agent="monolithic", correct=True, split="train"),
    ]
    assert stats.mcnemar(rows, "test")["n_paired"] == 1
    assert stats.mcnemar(rows, "all")["n_paired"] == 2


# -- the table -------------------------------------------------------------------------


def test_the_table_separates_the_splits_and_keeps_the_denominators():
    rows = [
        row(pid="a", tipo="cuota", agent="cuota_tae", correct=True, split="test"),
        row(pid="b", tipo="tae", agent="cuota_tae", correct=False, split="train"),
        row(pid="a", tipo="cuota", agent="monolithic", correct=False, split="test"),
    ]
    table = stats.counts_table(rows)
    assert table["test"]["specialists"]["cuota"] == {"hits": 1, "n": 1}
    assert table["train"]["specialists"]["tae"] == {"hits": 0, "n": 1}
    assert table["all"]["specialists"]["total"] == {"hits": 1, "n": 2}
    assert table["all"]["monolithic"]["total"] == {"hits": 0, "n": 1}
    assert table["test"]["specialists"]["ratio_endeudamiento"] == {"hits": 0, "n": 0}


# -- the failure classifier ------------------------------------------------------------


def test_an_exception_is_its_own_mechanism():
    assert stats.classify_failure(row(correct=False, error="RuntimeError: cuda", tools=())) == (
        "excepcion"
    )


def test_no_tool_call_at_all():
    assert stats.classify_failure(row(correct=False, tools=())) == "no llama a ninguna tool"


def test_running_out_of_steps_without_answering():
    failure = row(
        correct=False,
        tools=("calculate_apr",),
        trace=[step("calculate_apr", {"principal": 1.0}, '{"result": 12.6}')],
    )
    assert stats.classify_failure(failure) == "se agota sin contestar (sin final_answer)"


def test_answering_with_a_number_from_inputs_instead_of_result():
    failure = row(
        correct=False,
        expected="12.60",
        answer="La TAE es del 9.26%",
        trace=[
            step(
                "calculate_apr",
                {"principal": 17050.72, "annual_rate_pct": 9.26, "months": 36, "upfront_fee": 254.37},
                json.dumps({"result": 12.60, "inputs": {"annual_rate_pct": 9.26}}),
            ),
            step("final_answer", {"answer": "9.26"}, "9.26"),
        ],
    )
    assert stats.classify_failure(failure) == (
        "contesta con un argumento/input en vez de con el result"
    )


def test_answering_with_another_field_of_the_observation():
    failure = row(
        correct=False,
        agent="monolithic",
        expected="544.27",
        answer="2797.37",
        trace=[
            step(
                "loan_metrics",
                {"principal": 17050.72, "annual_rate_pct": 7.0, "months": 36},
                json.dumps({"monthly_payment": 544.27, "total_cost": 2797.37}),
            ),
            step("final_answer", {"answer": "2797.37"}, "2797.37"),
        ],
    )
    assert stats.classify_failure(failure) == "contesta con otro campo de la observacion"


def test_an_invented_number_is_in_no_part_of_the_trace():
    failure = row(
        correct=False,
        expected="544.27",
        answer="124.32",
        trace=[
            step(
                "calculate_monthly_payment",
                {"principal": 17050.72, "annual_rate_pct": 9.26, "months": 36},
                json.dumps({"result": 544.27, "inputs": {"months": 36}}),
            ),
            step("final_answer", {"answer": "124.32"}, "124.32"),
        ],
    )
    assert stats.classify_failure(failure) == "cifra inventada (no esta en la traza)"


def test_a_wrong_argument_shows_up_as_the_right_number_never_appearing():
    failure = row(
        correct=False,
        expected="544.27",
        answer="500.00",
        trace=[
            step(
                "calculate_monthly_payment",
                {"principal": 17050.72, "annual_rate_pct": 9.26, "months": 48},
                json.dumps({"result": 500.0, "inputs": {"months": 48}}),
            ),
            step("final_answer", {"answer": "500.00"}, "500.00"),
        ],
    )
    assert stats.classify_failure(failure) == (
        "argumento mal extraido o tool equivocada (la observacion nunca trae el numero)"
    )


def test_a_tool_error_is_told_apart_from_a_wrong_argument():
    failure = row(
        correct=False,
        expected="544.27",
        answer="no he podido",
        trace=[
            step("calculate_apr", {"principal": -1}, "ValueError: principal must be positive"),
            step("final_answer", {"answer": "no he podido"}, "no he podido"),
        ],
    )
    assert stats.classify_failure(failure) == "la tool devuelve error (argumentos invalidos)"


def test_the_groups_are_ordered_by_size_and_carry_one_example_each():
    rows = [
        row(pid="a", correct=False, tools=()),
        row(pid="b", correct=False, tools=()),
        row(pid="c", correct=False, error="boom", tools=()),
        row(pid="d", correct=True),
    ]
    report = stats.failure_report(rows)
    groups = report["specialists"]["groups"]
    assert report["specialists"]["n_failures"] == 3
    assert [g["n"] for g in groups] == [2, 1]
    assert groups[0]["mechanism"] == "no llama a ninguna tool"
    assert groups[0]["example"]["id"] == "a"


# -- reading the files -----------------------------------------------------------------


def test_rows_are_read_from_sidecars_and_deduplicated(tmp_path):
    first = tmp_path / "a.rows.jsonl"
    second = tmp_path / "b.rows.jsonl"
    first.write_text(
        json.dumps(row(pid="a", correct=False)) + "\n" + json.dumps(row(pid="a", correct=True)) + "\n",
        encoding="utf-8",
    )
    second.write_text(json.dumps(row(pid="a", agent="monolithic")) + "\n", encoding="utf-8")
    rows = stats.load_rows([first, second])
    assert len(rows) == 2  # (a, cuota_tae) once, (a, monolithic) once
    assert [r["correct"] for r in rows if r["agent"] == "cuota_tae"] == [True]  # the last wins


def test_rows_can_also_be_read_from_a_benchmark_json(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"rows": [row(pid="a")]}), encoding="utf-8")
    assert [r["id"] for r in stats.load_rows([path])] == ["a"]


def test_an_omitted_optional_argument_is_its_own_mechanism():
    # The baseline's loan_metrics takes monthly_income as optional: not sending it cannot
    # raise, so the field comes back null and the model has no number to report.
    failure = row(
        agent="monolithic",
        correct=False,
        tipo="ratio_endeudamiento",
        expected="33.96",
        answer="Please provide your monthly income to calculate the debt ratio.",
        trace=[
            step(
                "loan_metrics",
                {"principal": 24847.9, "annual_rate_pct": 7.94, "months": 60},
                '{"monthly_payment": 503.11, "debt_ratio_pct": null}',
            ),
            step("final_answer", {"answer": "Please provide your monthly income"}, "..."),
        ],
    )
    assert stats.classify_failure(failure) == "omite un argumento opcional y la tool devuelve null"
