"""The full benchmark's plumbing, with FakeLLM: no GPU, no network.

What matters here is what the smoke script does not have: the run is 180 model calls long,
so it must be resumable. These tests pin that a killed run keeps its finished problems, that
resuming does not ask the model again, and that the per-type counts carry their denominators
(the report puts them next to ARCA's 4/90, 13/90 and 23/90).
"""

import json

import pytest

from eval import numeric_bench


def call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


# -- problem selection ----------------------------------------------------------------


def test_every_problem_of_the_dataset_in_a_reproducible_order():
    problems = numeric_bench.load_all_problems()
    assert len(problems) == 90
    assert [p["id"] for p in problems] == [p["id"] for p in numeric_bench.load_all_problems()]


def test_every_type_has_a_specialist_and_the_counts_are_the_expected_ones():
    problems = numeric_bench.load_all_problems()
    counts = {kind: sum(p["tipo"] == kind for p in problems) for kind in {p["tipo"] for p in problems}}
    # The denominators of ARCA's figure 2, which the report compares against.
    assert counts == {
        "coste_total": 19,
        "cuota": 23,
        "ratio_endeudamiento": 27,
        "tae": 21,
    }


def test_limit_keeps_the_first_problems():
    assert numeric_bench.load_all_problems(limit=3) == numeric_bench.load_all_problems()[:3]


# -- the sidecar: surviving a killed run ----------------------------------------------


def test_load_done_of_a_missing_sidecar_is_empty(tmp_path):
    assert numeric_bench.load_done(tmp_path / "nothing.jsonl") == {}


def test_a_half_written_last_line_is_dropped_not_raised(tmp_path):
    sidecar = tmp_path / "rows.jsonl"
    good = {"id": "cuota_00001", "agent": "cuota_tae", "correct": True}
    sidecar.write_text(
        json.dumps(good) + "\n" + '{"id": "cuota_00002", "agent": "cuo',  # killed mid-write
        encoding="utf-8",
    )
    done = numeric_bench.load_done(sidecar)
    assert list(done) == ["cuota_00001::cuota_tae"]


def test_rows_are_flushed_as_the_run_goes_and_resuming_skips_them(tmp_path, fake_llm):
    problems = [p for p in numeric_bench.load_all_problems() if p["tipo"] == "cuota"][:1]
    row = problems[0]
    loan = row["_debug_prestamo"]
    replies = []
    for _ in range(2):  # the specialist and the baseline each answer this problem
        replies.append(
            call(
                "calculate_monthly_payment",
                principal=loan["principal"],
                annual_rate_pct=loan["tipo_nominal"],
                months=loan["meses"],
            )
        )
        replies.append(call("final_answer", answer=row["answer"]))
    llm = fake_llm(replies)
    sidecar = tmp_path / "rows.jsonl"

    rows = numeric_bench.run_all_incremental(problems, llm, sidecar, max_steps=6)

    assert len(rows) == 2  # one per architecture
    assert {r["agent"] for r in rows} == {"cuota_tae", "monolithic"}
    assert len(numeric_bench.load_done(sidecar)) == 2

    # Resuming asks the model nothing: a brain with no replies left must not be called.
    exhausted = fake_llm([])
    again = numeric_bench.run_all_incremental(
        problems, exhausted, sidecar, max_steps=6, done=numeric_bench.load_done(sidecar)
    )
    assert [r["id"] for r in again] == [r["id"] for r in rows]
    assert exhausted.last_tools is None  # generate was never reached


def test_resuming_only_runs_what_is_missing(tmp_path, fake_llm):
    problems = [p for p in numeric_bench.load_all_problems() if p["tipo"] == "cuota"][:1]
    row = problems[0]
    sidecar = tmp_path / "rows.jsonl"
    # The specialist finished before the run died; only the baseline is left.
    sidecar.write_text(
        json.dumps(
            {
                "id": row["id"],
                "tipo": row["tipo"],
                "agent": "cuota_tae",
                "correct": True,
                "n_steps": 2,
                "tokens": 10,
                "latency_s": 1.0,
                "expected": row["answer"],
                "answer": row["answer"],
                "tools": ["calculate_monthly_payment", "final_answer"],
                "trace": [],
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llm = fake_llm([call("final_answer", answer=row["answer"])])
    rows = numeric_bench.run_all_incremental(
        problems, llm, sidecar, max_steps=6, done=numeric_bench.load_done(sidecar)
    )
    assert len(rows) == 2
    assert llm.replies == []  # exactly the one missing run consumed its reply


# -- the per-type table ---------------------------------------------------------------


def make_row(tipo, agent, correct):
    return {
        "id": f"{tipo}_x",
        "tipo": tipo,
        "agent": agent,
        "correct": correct,
        "n_steps": 2,
        "tokens": 1,
        "latency_s": 0.1,
        "tools": ["final_answer"],
    }


def test_by_type_counts_keeps_the_denominators():
    rows = [
        make_row("cuota", "cuota_tae", True),
        make_row("cuota", "cuota_tae", False),
        make_row("tae", "cuota_tae", True),
        make_row("cuota", "monolithic", False),
        make_row("cuota", "monolithic", False),
        make_row("tae", "monolithic", False),
    ]
    counts = numeric_bench.by_type_counts(rows)
    assert counts["specialists"]["cuota"] == {"hits": 1, "n": 2}
    assert counts["specialists"]["tae"] == {"hits": 1, "n": 1}
    assert counts["specialists"]["total"] == {"hits": 2, "n": 3}
    assert counts["monolithic"]["total"] == {"hits": 0, "n": 3}
    # A type nobody was asked about is reported as 0/0, not missing.
    assert counts["specialists"]["coste_total"] == {"hits": 0, "n": 0}


def test_row_key_separates_the_two_architectures():
    assert numeric_bench.row_key("cuota_00001", "cuota_tae") != numeric_bench.row_key(
        "cuota_00001", "monolithic"
    )


# -- several datasets in one run ------------------------------------------------------

TEST_SPLIT = numeric_bench.REPO_ROOT / "data" / "rlm_data_test.jsonl"
TRAIN_SPLIT = numeric_bench.REPO_ROOT / "data" / "rlm_data_train.jsonl"


def test_several_datasets_are_concatenated_in_the_order_given():
    both = numeric_bench.load_all_problems([TEST_SPLIT, TRAIN_SPLIT])
    assert len(both) == 600
    assert both[:90] == numeric_bench.load_all_problems(TEST_SPLIT)
    assert {p["split"] for p in both} == {"test", "train"}
    assert sum(p["split"] == "test" for p in both) == 90
    assert sum(p["split"] == "train" for p in both) == 510


def test_every_problem_says_which_file_it_came_from():
    problems = numeric_bench.load_all_problems([TRAIN_SPLIT])
    assert {p["dataset"] for p in problems} == {str(TRAIN_SPLIT)}
    assert {p["split"] for p in problems} == {"train"}


def test_limit_applies_to_each_dataset():
    assert len(numeric_bench.load_all_problems([TEST_SPLIT, TRAIN_SPLIT], limit=4)) == 8


def test_an_id_in_two_datasets_is_an_error_because_the_sidecar_keys_on_it(tmp_path):
    # The real files do not collide, but a repeated --dataset would resume wrongly.
    with pytest.raises(ValueError, match="unique across the datasets"):
        numeric_bench.load_all_problems([TEST_SPLIT, TEST_SPLIT])


def test_the_rows_carry_the_split_so_test_and_train_can_be_reported_apart(tmp_path, fake_llm):
    problems = [p for p in numeric_bench.load_all_problems(TRAIN_SPLIT) if p["tipo"] == "cuota"][:1]
    row = problems[0]
    loan = row["_debug_prestamo"]
    replies = []
    for _ in range(2):
        replies.append(
            call(
                "calculate_monthly_payment",
                principal=loan["principal"],
                annual_rate_pct=loan["tipo_nominal"],
                months=loan["meses"],
            )
        )
        replies.append(call("final_answer", answer=row["answer"]))
    rows = numeric_bench.run_all_incremental(
        problems, fake_llm(replies), tmp_path / "rows.jsonl", max_steps=6
    )
    assert {r["split"] for r in rows} == {"train"}
    assert {r["dataset"] for r in rows} == {str(TRAIN_SPLIT)}


def test_by_split_counts_separates_the_splits_and_keeps_the_total():
    rows = [
        {**make_row("cuota", "cuota_tae", True), "split": "test"},
        {**make_row("cuota", "cuota_tae", False), "split": "train"},
        {**make_row("cuota", "monolithic", True), "split": "test"},
        {**make_row("cuota", "monolithic", True), "split": "train"},
    ]
    counts = numeric_bench.by_split_counts(rows)
    assert list(counts) == ["test", "train", "all"]
    assert counts["test"]["specialists"]["total"] == {"hits": 1, "n": 1}
    assert counts["train"]["specialists"]["total"] == {"hits": 0, "n": 1}
    assert counts["all"]["specialists"]["total"] == {"hits": 1, "n": 2}
    assert counts["all"]["monolithic"]["total"] == {"hits": 2, "n": 2}


def test_split_name_is_the_short_name_of_the_file():
    assert numeric_bench.split_name(TEST_SPLIT) == "test"
    assert numeric_bench.split_name(TRAIN_SPLIT) == "train"


# -- one architecture per process -----------------------------------------------------


def test_only_the_specialists_run_when_asked_for(tmp_path, fake_llm):
    problems = [p for p in numeric_bench.load_all_problems(TEST_SPLIT) if p["tipo"] == "cuota"][:1]
    row = problems[0]
    loan = row["_debug_prestamo"]
    llm = fake_llm(
        [
            call(
                "calculate_monthly_payment",
                principal=loan["principal"],
                annual_rate_pct=loan["tipo_nominal"],
                months=loan["meses"],
            ),
            call("final_answer", answer=row["answer"]),
        ]
    )
    rows = numeric_bench.run_all_incremental(
        problems, llm, tmp_path / "rows.jsonl", max_steps=6, architectures=("specialists",)
    )
    assert [r["agent"] for r in rows] == ["cuota_tae"]
    assert llm.replies == []


def test_only_the_baseline_runs_when_asked_for(tmp_path, fake_llm):
    problems = [p for p in numeric_bench.load_all_problems(TEST_SPLIT) if p["tipo"] == "cuota"][:1]
    row = problems[0]
    llm = fake_llm([call("final_answer", answer=row["answer"])])
    rows = numeric_bench.run_all_incremental(
        problems, llm, tmp_path / "rows.jsonl", max_steps=6, architectures=("monolithic",)
    )
    assert [r["agent"] for r in rows] == ["monolithic"]


def test_an_unknown_architecture_is_rejected(tmp_path, fake_llm):
    with pytest.raises(ValueError, match="unknown architectures"):
        numeric_bench.run_all_incremental(
            [], fake_llm([]), tmp_path / "rows.jsonl", architectures=("router",)
        )
