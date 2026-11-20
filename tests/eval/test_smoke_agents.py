"""The smoke script's plumbing, exercised with FakeLLM: no GPU, no network.

The script itself needs a real brain to be useful, but its selection of problems, its
routing by type, its correctness check and its summary must not be discovered to be broken
on the DGX.
"""

import json

import pytest

from eval import smoke_agents


def call(name: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


# -- problem selection ----------------------------------------------------------------


def test_two_problems_of_each_type_in_a_reproducible_order():
    problems = smoke_agents.load_problems()
    assert len(problems) == 8
    types = [p["tipo"] for p in problems]
    for kind in smoke_agents.SPECIALIST_OF_TYPE:
        assert types.count(kind) == 2
    assert [p["id"] for p in problems] == [p["id"] for p in smoke_agents.load_problems()]


def test_every_type_has_an_owner():
    assert set(smoke_agents.SPECIALIST_OF_TYPE) == {
        "cuota",
        "tae",
        "coste_total",
        "ratio_endeudamiento",
    }
    for module in smoke_agents.SPECIALIST_OF_TYPE.values():
        assert module.NAME in {"cuota_tae", "coste_ratio"}


def test_the_output_format_instruction_is_stripped_from_the_task():
    row = smoke_agents.load_problems()[0]
    task = smoke_agents.task_of(row)
    assert "<answer>" not in task
    assert "Da la respuesta final" not in task
    assert "préstamo" in task  # the question itself survives


# -- the correctness check ------------------------------------------------------------


@pytest.mark.parametrize(
    "expected, answer, ok",
    [
        ("174.09", "174.09", True),
        ("174.09", "La cuota es de 174.09 euros al mes.", True),
        ("174.09", "174,09", True),  # a Spanish decimal comma
        ("174.09", "174.10", True),  # exactly at the 0.01 tolerance, which is inclusive
        ("174.09", "174.11", False),  # outside it
        ("174.09", "No puedo calcularlo.", False),
        ("7.48", "La TAE es del 7.48 % para un préstamo de 36 meses", True),
        # Spanish notation: dot for thousands, comma for decimals. Reading "3.930,84" as
        # 3.930 and 84 scored two correct answers of the 27B as wrong (2026-11-19).
        ("3930.84", "El coste total del crédito es de 3.930,84 euros.", True),
        ("3930.84", "3.930,84 euros (16.848,00 € de cuotas - 12.939,59 € de capital)", True),
        ("2803.54", "El coste total es de 2.803,54 euros.", True),
        ("16848", "16.848,00 € de cuotas", True),
        ("3930.84", "El coste total es de 3.931,84 euros.", False),  # still wrong is wrong
        ("174.09", "174.09 euros", True),  # a plain decimal point is not a separator
    ],
)
def test_is_correct(expected, answer, ok):
    assert smoke_agents.is_correct(expected, answer) is ok


def test_numbers_in_reads_both_notations():
    assert smoke_agents.numbers_in("3.930,84 y 174.09 y 1.200 y 45,88") == [
        3930.84,
        174.09,
        1200.0,
        45.88,
    ]


def test_trailing_punctuation_is_not_part_of_the_number():
    assert smoke_agents.numbers_in("son 174.09.") == [174.09]


# -- one problem through one agent ----------------------------------------------------


def test_run_one_records_the_tools_and_the_tokens(fake_llm):
    from agents import cuota_tae

    row = [p for p in smoke_agents.load_problems() if p["tipo"] == "cuota"][0]
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
    result = smoke_agents.run_one(cuota_tae.build(llm), row, max_steps=6)
    assert result["correct"]
    assert result["agent"] == "cuota_tae"
    assert result["tools"] == ["calculate_monthly_payment", "final_answer"]
    assert result["n_steps"] == 2
    assert result["tokens"] > 0
    assert result["error"] is None


def test_a_wrong_answer_is_recorded_not_raised(fake_llm):
    from agents import cuota_tae

    row = [p for p in smoke_agents.load_problems() if p["tipo"] == "cuota"][0]
    llm = fake_llm([call("final_answer", answer="0.00")])
    result = smoke_agents.run_one(cuota_tae.build(llm), row, max_steps=6)
    assert not result["correct"]
    assert result["tools"] == ["final_answer"]


def test_a_crashing_brain_is_a_failed_row_not_a_crashed_run():
    class BrokenLLM:
        model_id = "broken"

        def generate(self, messages, max_new_tokens=None, tools=None):
            raise RuntimeError("the service is down")

        def count_tokens(self, text):
            return 1

    from agents import cuota_tae

    row = smoke_agents.load_problems()[0]
    result = smoke_agents.run_one(cuota_tae.build(BrokenLLM()), row, max_steps=6)
    assert not result["correct"]
    assert "the service is down" in result["error"]


# -- the summary ----------------------------------------------------------------------


def test_summarise_counts_what_the_report_needs():
    rows = [
        {"correct": True, "n_steps": 2, "tokens": 100, "latency_s": 1.0, "tools": ["a"]},
        {"correct": False, "n_steps": 4, "tokens": 300, "latency_s": 3.0, "tools": []},
    ]
    stats = smoke_agents.summarise(rows)
    assert stats == {
        "n": 2,
        "accuracy": 0.5,
        "mean_steps": 3.0,
        "mean_tokens": 200.0,
        "mean_latency_s": 2.0,
        "no_tool_call": 0.5,
    }


def test_summarise_of_nothing_does_not_divide_by_zero():
    assert smoke_agents.summarise([])["accuracy"] == 0


def test_report_covers_both_architectures_and_every_type(fake_llm, capsys):
    """One scripted run of the whole script, with a brain that always answers correctly."""

    class AlwaysRight:
        """Answers every problem with its expected value, without calling any tool."""

        model_id = "always-right"

        def __init__(self, answers):
            self.answers = list(answers)

        def generate(self, messages, max_new_tokens=None, tools=None):
            return call("final_answer", answer=self.answers.pop(0))

        def count_tokens(self, text):
            return max(1, len(text) // 4)

    problems = smoke_agents.load_problems()
    # Each problem is run twice: once by its specialist, once by the baseline.
    llm = AlwaysRight([p["answer"] for p in problems for _ in range(2)])
    result = smoke_agents.report(smoke_agents.run_all(problems, llm, max_steps=4))

    assert result["summary"]["specialists"]["n"] == 8
    assert result["summary"]["monolithic"]["n"] == 8
    assert result["summary"]["specialists"]["accuracy"] == 1.0
    assert set(result["by_type"]) == set(smoke_agents.SPECIALIST_OF_TYPE)
    assert "architecture" in capsys.readouterr().out


def test_the_trace_records_the_arguments_of_every_call(fake_llm):
    """A failed row has to be readable from the JSON: tool, arguments and observation."""
    from agents import cuota_tae

    row = [p for p in smoke_agents.load_problems() if p["tipo"] == "tae"][0]
    loan = row["_debug_prestamo"]
    llm = fake_llm(
        [
            call(
                "calculate_apr",
                principal=loan["principal"],
                annual_rate_pct=loan["tipo_nominal"],
                months=loan["meses"],
                upfront_fee=0,  # the ARCA mistake: a fee was given but 0 was passed
            ),
            call("final_answer", answer="wrong"),
        ]
    )
    result = smoke_agents.run_one(cuota_tae.build(llm), row, max_steps=6)
    assert not result["correct"]
    first = result["trace"][0]
    assert first["tool"] == "calculate_apr"
    assert first["arguments"]["upfront_fee"] == 0  # the cause is visible in the JSON
    assert loan["comision_apertura"] > 0  # the problem did mention a fee
    assert "result" in first["observation"]
