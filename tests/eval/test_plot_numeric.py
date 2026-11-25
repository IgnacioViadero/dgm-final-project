"""The figure's arithmetic, without drawing anything (matplotlib is an optional extra).

What can go silently wrong in a figure is the denominator: drawing hits out of 90 when the
run answered 600 problems, or putting ARCA's 90 next to our 600 as if they were the same
set. These tests pin that the denominators come from the data and that runs which do not
share them cannot end up in one figure.
"""

import json

import pytest

from eval import plot_numeric


def counts(architecture="specialists", hits=1, n=1):
    return {
        architecture: {
            kind: {"hits": hits, "n": n}
            for kind in [*plot_numeric.TYPES, "total"]
        }
    }


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_multi_split_run_is_read_by_split(tmp_path):
    path = write(
        tmp_path,
        "run.json",
        {
            "by_type_counts": counts(hits=600, n=600),
            "by_split_counts": {
                "test": counts(hits=90, n=90),
                "all": counts(hits=600, n=600),
            },
            "config": {"model": "Qwen/Qwen3-1.7B", "enable_thinking": False},
        },
    )
    assert plot_numeric.load_runs([path], "test")[0]["counts"]["specialists"]["total"]["n"] == 90
    assert plot_numeric.load_runs([path], "all")[0]["counts"]["specialists"]["total"]["n"] == 600


def test_a_single_split_run_is_taken_as_it_is(tmp_path):
    path = write(tmp_path, "run.json", {"by_type_counts": counts(hits=90, n=90), "config": {}})
    assert plot_numeric.load_runs([path], "all")[0]["counts"]["specialists"]["total"]["n"] == 90


def test_an_architecture_with_no_rows_is_left_out_not_drawn_as_zero(tmp_path):
    path = write(tmp_path, "run.json", {"by_type_counts": {**counts(), **counts("monolithic", 0, 0)}, "config": {}})
    runs = plot_numeric.load_runs([path])
    assert plot_numeric.series_of(runs, "specialists")
    assert plot_numeric.series_of(runs, "monolithic") == []


def test_the_denominators_come_from_the_run_not_from_arca(tmp_path):
    path = write(tmp_path, "run.json", {"by_type_counts": counts(hits=150, n=150), "config": {}})
    runs = plot_numeric.load_runs([path])
    assert plot_numeric.denominators_of(runs, "specialists")["total"] == 150


def test_runs_with_different_denominators_cannot_share_a_figure(tmp_path):
    first = write(tmp_path, "a.json", {"by_type_counts": counts(hits=90, n=90), "config": {}})
    second = write(tmp_path, "b.json", {"by_type_counts": counts(hits=600, n=600), "config": {}})
    runs = plot_numeric.load_runs([first, second])
    with pytest.raises(SystemExit, match="do not share denominators"):
        plot_numeric.denominators_of(runs, "specialists")
