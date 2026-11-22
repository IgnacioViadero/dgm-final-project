"""The numeric benchmark as a bar chart per problem type, in the format of ARCA's figure 2.

What the figure has to let the reader do is compare, on the same 90 problems and the same
four types, the phase-1 models of ARCA (a fine-tuned 1.7B answering straight from the
prompt) against this project's agents (the same maths, but called as a tool). That is the
whole argument of the report, so the ARCA columns are drawn as a reference next to ours and
the denominators of each type are written in the axis labels::

    uv run --extra viz python -m eval.plot_numeric            # every reports/numeric_*.json
    uv run --extra viz python -m eval.plot_numeric --out reports/numeric_bench.png

The ARCA numbers are the measured results of phase 1 (see ``ARCA_PHASE1``), typed in by
hand because they come from another repository. Everything else is read from the JSON the
benchmark wrote, so the figure can never disagree with the table.

A run over several splits (``--dataset`` twice: the 600) carries ``by_split_counts``, and
``--split`` chooses which of them the bars show (``test``, ``train`` or ``all``). The ARCA
columns are always the 90 of test, because that is the only set phase 1 was measured on: when
the bars are the 600 or the 510, the legend says so, so nobody reads the grey bars as if they
had been measured on the same problems.

``--extra viz`` installs matplotlib; it is not a base dependency, so importing this module
is the only thing in the project that needs it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"

# The four types, in the order of ARCA's figure, plus the total.
TYPES = ["coste_total", "cuota", "ratio_endeudamiento", "tae"]
LABELS = {
    "coste_total": "coste total",
    "cuota": "cuota",
    "ratio_endeudamiento": "ratio",
    "tae": "TAE",
    "total": "TOTAL",
}
# Phase 1 of ARCA, measured on these same 90 problems (hits per type, same denominators).
# The model answered from the prompt, with no tools: this is what the tools have to beat.
ARCA_PHASE1 = {
    "base": {"coste_total": 4, "cuota": 0, "ratio_endeudamiento": 0, "tae": 0, "total": 4},
    "SFT": {"coste_total": 13, "cuota": 0, "ratio_endeudamiento": 0, "tae": 0, "total": 13},
    "GRPO": {"coste_total": 16, "cuota": 0, "ratio_endeudamiento": 7, "tae": 0, "total": 23},
}
# The denominators of the 90 of test, which are ARCA's. Any other set reads its own out of
# the benchmark's JSON, so a figure can never divide by a number the run did not have.
ARCA_DENOMINATORS = {
    "coste_total": 19,
    "cuota": 23,
    "ratio_endeudamiento": 27,
    "tae": 21,
    "total": 90,
}
DENOMINATORS = ARCA_DENOMINATORS

# Greys for the reference (ARCA, no tools), colour for this project (tools).
ARCA_COLOURS = {"base": "#c9c9c9", "SFT": "#9a9a9a", "GRPO": "#6b6b6b"}
OURS_COLOURS = ["#1f6fb4", "#49a0dc", "#0d3b60", "#7fc3f0", "#06243c"]


def config_label(config: dict) -> str:
    """A short name for one run: the brain and whether it was thinking."""
    model = str(config.get("model", "?"))
    brain = "1.7B" if "1.7" in model else ("27B" if "27" in model else model)
    return f"{brain} think={'1' if config.get('enable_thinking') else '0'}"


def load_runs(paths: list[Path], split: str = "all") -> list[dict]:
    """Read the benchmark's JSONs, keeping only the finished ones, ordered by brain.

    ``split`` picks a subset of a multi-split run (``by_split_counts``); a run of a single
    split has only ``by_type_counts`` and is taken as it is.
    """
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "by_type_counts" not in data:
            continue  # a smoke JSON, or a run that died before writing its summary
        by_split = data.get("by_split_counts") or {}
        counts = by_split.get(split, data["by_type_counts"]) if by_split else data["by_type_counts"]
        runs.append(
            {
                "path": path,
                "label": config_label(data.get("config", {})),
                "counts": counts,
                "config": data.get("config", {}),
            }
        )
    return sorted(runs, key=lambda r: ("27B" in r["label"], r["label"]))


def series_of(runs: list[dict], architecture: str) -> list[tuple[str, dict]]:
    """One series per run for a given architecture, as hits per type.

    A run with no rows for this architecture (the 600 were run as one process per
    architecture, so each JSON holds only its own) is left out instead of drawn as zeros.
    """
    series = []
    for run in runs:
        counts = run["counts"].get(architecture)
        if not counts or not counts["total"]["n"]:
            continue
        series.append(
            (
                f"{run['label']} · {architecture[:4]}",
                {kind: counts[kind]["hits"] for kind in [*TYPES, "total"]},
            )
        )
    return series


def denominators_of(runs: list[dict], architecture: str) -> dict:
    """The attempts per type of these runs, which is what the bars divide by.

    Taken from the data and not hard-coded, so the same figure works for the 90 of test, the
    510 of train and the 600 together. Runs with different denominators cannot share a
    figure: that would be comparing percentages of different problem sets.
    """
    found = {}
    for run in runs:
        counts = run["counts"].get(architecture)
        if not counts or not counts["total"]["n"]:
            continue
        theirs = {kind: counts[kind]["n"] for kind in [*TYPES, "total"]}
        if found and theirs != found:
            raise SystemExit(
                f"these runs do not share denominators for {architecture}: "
                f"{found} vs {theirs} ({run['path']})"
            )
        found = theirs
    return found or dict(ARCA_DENOMINATORS)


def plot(
    runs: list[dict], out: Path, architecture: str = "specialists", split: str = "all"
) -> Path:
    """Draw the figure: ARCA's phase 1 in grey, this project's runs in colour.

    Each group of bars is divided by its own denominators: ARCA's are the 90 of test (the
    only set it was measured on) and ours are whatever the run answered. That is why the
    legend names the problem count of each side.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on the DGX
    import matplotlib.pyplot as plt

    kinds = [*TYPES, "total"]
    ours = series_of(runs, architecture)
    if not ours:
        raise SystemExit(f"no run has rows for {architecture}")
    mine = denominators_of(runs, architecture)
    arca_suffix = "" if mine == ARCA_DENOMINATORS else " (90 de test)"
    groups = [
        (f"ARCA fase 1 · {name}{arca_suffix}", ARCA_PHASE1[name], ARCA_COLOURS[name],
         ARCA_DENOMINATORS)
        for name in ("base", "SFT", "GRPO")
    ]
    groups += [
        (name, hits, OURS_COLOURS[i % len(OURS_COLOURS)], mine)
        for i, (name, hits) in enumerate(ours)
    ]

    width = 0.8 / max(len(groups), 1)
    figure, axes = plt.subplots(figsize=(1.9 * len(kinds) + 3, 5.2))
    for index, (name, hits, colour, denominators) in enumerate(groups):
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(kinds))]
        values = [
            100 * hits[kind] / denominators[kind] if denominators[kind] else 0.0
            for kind in kinds
        ]
        bars = axes.bar(offsets, values, width, label=name, color=colour, edgecolor="white")
        for bar, kind in zip(bars, kinds, strict=True):
            axes.annotate(
                f"{hits[kind]}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=7,
            )

    axes.set_xticks(range(len(kinds)))
    axes.set_xticklabels(
        [
            f"{LABELS[k]}\n(n={mine[k]})"
            + ("" if mine == ARCA_DENOMINATORS else f"\nARCA n={ARCA_DENOMINATORS[k]}")
            for k in kinds
        ]
    )
    axes.set_ylabel("aciertos (%)")
    axes.set_ylim(0, 105)
    where = {
        "test": "90 problemas de data/rlm_data_test.jsonl",
        "train": "510 problemas de data/rlm_data_train.jsonl",
        "all": "600 problemas (90 de test + 510 de train)",
    }.get(split, split)
    axes.set_title(
        f"Exactitud numérica por tipo · {architecture} con tools vs ARCA fase 1 sin tools\n"
        f"{where} · la cifra sobre la barra son aciertos",
        pad=34,
    )
    # Outside the axes: the bars of 'coste total' reach the top and the legend hid them.
    axes.legend(
        ncols=min(len(groups), 6),
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    axes.grid(axis="y", alpha=0.3, linestyle=":")
    axes.set_axisbelow(True)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160)
    plt.close(figure)
    return out


def plot_from_stats(stats: dict, out: Path, splits: tuple[str, ...] = ("test", "all")) -> Path:
    """The 600-problem figure: one panel per split, both architectures in each.

    The input is what ``eval.stats_numeric`` wrote, because the 600 were run as one process
    per architecture and only that JSON has the two of them together. One panel per split so
    that the comparison the report needs is the one the eye makes: the 90 of test (where
    ARCA's phase 1 was measured, drawn in grey) and the 600 of the whole dataset. ARCA is
    only in the panel of its own problems; putting it next to the 600 would be comparing
    percentages of different problem sets.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on the DGX
    import matplotlib.pyplot as plt

    kinds = [*TYPES, "total"]
    counts = stats["counts"]
    panels = [s for s in splits if counts.get(s, {}).get("specialists", {}).get("total", {}).get("n")]
    if not panels:
        raise SystemExit(f"the stats JSON has no rows for {splits}")

    figure, axes_list = plt.subplots(
        len(panels), 1, figsize=(1.9 * len(kinds) + 3, 5.0 * len(panels)), squeeze=False
    )
    titles = {
        "test": "los 90 de data/rlm_data_test.jsonl (los mismos que midió ARCA)",
        "train": "los 510 de data/rlm_data_train.jsonl (nunca vistos)",
        "all": "los 600 del dataset completo (test + train)",
    }
    for axes, split in zip([a[0] for a in axes_list], panels, strict=True):
        mine = {
            kind: counts[split]["specialists"][kind]["n"] for kind in kinds
        }
        groups = []
        if split == "test":
            groups += [
                (f"ARCA fase 1 · {name}", ARCA_PHASE1[name], ARCA_COLOURS[name],
                 ARCA_DENOMINATORS)
                for name in ("base", "SFT", "GRPO")
            ]
        for index, architecture in enumerate(("specialists", "monolithic")):
            hits = {kind: counts[split][architecture][kind]["hits"] for kind in kinds}
            denominators = {kind: counts[split][architecture][kind]["n"] for kind in kinds}
            groups.append(
                (
                    f"1.7B think=0 · {architecture}",
                    hits,
                    OURS_COLOURS[index],
                    denominators,
                )
            )

        width = 0.8 / max(len(groups), 1)
        for index, (name, hits, colour, denominators) in enumerate(groups):
            offsets = [i + index * width - 0.4 + width / 2 for i in range(len(kinds))]
            values = [
                100 * hits[kind] / denominators[kind] if denominators.get(kind) else 0.0
                for kind in kinds
            ]
            bars = axes.bar(offsets, values, width, label=name, color=colour, edgecolor="white")
            for bar, kind in zip(bars, kinds, strict=True):
                axes.annotate(
                    f"{hits[kind]}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    fontsize=7,
                )
        axes.set_xticks(range(len(kinds)))
        axes.set_xticklabels(
            [
                f"{LABELS[k]}\n(n={mine[k]})"
                + (f"\nARCA n={ARCA_DENOMINATORS[k]}" if split == "test"
                   and mine[k] != ARCA_DENOMINATORS[k] else "")
                for k in kinds
            ]
        )
        axes.set_ylabel("aciertos (%)")
        axes.set_ylim(0, 105)
        axes.set_title(
            f"Exactitud numérica por tipo · {titles.get(split, split)}\n"
            "la cifra sobre la barra son aciertos",
            pad=34,
        )
        axes.legend(
            ncols=min(len(groups), 5),
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )
        axes.grid(axis="y", alpha=0.3, linestyle=":")
        axes.set_axisbelow(True)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160)
    plt.close(figure)
    return out


def table(runs: list[dict]) -> str:
    """The same numbers as text, so the report and the figure cannot drift apart."""
    kinds = [*TYPES, "total"]
    lines = []
    header = f"{'configuracion':<26}" + "".join(f"{LABELS[k]:>14}" for k in kinds)
    lines.append(header)
    lines.append("-" * len(header))
    for name in ("base", "SFT", "GRPO"):
        hits = ARCA_PHASE1[name]
        row = f"{'ARCA fase 1 · ' + name:<26}"
        row += "".join(f"{hits[k]:>7}/{ARCA_DENOMINATORS[k]:<6}" for k in kinds)
        lines.append(row)
    lines.append("-" * len(header))
    for architecture in ("specialists", "monolithic"):
        denominators = denominators_of(runs, architecture)
        for label, hits in series_of(runs, architecture):
            row = f"{label:<26}"
            row += "".join(f"{hits[k]:>7}/{denominators[k]:<6}" for k in kinds)
            lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "jsons",
        nargs="*",
        type=Path,
        help="the benchmark's JSONs (default: reports/numeric_*.json)",
    )
    parser.add_argument("--out", type=Path, default=REPORTS / "numeric_bench.png")
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="a JSON written by eval.stats_numeric: draws one panel per split with both "
        "architectures together (the figure of the 600-problem run)",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="which split of a multi-split run to draw: test, train or all (default: all)",
    )
    args = parser.parse_args()

    if args.stats:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        print(f"written to {plot_from_stats(stats, args.out)}")
        return

    paths = args.jsons or sorted(REPORTS.glob("numeric_*.json"))
    runs = load_runs(paths, args.split)
    if not runs:
        raise SystemExit(f"no finished benchmark JSON in {[str(p) for p in paths]}")
    print(f"{len(runs)} runs: {[r['label'] for r in runs]}\n")
    print(table(runs))
    for architecture in ("specialists", "monolithic"):
        if not series_of(runs, architecture):
            print(f"\nno run has rows for {architecture}: no figure")
            continue
        out = args.out if architecture == "specialists" else args.out.with_stem(
            args.out.stem + "_monolithic"
        )
        print(f"\nwritten to {plot(runs, out, architecture, args.split)}")


if __name__ == "__main__":
    main()
