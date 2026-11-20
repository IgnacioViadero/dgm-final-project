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
DENOMINATORS = {"coste_total": 19, "cuota": 23, "ratio_endeudamiento": 27, "tae": 21, "total": 90}

# Greys for the reference (ARCA, no tools), colour for this project (tools).
ARCA_COLOURS = {"base": "#c9c9c9", "SFT": "#9a9a9a", "GRPO": "#6b6b6b"}
OURS_COLOURS = ["#1f6fb4", "#49a0dc", "#0d3b60", "#7fc3f0", "#06243c"]


def config_label(config: dict) -> str:
    """A short name for one run: the brain and whether it was thinking."""
    model = str(config.get("model", "?"))
    brain = "1.7B" if "1.7" in model else ("27B" if "27" in model else model)
    return f"{brain} think={'1' if config.get('enable_thinking') else '0'}"


def load_runs(paths: list[Path]) -> list[dict]:
    """Read the benchmark's JSONs, keeping only the finished ones, ordered by brain."""
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "by_type_counts" not in data:
            continue  # a smoke JSON, or a run that died before writing its summary
        runs.append(
            {
                "path": path,
                "label": config_label(data.get("config", {})),
                "counts": data["by_type_counts"],
                "config": data.get("config", {}),
            }
        )
    return sorted(runs, key=lambda r: ("27B" in r["label"], r["label"]))


def series_of(runs: list[dict], architecture: str) -> list[tuple[str, dict]]:
    """One series per run for a given architecture, as hits per type."""
    series = []
    for run in runs:
        counts = run["counts"][architecture]
        series.append(
            (
                f"{run['label']} · {architecture[:4]}",
                {kind: counts[kind]["hits"] for kind in [*TYPES, "total"]},
            )
        )
    return series


def plot(runs: list[dict], out: Path, architecture: str = "specialists") -> Path:
    """Draw the figure: ARCA's phase 1 in grey, this project's runs in colour."""
    import matplotlib

    matplotlib.use("Agg")  # no display on the DGX
    import matplotlib.pyplot as plt

    kinds = [*TYPES, "total"]
    ours = series_of(runs, architecture)
    groups = [(name, ARCA_PHASE1[name], ARCA_COLOURS[name]) for name in ("base", "SFT", "GRPO")]
    groups += [
        (name, hits, OURS_COLOURS[i % len(OURS_COLOURS)]) for i, (name, hits) in enumerate(ours)
    ]

    width = 0.8 / max(len(groups), 1)
    figure, axes = plt.subplots(figsize=(1.9 * len(kinds) + 3, 5.2))
    for index, (name, hits, colour) in enumerate(groups):
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(kinds))]
        values = [100 * hits[kind] / DENOMINATORS[kind] for kind in kinds]
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
    axes.set_xticklabels([f"{LABELS[k]}\n(n={DENOMINATORS[k]})" for k in kinds])
    axes.set_ylabel("aciertos (%)")
    axes.set_ylim(0, 105)
    axes.set_title(
        f"Exactitud numérica por tipo · {architecture} con tools vs ARCA fase 1 sin tools\n"
        "90 problemas de data/rlm_data_test.jsonl · la cifra sobre la barra son aciertos"
    )
    axes.legend(ncols=2, fontsize=8, loc="upper left")
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
        row += "".join(f"{hits[k]:>7}/{DENOMINATORS[k]:<6}" for k in kinds)
        lines.append(row)
    lines.append("-" * len(header))
    for architecture in ("specialists", "monolithic"):
        for label, hits in series_of(runs, architecture):
            row = f"{label:<26}"
            row += "".join(f"{hits[k]:>7}/{DENOMINATORS[k]:<6}" for k in kinds)
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
    args = parser.parse_args()

    paths = args.jsons or sorted(REPORTS.glob("numeric_*.json"))
    runs = load_runs(paths)
    if not runs:
        raise SystemExit(f"no finished benchmark JSON in {[str(p) for p in paths]}")
    print(f"{len(runs)} runs: {[r['label'] for r in runs]}\n")
    print(table(runs))
    for architecture in ("specialists", "monolithic"):
        out = args.out if architecture == "specialists" else args.out.with_stem(
            args.out.stem + "_monolithic"
        )
        print(f"\nwritten to {plot(runs, out, architecture)}")


if __name__ == "__main__":
    main()
