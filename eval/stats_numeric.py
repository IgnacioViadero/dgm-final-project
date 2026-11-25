"""What the 600-problem benchmark means: confidence intervals, McNemar and failure mechanisms.

The benchmark writes rows; this module turns them into the numbers the report quotes, and
nothing else. It is deliberately separate from ``eval.numeric_bench`` because it needs no
GPU: it reads the sidecars (which the two parallel runs append to as they go) and can
therefore be run while a run is still going, or afterwards on a run that died half way::

    uv run python -m eval.stats_numeric \\
        reports/numeric_hf_t0_600_specialists.rows.jsonl \\
        reports/numeric_hf_t0_600_monolithic.rows.jsonl \\
        --out reports/numeric_hf_t0_600_stats.json

Three things come out of it:

* **The table**, hits/attempts and percentage per type and architecture, split into test
  (the 90 that are comparable with ARCA's phase 1 and with our earlier runs), train (the
  510 nobody had looked at) and the 600 together.
* **A 95 % Wilson interval** on each architecture's total accuracy, and **McNemar's exact
  test** between the two architectures over the problems both of them answered. Wilson
  rather than the normal approximation because at 99 % accuracy the normal interval goes
  over 1; McNemar rather than a two-sample test because the two architectures see the same
  problems, and exact rather than the chi-square because the discordant pairs are a handful.
* **Every failure, grouped by mechanism** (``classify_failure``), with one trace per group.
  The mechanism is what a fix has to attack, and reading 20 traces by hand to find out they
  are all the same two bugs is the part of the analysis a script can do.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TYPES = ["coste_total", "cuota", "ratio_endeudamiento", "tae"]
LABELS = {
    "coste_total": "coste total",
    "cuota": "cuota",
    "ratio_endeudamiento": "ratio",
    "tae": "TAE",
    "total": "TOTAL",
}
SPLITS = ["test", "train", "all"]
TOLERANCE = 0.01
Z95 = 1.959963984540054


# -- reading the rows -------------------------------------------------------------------


def load_rows(paths: list[Path]) -> list[dict]:
    """Every row of every sidecar or benchmark JSON, de-duplicated by (id, agent).

    A sidecar may hold the same pair twice if a run was resumed after being killed between
    the append and the flush of the same row; the last one wins, as the file order says.
    """
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            found = json.loads(text).get("rows", [])
        else:
            found = [json.loads(line) for line in text.splitlines() if line.strip()]
        for row in found:
            rows[(row["id"], row["agent"])] = row
    return list(rows.values())


def architecture_of(row: dict) -> str:
    return "monolithic" if row["agent"] == "monolithic" else "specialists"


def split_of(row: dict) -> str:
    """The split a row came from, falling back to the id when an old row has no field."""
    return row.get("split") or "test"


# -- the table --------------------------------------------------------------------------


def counts_table(rows: list[dict]) -> dict:
    """hits/attempts per split, architecture and type, with 'total' and 'all' included."""
    table: dict = {}
    for split in SPLITS:
        subset = rows if split == "all" else [r for r in rows if split_of(r) == split]
        table[split] = {}
        for architecture in ("specialists", "monolithic"):
            of_arch = [r for r in subset if architecture_of(r) == architecture]
            per_type = {
                kind: {
                    "hits": sum(r["correct"] for r in of_arch if r["tipo"] == kind),
                    "n": sum(r["tipo"] == kind for r in of_arch),
                }
                for kind in TYPES
            }
            per_type["total"] = {
                "hits": sum(r["correct"] for r in of_arch),
                "n": len(of_arch),
            }
            table[split][architecture] = per_type
    return table


def format_table(table: dict) -> str:
    """The table as text, with the denominators visible (the report quotes them)."""
    kinds = [*TYPES, "total"]
    lines = []
    width = 16 + 17 * len(kinds)
    for split in SPLITS:
        n = table[split]["specialists"]["total"]["n"] + table[split]["monolithic"]["total"]["n"]
        lines.append("")
        lines.append(f"{split.upper()}  ({n} corridas)")
        lines.append(f"{'arquitectura':<16}" + "".join(f"{LABELS[k]:>17}" for k in kinds))
        lines.append("-" * width)
        for architecture in ("specialists", "monolithic"):
            cells = []
            for kind in kinds:
                cell = table[split][architecture][kind]
                share = cell["hits"] / cell["n"] if cell["n"] else 0.0
                cells.append(f"{cell['hits']:>4}/{cell['n']:<4}{share:>6.1%}")
            lines.append(f"{architecture:<16}" + "".join(f"{c:>17}" for c in cells))
    return "\n".join(lines)


# -- the statistics ---------------------------------------------------------------------


def wilson_interval(hits: int, n: int, z: float = Z95) -> dict:
    """The 95 % Wilson score interval of a proportion.

    Preferred over the normal approximation because this benchmark lives near 100 %: with
    599/600 the normal interval reaches past 1, and with 600/600 it has zero width, which
    would claim certainty the data does not support.
    """
    if n == 0:
        return {"hits": 0, "n": 0, "point": 0.0, "low": 0.0, "high": 1.0}
    phat = hits / n
    denominator = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denominator
    return {
        "hits": hits,
        "n": n,
        "point": phat,
        "low": max(0.0, centre - half),
        "high": min(1.0, centre + half),
    }


def binomial_two_sided(successes: int, n: int) -> float:
    """Two-sided exact binomial p-value against p=0.5 (the one McNemar's test needs)."""
    if n == 0:
        return 1.0
    k = min(successes, n - successes)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar(rows: list[dict], split: str = "all") -> dict:
    """McNemar's exact test between the two architectures on the problems both answered.

    Only paired problems count: one architecture having run further than the other (the two
    processes are independent) must not turn into a difference in accuracy.
    """
    by_id: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if split != "all" and split_of(row) != split:
            continue
        by_id[row["id"]][architecture_of(row)] = bool(row["correct"])
    paired = {
        pid: outcomes
        for pid, outcomes in by_id.items()
        if {"specialists", "monolithic"} <= set(outcomes)
    }
    both = sum(o["specialists"] and o["monolithic"] for o in paired.values())
    only_specialists = sum(o["specialists"] and not o["monolithic"] for o in paired.values())
    only_monolithic = sum(o["monolithic"] and not o["specialists"] for o in paired.values())
    neither = sum(not o["specialists"] and not o["monolithic"] for o in paired.values())
    discordant = only_specialists + only_monolithic
    return {
        "split": split,
        "n_paired": len(paired),
        "both_correct": both,
        "only_specialists": only_specialists,
        "only_monolithic": only_monolithic,
        "neither": neither,
        "discordant": discordant,
        "p_value": binomial_two_sided(only_specialists, discordant),
        "test": "McNemar exacto (binomial de dos colas sobre los pares discordantes)",
    }


# -- the failures -----------------------------------------------------------------------

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_of(value) -> list[float]:
    """Every number inside a JSON-ish value (arguments dict, observation string, ...)."""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in numbers_of(v)]
    if isinstance(value, list):
        return [n for v in value for n in numbers_of(v)]
    return [float(m.group()) for m in NUMBER.finditer(str(value))]


def close(a: float, b: float, tolerance: float = TOLERANCE) -> bool:
    return abs(a - b) <= tolerance


def answered_numbers(row: dict) -> list[float]:
    """The numbers the agent put in its final answer."""
    from eval.smoke_agents import numbers_in

    return numbers_in(row.get("answer") or "")


def classify_failure(row: dict) -> str:
    """The mechanism behind one wrong answer, as a short Spanish label.

    The order matters: the first thing that explains the failure is the one reported, from
    the most mechanical (it crashed, it never called a tool) to the most semantic (the tool
    gave the right number and the model sent another one).
    """
    if row.get("error"):
        return "excepcion"
    tools = row.get("tools") or []
    if not tools:
        return "no llama a ninguna tool"
    trace = row.get("trace") or []
    expected = float(row["expected"])
    calls = [step for step in trace if step.get("tool") and step["tool"] != "final_answer"]
    if "final_answer" not in tools:
        return "se agota sin contestar (sin final_answer)"
    if not calls:
        return "contesta sin llamar a ninguna tool de calculo"

    observed = [n for step in calls for n in numbers_of(step.get("observation"))]
    arguments = [n for step in calls for n in numbers_of(step.get("arguments"))]
    answered = answered_numbers(row)
    right_number_available = any(close(n, expected) for n in observed)

    if not right_number_available:
        # The tool was called, but with something other than the problem's data, or the
        # wrong tool for the question: the right number was never on the screen.
        failed = [step for step in calls if "error" in str(step.get("observation", "")).lower()]
        if failed:
            return "la tool devuelve error (argumentos invalidos)"
        # A field of the observation came back empty. On the baseline this is the optional
        # argument of loan_metrics (monthly_income) not being sent: the tool cannot refuse,
        # it answers null, and the model has nothing to report. It is the exact failure the
        # mandatory-argument design of the specialists exists to prevent, so it gets a
        # label of its own instead of being filed under "wrong argument".
        if any(": null" in str(step.get("observation", "")) for step in calls):
            return "omite un argumento opcional y la tool devuelve null"
        return "argumento mal extraido o tool equivocada (la observacion nunca trae el numero)"
    if not answered:
        return "contesta sin ningun numero"
    if any(close(a, g) for a in answered for g in arguments):
        return "contesta con un argumento/input en vez de con el result"
    if any(close(a, o) for a in answered for o in observed):
        return "contesta con otro campo de la observacion"
    return "cifra inventada (no esta en la traza)"


def failure_report(rows: list[dict]) -> dict:
    """Every failure grouped by mechanism, with counts and one example trace per group."""
    report: dict = {}
    for architecture in ("specialists", "monolithic"):
        failures = [
            r for r in rows if architecture_of(r) == architecture and not r["correct"]
        ]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in failures:
            groups[classify_failure(row)].append(row)
        report[architecture] = {
            "n_failures": len(failures),
            "groups": [
                {
                    "mechanism": mechanism,
                    "n": len(members),
                    "ids": [m["id"] for m in members],
                    "splits": sorted({split_of(m) for m in members}),
                    "types": sorted({m["tipo"] for m in members}),
                    "example": {
                        "id": members[0]["id"],
                        "tipo": members[0]["tipo"],
                        "split": split_of(members[0]),
                        "expected": members[0]["expected"],
                        "answer": members[0]["answer"],
                        "n_steps": members[0].get("n_steps"),
                        "trace": members[0].get("trace"),
                        "error": members[0].get("error"),
                    },
                }
                for mechanism, members in sorted(
                    groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
                )
            ],
        }
    return report


def format_failures(report: dict) -> str:
    lines = []
    for architecture, data in report.items():
        lines.append("")
        lines.append(f"{architecture}: {data['n_failures']} fallos")
        for group in data["groups"]:
            lines.append(
                f"  {group['n']:>3}  {group['mechanism']}  "
                f"[{', '.join(group['types'])}] {group['ids'][:6]}"
                + (" ..." if len(group["ids"]) > 6 else "")
            )
            example = group["example"]
            lines.append(
                f"       ej. {example['id']} esperado {example['expected']} · "
                f"contesto {str(example['answer'])[:70]!r}"
            )
            for step in example["trace"] or []:
                lines.append(
                    f"         {step.get('tool')}({json.dumps(step.get('arguments'), ensure_ascii=False)})"
                )
                lines.append(f"           -> {str(step.get('observation'))[:160]}")
    return "\n".join(lines)


# -- the whole thing --------------------------------------------------------------------


def analyse(rows: list[dict]) -> dict:
    table = counts_table(rows)
    return {
        "n_rows": len(rows),
        "counts": table,
        "wilson": {
            split: {
                architecture: wilson_interval(
                    table[split][architecture]["total"]["hits"],
                    table[split][architecture]["total"]["n"],
                )
                for architecture in ("specialists", "monolithic")
            }
            for split in SPLITS
        },
        "mcnemar": {split: mcnemar(rows, split) for split in SPLITS},
        "failures": failure_report(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rows", nargs="+", type=Path, help="the sidecars or JSONs to read")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    result = analyse(rows)
    print(f"{result['n_rows']} corridas leidas de {len(args.rows)} ficheros")
    print(format_table(result["counts"]))
    print("\nIntervalo de confianza del 95 % (Wilson) del acierto total")
    for split in SPLITS:
        for architecture in ("specialists", "monolithic"):
            w = result["wilson"][split][architecture]
            print(
                f"  {split:<6} {architecture:<12} {w['hits']:>4}/{w['n']:<4} "
                f"{w['point']:>7.2%}  [{w['low']:.2%}, {w['high']:.2%}]"
            )
    print("\nMcNemar (especialistas vs monolitico, mismos problemas)")
    for split in SPLITS:
        m = result["mcnemar"][split]
        print(
            f"  {split:<6} pares={m['n_paired']:<4} solo_esp={m['only_specialists']} "
            f"solo_mono={m['only_monolithic']} discordantes={m['discordant']} "
            f"p={m['p_value']:.4f}"
        )
    print("\nFallos por mecanismo")
    print(format_failures(result["failures"]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
