"""The full numeric benchmark: every problem of one or more splits, by architecture.

Bank 1 of section 2 of ``docs/PLAN.md``, and the figure the report compares against phase 1
of ARCA. Every problem of the datasets given with ``--dataset`` goes through the specialist
that owns its type and through the monolithic ARCA baseline, with whichever brain
``ARCA_LLM_BACKEND`` selects::

    uv run python -m eval.numeric_bench --out reports/numeric_hf_t0.json
    ARCA_LLM_BACKEND=endpoint uv run python -m eval.numeric_bench --out reports/numeric_ep_t0.json

``--dataset`` may be repeated, which is how the 600-problem run is made (the 90 of test plus
the 510 of train). Every row records the file it came from (``dataset``) and its short name
(``split``: "test", "train"), so the results of each split can be reported apart without a
second run. The ids are unique across the two files, which is what lets the sidecar resume a
mixed run::

    uv run python -m eval.numeric_bench \
        --dataset data/rlm_data_test.jsonl --dataset data/rlm_data_train.jsonl \
        --out reports/numeric_hf_t0_600.json

The problem selection, the routing by type, the correctness check and the per-run metrics
are ``eval.smoke_agents``': this module only scales it from 8 problems to all 90 and makes
the run survivable. The smoke test answers "does the chain work at all"; this one answers
"how often is it right", which is what the report needs.

**Nothing is lost if the run dies.** Each finished problem is appended to a JSONL sidecar
(``--rows``, by default the output path with a ``.rows.jsonl`` suffix) and flushed, so a run
killed at problem 70 keeps those 70. Re-running the same command resumes: the pairs
(problem, agent) already in the sidecar are skipped and only the rest are sent to the model.
``--restart`` throws the sidecar away instead. 180 runs of a local 1.7B are about two hours,
which is long enough that a lost session must not mean a lost afternoon.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent_core.config import load_env
from agent_core.llm import build_llm
from agents import monolithic
from eval.smoke_agents import (
    DEFAULT_DATASET,
    REPO_ROOT,
    SPECIALIST_OF_TYPE,
    report,
    run_one,
)


def split_name(dataset: Path) -> str:
    """The short name of a dataset file: "rlm_data_train.jsonl" -> "train"."""
    stem = dataset.stem
    return stem.split("rlm_data_", 1)[1] if "rlm_data_" in stem else stem


def load_all_problems(
    datasets: Path | list[Path] = DEFAULT_DATASET, limit: int | None = None
) -> list[dict]:
    """Every problem of every dataset, in file order, so two runs are comparable.

    Each problem is tagged with the file it came from, because the report gives test and
    train apart and a row has to say which one it is without looking anything up.

    Args:
        datasets: The JSONL to read, or several of them in the order given.
        limit: Keep only the first N problems *of each file*. For smoke-testing this
            script itself, not for the real run.

    Raises:
        ValueError: A problem has a type no specialist owns, or the same id appears in two
            files (the sidecar keys on the id, so duplicates would resume wrongly).
    """
    paths = [datasets] if isinstance(datasets, Path) else list(datasets)
    problems: list[dict] = []
    seen: dict[str, str] = {}
    for dataset in paths:
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        unknown = {r["tipo"] for r in rows} - set(SPECIALIST_OF_TYPE)
        if unknown:
            raise ValueError(f"{dataset} has types with no specialist: {sorted(unknown)}")
        for row in rows[:limit] if limit else rows:
            clash = seen.get(row["id"])
            if clash:
                raise ValueError(
                    f"id {row['id']!r} is in {clash} and in {dataset}: the ids must be "
                    "unique across the datasets of one run"
                )
            seen[row["id"]] = str(dataset)
            problems.append({**row, "dataset": str(dataset), "split": split_name(dataset)})
    return problems


def row_key(problem_id: str, agent_name: str) -> str:
    """What identifies one unit of work: one problem through one architecture."""
    return f"{problem_id}::{agent_name}"


def load_done(rows_path: Path) -> dict[str, dict]:
    """The results already in the sidecar, by key. A half-written last line is dropped."""
    if not rows_path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # the run was killed mid-write
            continue
        done[row_key(row["id"], row["agent"])] = row
    return done


ARCHITECTURES = ("specialists", "monolithic")


def run_all_incremental(
    problems: list[dict],
    llm,
    rows_path: Path,
    max_steps: int = 8,
    done: dict[str, dict] | None = None,
    architectures: tuple[str, ...] = ARCHITECTURES,
) -> list[dict]:
    """Every problem through its specialist and the baseline, appending as it goes.

    The brain is shared by both architectures, exactly as the orchestrator will share it.

    Args:
        architectures: Which of ``ARCHITECTURES`` to run. One at a time is how the 600 are
            run: two processes, each with its own brain, its own sidecar and its own JSON,
            so both fit on the GPU at once and a failure in one does not cost the other.
    """
    done = dict(done or {})
    unknown = set(architectures) - set(ARCHITECTURES)
    if unknown:
        raise ValueError(f"unknown architectures {sorted(unknown)}: use {list(ARCHITECTURES)}")
    specialists = {module.NAME: module.build(llm) for module in set(SPECIALIST_OF_TYPE.values())}
    baseline = monolithic.build(llm) if "monolithic" in architectures else None
    results: list[dict] = []
    total = len(architectures) * len(problems)
    started = time.perf_counter()

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a", encoding="utf-8") as sidecar:
        for number, problem in enumerate(problems, 1):
            specialist = specialists[SPECIALIST_OF_TYPE[problem["tipo"]].NAME]
            chosen = [specialist if name == "specialists" else baseline for name in architectures]
            for agent in chosen:
                key = row_key(problem["id"], agent.name)
                if key in done:
                    results.append(done[key])
                    continue
                result = run_one(agent, problem, max_steps)
                # Where the problem came from, so test and train can be reported apart.
                result["dataset"] = problem.get("dataset", "")
                result["split"] = problem.get("split", "")
                results.append(result)
                sidecar.write(json.dumps(result, ensure_ascii=False) + "\n")
                sidecar.flush()  # a killed run keeps everything up to here
                flag = "ok " if result["correct"] else "FAIL"
                elapsed = time.perf_counter() - started
                print(
                    f"[{len(results):>3}/{total}] {flag} {result['agent']:>11} · "
                    f"{result['id']:<24} expected {result['expected']:>9} · "
                    f"got {result['answer'][:34]!r} · {result['n_steps']} steps · "
                    f"{result['tokens']} tok · {elapsed / 60:.0f} min",
                    flush=True,
                )
            if number % 10 == 0:
                hits = sum(r["correct"] for r in results)
                print(
                    f"  -- {number}/{len(problems)} problems · {hits}/{len(results)} correct",
                    flush=True,
                )
    return results


def by_type_counts(rows: list[dict]) -> dict:
    """Hits out of attempts per type and architecture: the shape of ARCA's figure 2.

    The report puts this next to phase 1 of ARCA (4/90 base, 13/90 SFT, 23/90 GRPO), so the
    denominators have to be visible, not just a percentage.
    """
    architectures = {
        "specialists": [r for r in rows if r["agent"] != monolithic.NAME],
        "monolithic": [r for r in rows if r["agent"] == monolithic.NAME],
    }
    counts: dict[str, dict] = {}
    for name, subset in architectures.items():
        per_type = {}
        for kind in SPECIALIST_OF_TYPE:
            of_kind = [r for r in subset if r["tipo"] == kind]
            per_type[kind] = {
                "hits": sum(r["correct"] for r in of_kind),
                "n": len(of_kind),
            }
        per_type["total"] = {
            "hits": sum(r["correct"] for r in subset),
            "n": len(subset),
        }
        counts[name] = per_type
    return counts


def splits_of(rows: list[dict]) -> list[str]:
    """The splits present in the rows, test first and then the rest in alphabetical order."""
    names = {r.get("split") or "" for r in rows} - {""}
    return sorted(names, key=lambda name: (name != "test", name))


def by_split_counts(rows: list[dict]) -> dict:
    """``by_type_counts`` for each split and for everything together.

    The report has to give the 90 of test (the ones comparable with ARCA's phase 1 and with
    our earlier runs) apart from the 510 of train, and both apart from the 600.
    """
    counts = {name: by_type_counts([r for r in rows if r.get("split") == name])
              for name in splits_of(rows)}
    counts["all"] = by_type_counts(rows)
    return counts


def print_by_type(counts: dict, title: str = "") -> None:
    """The per-type table, as hits/attempts, for the report's figure."""
    kinds = [*SPECIALIST_OF_TYPE, "total"]
    print(f"\n{title or 'type':>20} {'specialists':>14} {'monolithic':>14}")
    for kind in kinds:
        cells = []
        for name in ("specialists", "monolithic"):
            cell = counts[name][kind]
            share = cell["hits"] / cell["n"] if cell["n"] else 0.0
            cells.append(f"{cell['hits']:>3}/{cell['n']:<3} {share:>5.0%}")
        print(f"{kind:>20} {cells[0]:>14} {cells[1]:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="a problems JSONL; repeat the flag to run several in one go "
        f"(default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--architectures",
        default="both",
        choices=[*ARCHITECTURES, "both"],
        help="which architecture to run (default: both, in one process)",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N problems of each dataset (to test this script)",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "reports" / "numeric_bench.json"
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="JSONL written as the run goes, and read back to resume "
        "(default: the --out path with .rows.jsonl)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore and overwrite the sidecar instead of resuming from it",
    )
    args = parser.parse_args()
    datasets = args.dataset or [DEFAULT_DATASET]
    architectures = ARCHITECTURES if args.architectures == "both" else (args.architectures,)

    rows_path = args.rows or args.out.with_suffix(".rows.jsonl")
    if args.restart and rows_path.exists():
        rows_path.unlink()

    load_env()
    llm = build_llm()
    problems = load_all_problems(datasets, args.limit)
    done = load_done(rows_path)
    per_split = {name: sum(p["split"] == name for p in problems) for name in
                 sorted({p["split"] for p in problems})}
    print(
        f"brain: {type(llm).__name__} · {llm.model_id} · thinking={llm.enable_thinking}\n"
        f"{len(problems)} problems ({per_split}) × {list(architectures)} = "
        f"{len(architectures) * len(problems)} runs\n"
        f"resuming with {len(done)} already done · sidecar {rows_path}\n",
        flush=True,
    )

    rows = run_all_incremental(
        problems, llm, rows_path, args.max_steps, done, architectures=architectures
    )
    result = report(rows)
    counts = by_type_counts(rows)
    by_split = by_split_counts(rows)
    for name, split_counts in by_split.items():
        print_by_type(split_counts, title=f"{name} (n={split_counts['specialists']['total']['n']})")
    result["by_type_counts"] = counts
    result["by_split_counts"] = by_split
    result["config"] = {
        "backend": type(llm).__name__,
        "model": llm.model_id,
        "enable_thinking": llm.enable_thinking,
        "max_steps": args.max_steps,
        "architectures": list(architectures),
        "dataset": str(datasets[0]),
        "datasets": [str(d) for d in datasets],
        "n_problems": len(problems),
        "n_problems_per_split": per_split,
    }
    result["total_usage"] = getattr(llm, "total_usage", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
