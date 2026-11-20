"""The full numeric benchmark: the 90 problems of the test split, by architecture.

Bank 1 of section 2 of ``docs/PLAN.md``, and the figure the report compares against phase 1
of ARCA. Every problem of ``data/rlm_data_test.jsonl`` goes through the specialist that owns
its type and through the monolithic ARCA baseline, with whichever brain
``ARCA_LLM_BACKEND`` selects::

    uv run python -m eval.numeric_bench --out reports/numeric_hf_t0.json
    ARCA_LLM_BACKEND=endpoint uv run python -m eval.numeric_bench --out reports/numeric_ep_t0.json

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


def load_all_problems(dataset: Path = DEFAULT_DATASET, limit: int | None = None) -> list[dict]:
    """Every problem of the dataset, in its own order, so two runs are comparable.

    Args:
        dataset: The JSONL to read.
        limit: Keep only the first N problems. For smoke-testing this script itself, not
            for the real run.
    """
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unknown = {r["tipo"] for r in rows} - set(SPECIALIST_OF_TYPE)
    if unknown:
        raise ValueError(f"{dataset} has types with no specialist: {sorted(unknown)}")
    return rows[:limit] if limit else rows


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


def run_all_incremental(
    problems: list[dict],
    llm,
    rows_path: Path,
    max_steps: int = 8,
    done: dict[str, dict] | None = None,
) -> list[dict]:
    """Every problem through its specialist and the baseline, appending as it goes.

    The brain is shared by both architectures, exactly as the orchestrator will share it.
    """
    done = dict(done or {})
    specialists = {module.NAME: module.build(llm) for module in set(SPECIALIST_OF_TYPE.values())}
    baseline = monolithic.build(llm)
    results: list[dict] = []
    total = 2 * len(problems)
    started = time.perf_counter()

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a", encoding="utf-8") as sidecar:
        for number, problem in enumerate(problems, 1):
            specialist = specialists[SPECIALIST_OF_TYPE[problem["tipo"]].NAME]
            for agent in (specialist, baseline):
                key = row_key(problem["id"], agent.name)
                if key in done:
                    results.append(done[key])
                    continue
                result = run_one(agent, problem, max_steps)
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


def print_by_type(counts: dict) -> None:
    """The per-type table, as hits/attempts, for the report's figure."""
    kinds = [*SPECIALIST_OF_TYPE, "total"]
    print(f"\n{'type':>20} {'specialists':>14} {'monolithic':>14}")
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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N problems (to test this script)"
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

    rows_path = args.rows or args.out.with_suffix(".rows.jsonl")
    if args.restart and rows_path.exists():
        rows_path.unlink()

    load_env()
    llm = build_llm()
    problems = load_all_problems(args.dataset, args.limit)
    done = load_done(rows_path)
    print(
        f"brain: {type(llm).__name__} · {llm.model_id} · thinking={llm.enable_thinking}\n"
        f"{len(problems)} problems × 2 architectures = {2 * len(problems)} runs\n"
        f"resuming with {len(done)} already done · sidecar {rows_path}\n",
        flush=True,
    )

    rows = run_all_incremental(problems, llm, rows_path, args.max_steps, done)
    result = report(rows)
    counts = by_type_counts(rows)
    print_by_type(counts)
    result["by_type_counts"] = counts
    result["config"] = {
        "backend": type(llm).__name__,
        "model": llm.model_id,
        "enable_thinking": llm.enable_thinking,
        "max_steps": args.max_steps,
        "dataset": str(args.dataset),
        "n_problems": len(problems),
    }
    result["total_usage"] = getattr(llm, "total_usage", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
