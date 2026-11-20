"""Smoke test of the specialists and the baseline against real problems and a real brain.

Eight problems from ``data/rlm_data_test.jsonl`` (two of each of the four types) go through
the specialist that owns them and through the monolithic ARCA baseline, with whichever brain
``ARCA_LLM_BACKEND`` selects. It prints accuracy, mean steps and mean tokens per
architecture and per problem type::

    uv run python -m eval.smoke_agents                       # local Qwen3-1.7B
    ARCA_LLM_BACKEND=endpoint uv run python -m eval.smoke_agents   # central service, in the DGX

This is NOT the benchmark of section 2 of docs/PLAN.md: eight problems say nothing about
accuracy. It answers a different question -- does the chain model -> tool call -> arguments
-> observation -> final answer work at all with a real model -- before spending a GPU on the
real evaluation. The routing is fixed here (taken from the problem's type), so a failure is
the agent's, never the router's.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agent_core.config import load_env
from agent_core.llm import build_llm
from agent_core.react_agent import ReActAgent
from agents import coste_ratio, cuota_tae, monolithic
from api.schemas import AgentResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "rlm_data_test.jsonl"
TOLERANCE = 0.01
PER_TYPE = 2
# Which specialist owns each type of problem. This is what the router will have to infer.
SPECIALIST_OF_TYPE = {
    "cuota": cuota_tae,
    "tae": cuota_tae,
    "coste_total": coste_ratio,
    "ratio_endeudamiento": coste_ratio,
}
# The dataset's tail is an instruction for the phase-1 model ("answer inside <answer></answer>"),
# not for an agent that finishes with final_answer.
ANSWER_FORMAT_TAIL = re.compile(r"\n\nDa la respuesta final.*", re.DOTALL)


def load_problems(dataset: Path = DEFAULT_DATASET, per_type: int = PER_TYPE) -> list[dict]:
    """The first ``per_type`` problems of each type, in the dataset's order (reproducible)."""
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chosen: list[dict] = []
    for kind in SPECIALIST_OF_TYPE:
        of_kind = [r for r in rows if r["tipo"] == kind][:per_type]
        if len(of_kind) < per_type:
            raise ValueError(f"{dataset} has fewer than {per_type} problems of type {kind!r}")
        chosen.extend(of_kind)
    return chosen


def task_of(row: dict) -> str:
    """The question as a user would ask it, without the dataset's output-format instruction."""
    return ANSWER_FORMAT_TAIL.sub("", row["question"]).strip()


NUMBER_TOKEN = re.compile(r"-?\d[\d.,]*")
# A Spanish thousands separator: every dot group has exactly three digits ("3.930,84",
# "16.848"). "174.09" does not match, so a plain decimal point is never mangled.
ES_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+$")


def numbers_in(text: str) -> list[float]:
    """Every number in a free-text answer, read in both Spanish and English notation.

    The model writes "3.930,84 euros" as readily as "3930.84", and reading the first one as
    3.930 and 84 was scoring correct answers as wrong (two of the eight problems on the 27B,
    2026-11-19). Each token is therefore tried in the notations that are unambiguous:
    a comma is always the decimal separator, and a dot is a thousands separator only when
    every group it separates has exactly three digits.
    """
    values: list[float] = []
    for token in NUMBER_TOKEN.findall(text.replace(" ", "")):
        token = token.rstrip(".,")
        if not token:
            continue
        if "," in token:
            candidates = [token.replace(".", "").replace(",", ".")]
        elif ES_THOUSANDS.match(token):
            candidates = [token.replace(".", "")]
        else:
            candidates = [token]
        for candidate in candidates:
            try:
                values.append(float(candidate))
            except ValueError:
                continue
    return values


def is_correct(expected: str, final_answer: str) -> bool:
    """True when the expected number appears in the answer, within the tolerance.

    The answer is free text, so any number in it may be the one asked for -- the same rule
    as the ``number`` check of ``eval.evaluate``, with the notations of ``numbers_in``.
    """
    return any(abs(value - float(expected)) <= TOLERANCE for value in numbers_in(final_answer))


def tools_used(response: AgentResponse) -> list[str]:
    return [step.action.name for step in response.steps if step.action]


def trace_of(response: AgentResponse, max_chars: int = 300) -> list[dict]:
    """The steps, flattened, so a failure can be analysed from the JSON alone.

    Without the arguments a wrong answer is unreadable afterwards: 'the APR came out as the
    nominal rate' and 'the fee was passed as 0' look identical in a summary.
    """
    return [
        {
            "tool": step.action.name if step.action else None,
            "arguments": step.action.arguments if step.action else None,
            "observation": step.observation[:max_chars],
        }
        for step in response.steps
    ]


def run_one(agent: ReActAgent, row: dict, max_steps: int) -> dict:
    """One problem through one agent. An exception is a failed row, not a crashed run."""
    started = time.perf_counter()
    try:
        response = agent.run(task_of(row), max_steps=max_steps)
        error = None
    except Exception as exc:  # noqa: BLE001 - one broken problem must not stop the smoke test
        return {
            "id": row["id"],
            "tipo": row["tipo"],
            "agent": agent.name,
            "expected": row["answer"],
            "answer": "",
            "correct": False,
            "n_steps": 0,
            "tokens": 0,
            "tools": [],
            "trace": [],
            "latency_s": round(time.perf_counter() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "id": row["id"],
        "tipo": row["tipo"],
        "agent": agent.name,
        "expected": row["answer"],
        "answer": response.final_answer,
        "correct": response.succeeded and is_correct(row["answer"], response.final_answer),
        "n_steps": response.n_steps,
        "tokens": response.tokens_used,
        "tools": tools_used(response),
        "trace": trace_of(response),
        "latency_s": round(time.perf_counter() - started, 2),
        "error": error,
    }


def summarise(rows: list[dict]) -> dict:
    """Accuracy, mean steps and mean tokens over a set of rows."""
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
        "mean_latency_s": sum(r["latency_s"] for r in rows) / n,
        "no_tool_call": sum(not r["tools"] for r in rows) / n,
    }


def run_all(problems: list[dict], llm, max_steps: int = 8) -> list[dict]:
    """Every problem through its specialist and through the baseline, sharing one brain."""
    specialists = {name: module.build(llm) for name, module in _modules_by_name().items()}
    baseline = monolithic.build(llm)
    rows = []
    for row in problems:
        specialist = specialists[SPECIALIST_OF_TYPE[row["tipo"]].NAME]
        for agent in (specialist, baseline):
            result = run_one(agent, row, max_steps)
            rows.append(result)
            flag = "ok " if result["correct"] else "FAIL"
            print(
                f"  {flag} {result['agent']:>11} · {result['id']:<22} "
                f"expected {result['expected']:>9} · got {result['answer'][:40]!r} "
                f"· {result['n_steps']} steps · {result['tokens']} tok"
            )
    return rows


def _modules_by_name() -> dict:
    return {module.NAME: module for module in set(SPECIALIST_OF_TYPE.values())}


def report(rows: list[dict]) -> dict:
    """The printed tables: per architecture and, for the specialists, per problem type."""
    architectures = {
        "specialists": [r for r in rows if r["agent"] != monolithic.NAME],
        "monolithic": [r for r in rows if r["agent"] == monolithic.NAME],
    }
    summary = {name: summarise(subset) for name, subset in architectures.items()}
    by_type = {
        kind: {
            name: summarise([r for r in subset if r["tipo"] == kind])
            for name, subset in architectures.items()
        }
        for kind in SPECIALIST_OF_TYPE
    }

    header = f"{'architecture':>13} {'acc':>6} {'steps':>7} {'tokens':>9}"
    print(f"\n{header} {'latency':>9} {'no_tool':>8}")
    for name, stats in summary.items():
        print(
            f"{name:>13} {stats['accuracy']:>6.2f} {stats['mean_steps']:>7.1f} "
            f"{stats['mean_tokens']:>9.0f} {stats['mean_latency_s']:>9.1f} "
            f"{stats['no_tool_call']:>8.2f}"
        )
    print(f"\n{'type':>20} {'specialists':>12} {'monolithic':>12}")
    for kind, stats in by_type.items():
        print(
            f"{kind:>20} {stats['specialists']['accuracy']:>12.2f} "
            f"{stats['monolithic']['accuracy']:>12.2f}"
        )
    return {"summary": summary, "by_type": by_type, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--per-type", type=int, default=PER_TYPE, help="problems of each of the four types"
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "reports" / "smoke_agents.json",
        help="where to write the full result (one JSON per run, as EXPERIMENTS.md expects)",
    )
    args = parser.parse_args()

    load_env()
    llm = build_llm()
    problems = load_problems(args.dataset, args.per_type)
    print(
        f"brain: {type(llm).__name__} · {llm.model_id} · thinking={llm.enable_thinking}\n"
        f"{len(problems)} problems × 2 architectures\n"
    )
    result = report(run_all(problems, llm, args.max_steps))
    result["config"] = {
        "backend": type(llm).__name__,
        "model": llm.model_id,
        "enable_thinking": llm.enable_thinking,
        "max_steps": args.max_steps,
        "per_type": args.per_type,
        "dataset": str(args.dataset),
    }
    result["total_usage"] = getattr(llm, "total_usage", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
