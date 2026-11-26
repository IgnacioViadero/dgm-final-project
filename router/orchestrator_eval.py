"""Run the whole benchmark through the orchestrator and report routing and answer accuracy.

    uv run python -m scripts.eval_orchestrator --route-only
    uv run python -m scripts.eval_orchestrator --limit 20
    uv run python -m scripts.eval_orchestrator --bench bench.jsonl --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict

from router.orchestrator import build_orchestrator
from router.router import route


EXPECTED_AGENT = {
    "cuota": "cuota_tae",
    "tae": "cuota_tae",
    "coste_total": "coste_ratio",
    "ratio_endeudamiento": "coste_ratio",
    "rag": "rag",
    "fuera_corpus": "rag",
}
TEXT_TYPES = {"rag", "fuera_corpus"}
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_number(text: str) -> float | None:
    """The number inside <answer></answer>, or the last number of the text."""
    tagged = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    candidates = NUMBER.findall(tagged.group(1) if tagged else text or "")
    if not candidates:
        return None
    return float(candidates[-1].replace(",", "."))


def is_correct(final_answer: str, expected: str, tol: float = 0.01) -> bool:
    predicted = extract_number(final_answer)
    return predicted is not None and abs(predicted - float(expected)) <= tol


def text_match(final_answer: str, expected: str) -> bool:
    """Every |-separated term of ``expected`` must appear in the answer (before 'Fuentes:')."""
    body = (final_answer or "").split("Fuentes:")[0].lower()
    for term in expected.lower().split("|"):
        term = term.strip()
        pattern = rf"(?<![\d.]){re.escape(term)}(?!\d)" if term.isdigit() else re.escape(term)
        if not re.search(pattern, body):
            return False
    return True


def check_answer(tipo: str, final_answer: str, expected: str) -> bool:
    if tipo in TEXT_TYPES:
        return text_match(final_answer, expected)
    return is_correct(final_answer, expected)


def load(path: str, limit: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="data/bench_90.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--out", default="eval_orchestrator.csv")
    args = parser.parse_args()

    orchestrator = build_orchestrator()
    rows = load(args.bench, args.limit)
    results = []

    for index, row in enumerate(rows, 1):
        expected_agent = EXPECTED_AGENT[row["tipo"]]
        started = time.perf_counter()
        record = {
            "id": row["id"], "tipo": row["tipo"], "expected_agent": expected_agent,
            "routed_agent": "", "route_ok": False, "answer_ok": "", "final_answer": "",
            "expected_answer": row["answer"], "n_steps": "", "tokens": "", "succeeded": "",
            "error": "",
        }
        try:
            if args.route_only:
                decision = route(orchestrator.llm, row["question"])
                record["routed_agent"] = decision.agent
                record["route_ok"] = decision.agent == expected_agent
            else:
                response = orchestrator.run(row["question"], max_steps=args.max_steps)
                decision = orchestrator.last_decision
                record["routed_agent"] = decision.agent
                record["route_ok"] = decision.agent == expected_agent
                record.update(
                    final_answer=response.final_answer,
                    answer_ok=check_answer(row["tipo"], response.final_answer, row["answer"]),
                    n_steps=response.n_steps,
                    tokens=response.tokens_used,
                    succeeded=response.succeeded,
                )
        except Exception as exc:  # one bad question must not stop the run
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - started, 2)
        results.append(record)
        mark = "ok " if record["route_ok"] else "BAD"
        print(
            f"[{index}/{len(rows)}] {mark} {row['id']:28} -> {record['routed_agent']:12} "
            f"{record['error']}",
            flush=True,
        )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    report(results, args.route_only)
    print(f"\nper-question detail: {args.out}")


def report(results: list[dict], route_only: bool) -> None:
    total = len(results)
    route_hits = sum(r["route_ok"] for r in results)
    print(f"\nROUTING   {route_hits}/{total} = {route_hits / total:.1%}")

    confusion = Counter((r["tipo"], r["routed_agent"]) for r in results)
    for (tipo, routed), n in sorted(confusion.items()):
        print(f"  {tipo:20} -> {routed or '(error)':12} {n}")

    if not route_only:
        answer_hits = sum(r["answer_ok"] is True for r in results)
        print(f"\nANSWERS   {answer_hits}/{total} = {answer_hits / total:.1%}")
        by_tipo = defaultdict(lambda: [0, 0])
        for r in results:
            by_tipo[r["tipo"]][1] += 1
            by_tipo[r["tipo"]][0] += r["answer_ok"] is True
        for tipo, (hit, n) in sorted(by_tipo.items()):
            print(f"  {tipo:20} {hit}/{n}")

        routed_ok = [r for r in results if r["route_ok"]]
        lost = [r for r in routed_ok if r["answer_ok"] is not True]
        print(f"\nrouted correctly but answered wrong (specialist or tool fault): {len(lost)}")
        print(f"routed wrongly (router fault): {total - route_hits}")
        steps = [r["n_steps"] for r in results if isinstance(r["n_steps"], int)]
        tokens = [r["tokens"] for r in results if isinstance(r["tokens"], int)]
        if steps:
            print(f"mean steps {sum(steps) / len(steps):.1f} · mean tokens {sum(tokens) / len(tokens):.0f}")

    errors = [r for r in results if r["error"]]
    print(f"exceptions: {len(errors)}")
    for r in errors[:5]:
        print(f"  {r['id']}: {r['error']}")


if __name__ == "__main__":
    main()