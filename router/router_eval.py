import json
from collections import Counter

from agent_core.llm import build_llm
from router.router import route

EXPECTED = {
    "cuota": "cuota_tae",
    "tae": "cuota_tae",
    "coste_total": "coste_ratio",
    "ratio_endeudamiento": "coste_ratio",
}

llm = build_llm()
with open("data/rlm_data_test.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

confusion, errors, fallbacks = Counter(), [], 0
for row in rows:
    decision = route(llm, row["question"])
    expected = EXPECTED[row["tipo"]]
    confusion[(row["tipo"], decision.agent)] += 1
    fallbacks += decision.reason.startswith("fallback")
    if decision.agent != expected:
        errors.append((row["id"], expected, decision.agent))

hits = len(rows) - len(errors)
print(f"router accuracy: {hits}/{len(rows)} = {hits / len(rows):.1%}  (fallbacks: {fallbacks})")
for (tipo, predicted), n in sorted(confusion.items()):
    print(f"{tipo:20} -> {predicted:12} {n}")
for item in errors:
    print("FAIL", item)