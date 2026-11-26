import json

import pandas as pd

from router.orchestrator import build_orchestrator
from router.orchestrator_eval import is_correct  # or router.orchestrator_eval, wherever it lives

df = pd.read_csv("eval_orchestrator.csv")
failed = set(df[(df.tipo == "tae") & (df.answer_ok == False)].id)
rows = {r["id"]: r for r in map(json.loads, open("data/rlm_data_test.jsonl", encoding="utf-8"))}

orch = build_orchestrator()
for id_ in list(failed)[:5]:
    row = rows[id_]
    resp = orch.run(row["question"], max_steps=6)
    print("=" * 80)
    print(id_, "expected:", row["answer"], "got:", repr(resp.final_answer), "ok:", resp.succeeded)
    print("true inputs:", row["_debug_prestamo"])
    for n, step in enumerate(resp.steps, 1):
        action = step.action.name if step.action else "-"
        args = step.action.arguments if step.action else ""
        print(f"  {n}. {action} {args}\n     -> {str(step.observation)[:200]}")