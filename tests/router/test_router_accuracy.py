import json

import pytest

from agent_core.llm import build_llm
from router.router import route

EXPECTED = {"cuota": "cuota_tae", "tae": "cuota_tae",
            "coste_total": "coste_ratio", "ratio_endeudamiento": "coste_ratio"}


@pytest.mark.slow
def test_router_accuracy_on_benchmark():
    llm = build_llm()
    rows = [json.loads(line) for line in open("data/rlm_data_test.jsonl", encoding="utf-8")]
    hits = sum(route(llm, r["question"]).agent == EXPECTED[r["tipo"]] for r in rows)
    assert hits / len(rows) >= 0.95