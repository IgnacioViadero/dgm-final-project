from agent_core.llm import DEFAULT_MODEL, LLM


def test_model_is_not_loaded_on_construction():
    llm = LLM()
    assert llm._model is None
    assert llm.model_id == DEFAULT_MODEL


def test_model_id_comes_from_environment(monkeypatch):
    monkeypatch.setenv("ARCA_MODEL", "Qwen/Qwen3-4B")
    assert LLM().model_id == "Qwen/Qwen3-4B"


def test_count_tokens_estimates_before_loading():
    assert LLM().count_tokens("a" * 40) == 10