"""Loading of .env: the real environment wins, and the secret never has a default."""

import os

import pytest

from agent_core import config
from agent_core.llm import DEFAULT_ENDPOINT_URL, EndpointLLM


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "ARCA_LLM_URL=http://from-file:8001\nARCA_LLM_MODEL=from-file\n", encoding="utf-8"
    )
    return path


@pytest.fixture(autouse=True)
def allow_dotenv(monkeypatch):
    """Lift the test-wide skip and undo what loading a file writes into os.environ.

    ``load_dotenv`` writes straight into ``os.environ``, which monkeypatch cannot roll
    back, so the whole environment is snapshotted and restored around each test.
    """
    snapshot = dict(os.environ)
    monkeypatch.delenv("ARCA_SKIP_DOTENV", raising=False)
    config._loaded = False
    yield
    os.environ.clear()
    os.environ.update(snapshot)
    config._loaded = False


def test_values_from_the_file_reach_the_environment(monkeypatch, env_file):
    monkeypatch.delenv("ARCA_LLM_MODEL", raising=False)
    assert config.load_env(env_file)
    assert os.environ["ARCA_LLM_MODEL"] == "from-file"


def test_the_real_environment_wins_over_the_file(monkeypatch, env_file):
    monkeypatch.setenv("ARCA_LLM_MODEL", "from-the-shell")
    config.load_env(env_file)
    assert os.environ["ARCA_LLM_MODEL"] == "from-the-shell"


def test_override_is_available_but_not_the_default(monkeypatch, env_file):
    monkeypatch.setenv("ARCA_LLM_MODEL", "from-the-shell")
    config.load_env(env_file, override=True)
    assert os.environ["ARCA_LLM_MODEL"] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert config.load_env(tmp_path / "nope.env") is False


def test_the_skip_flag_disables_loading(monkeypatch, env_file):
    monkeypatch.setenv("ARCA_SKIP_DOTENV", "1")
    monkeypatch.delenv("ARCA_LLM_MODEL", raising=False)
    assert config.load_env(env_file) is False
    assert "ARCA_LLM_MODEL" not in os.environ


def test_the_repository_env_file_is_ignored_by_git():
    """A regression guard: the file that holds the API key must never be committed."""
    import subprocess

    done = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], cwd=config.REPO_ROOT, check=False
    )
    assert done.returncode == 0, ".env is NOT ignored by git: the API key would be committed"


# -- the service's address and its key ------------------------------------------------


def test_the_default_url_is_the_current_service():
    assert DEFAULT_ENDPOINT_URL == "http://0.0.0.0:8001"
    assert EndpointLLM(url=None, api_key="x").url == "http://0.0.0.0:8001/v1/chat/completions"


def test_the_api_key_travels_as_a_bearer_token(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(headers or {})

        class Reply:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

            def raise_for_status(self):
                return None

        return Reply()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    EndpointLLM(url="http://h:8001", api_key="a-secret-key").generate([])
    assert sent["Authorization"] == "Bearer a-secret-key"


def test_the_key_is_not_hardcoded_anywhere_in_the_source():
    """The key must come from the environment: no literal fallback in the code."""
    source = (config.REPO_ROOT / "agent_core" / "llm.py").read_text(encoding="utf-8")
    assert 'os.environ.get("ARCA_LLM_API_KEY"' in source
    example = (config.REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ARCA_LLM_API_KEY=\n" in example  # the template leaves it empty
