"""Loading of the local configuration, in one place.

Every entry point (the API, the agents' scripts, the evaluation) calls ``load_env`` before
reading any ``ARCA_*`` variable, so a developer only has to fill ``.env``. The rules:

* ``.env`` is ignored by git and holds the secrets (``ARCA_LLM_API_KEY``). It is never
  committed, printed or logged.
* ``.env.example`` is the versioned template: variable names and safe defaults, no values
  that are secret.
* a variable already set in the real environment wins over ``.env``, so SLURM, Docker and
  ``ARCA_LLM_BACKEND=endpoint uv run ...`` keep working as expected.

``python-dotenv`` is a base dependency, but if it were missing the import must not take the
application down: without it the environment simply has to be set by hand.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("arca.config")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
_loaded = False


def load_env(path: Path | None = None, override: bool = False) -> bool:
    """Load ``.env`` into the environment once. Returns whether a file was read.

    Args:
        path: The file to read. Defaults to ``.env`` at the root of the repository.
        override: Whether values in the file replace variables already set. False by
            default: the real environment wins.
    """
    global _loaded
    env_file = path or ENV_FILE
    if _loaded and path is None:
        return False
    if os.environ.get("ARCA_SKIP_DOTENV"):  # for tests and reproducible runs
        _loaded = True
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a base dependency
        log.warning("python-dotenv is not installed: set the ARCA_* variables by hand")
        _loaded = True
        return False
    read = load_dotenv(env_file, override=override)
    if path is None:
        _loaded = True
    return bool(read)
