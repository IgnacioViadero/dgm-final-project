"""The orchestrator: classify, decompose, dispatch, verify, synthesise.

Only the public entry point lives here for now. ``build_orchestrator`` is the single
factory the API (``api/backends.py``) and the evaluation (``eval/evaluate.py``) call, so
the rest of the system does not depend on how the orchestrator is built.
"""

from router.orchestrator import build_orchestrator

__all__ = ["build_orchestrator"]
