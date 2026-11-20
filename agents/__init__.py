"""Specialist agents.

Every module here exposes the same three names so the orchestrator can treat them
interchangeably:

* ``NAME``: the identifier the router uses.
* ``RULES``: the specialist-specific rules appended to the shared ReAct system prompt.
* ``build(llm) -> ReActAgent``: the agent, sharing the given model with the others.

Only the tools and the rules change between specialists; the loop is
``agent_core.react_agent.ReActAgent`` in every case.
"""
