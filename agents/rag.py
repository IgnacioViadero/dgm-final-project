"""Specialist for conceptual and legal questions about consumer credit (RAG).

Router -> this agent -> one retrieval tool. The model answers only from the retrieved
passages. Two guards sit outside the model, because a small model does not obey the prompt
when the search fails:
  * no successful search -> the model's answer is discarded and replaced by a fixed message;
  * the sources are taken from the tool observation and appended by the code.
"""

from __future__ import annotations

import re

from agent_core.llm import Brain
from agent_core.react_agent import ReActAgent
from api.schemas import AgentResponse
from tool_use.registry import ToolRegistry
from tool_use.tools_rag import search_knowledge


NAME = "rag"

NO_SEARCH = (
    "No he podido consultar la base de conocimiento, así que no puedo responder con fiabilidad."
)
GARBLED = "No he podido elaborar una respuesta fiable a partir de los documentos."
SOURCE_PATTERN = re.compile(r"\[([\w.\-]+\.(?:pdf|md|markdown|txt|html?))\]")
BAD_ANSWER = re.compile(r"</?tool_call>|</?think>|^\s*\{|\"result\"")
MAX_ANSWER_CHARS = 1500


RULES = """Additional rules:
- Reply in the same language as the user.
- Emit ONE tool call per message, inside <tool_call></tool_call>, and stop: the observation
  is written by the system, never by you.
- Always call search_knowledge first, with the user's question as the query. Call it at most
  twice (a second time only with different keywords if the first result is not relevant).
- Answer ONLY with information that appears in the returned passages. Never use your own
  knowledge and never invent figures, laws, articles or dates.
- Use only facts that are stated in the passages. Do not add consequences, conditions or
  figures that the passages do not state.
- If the passages do not contain the answer, call final_answer saying you do not have that
  information.
- If search_knowledge returns an error, do not retry the same query: call final_answer
  saying you could not consult the knowledge base.
- Every tool answers with {"result": the passages, "inputs": the query and the sources}.
  Read "result"; each passage starts with its source between brackets.
- Finish by calling final_answer with a clear answer of 2 to 4 sentences, written as text,
  in your own words.
"""


class RagAgent(ReActAgent):
    """ReAct agent whose answer must come from a successful search, and cites its sources."""

    def run(self, task: str, max_steps: int = 6) -> AgentResponse:
        response = super().run(task, max_steps)
        searches = [
            step.observation
            for step in self.steps
            if step.action
            and step.action.name == "search_knowledge"
            and step.observation
            and not step.observation.startswith("ERROR")
        ]
        if not searches:
            return response.model_copy(update={"final_answer": NO_SEARCH, "succeeded": False})

        answer = response.final_answer
        if answer and (BAD_ANSWER.search(answer) or len(answer) > MAX_ANSWER_CHARS):
            return response.model_copy(update={"final_answer": GARBLED, "succeeded": False})

        if answer and "no dispongo" not in answer.lower():
            sources = list(dict.fromkeys(SOURCE_PATTERN.findall(searches[-1])))
            if sources:
                answer = f"{answer}\n\nFuentes: {', '.join(sources)}"
        return response.model_copy(update={"final_answer": answer})


def build(llm: Brain) -> ReActAgent:
    """The knowledge-base specialist, sharing the given model with the other agents."""
    registry = ToolRegistry().register(search_knowledge)
    return RagAgent(NAME, llm, registry, RULES)