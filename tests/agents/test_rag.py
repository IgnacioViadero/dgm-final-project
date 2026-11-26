from agents import rag
from agent_core.llm import build_llm

llm = build_llm()
agent = rag.build(llm)
r = agent.run("¿Cuál es la capital de España?")
print(r.final_answer)
for s in r.steps:
    print(s.action, s.observation)