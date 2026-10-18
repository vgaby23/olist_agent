from langgraph_supervisor import create_supervisor
from src.config import llm, SUPERVISOR_PROMPT

supervisor = create_supervisor(
		agents=[],
		model=llm,
		prompt=SUPERVISOR_PROMPT,).compile()