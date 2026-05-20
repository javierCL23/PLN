from langchain_ollama import ChatOllama

from src.state import Tasks
from src.config import ORCHESTRATOR_MODEL, WORKER_MODEL
from src.tools import buscar_noticias

# LLM con esquema de tasks
llm_tasker = ChatOllama(
    model=ORCHESTRATOR_MODEL,
    temperature=0.3
    ).with_structured_output(Tasks)

# LLM con acceso a tools de recomendador
llm_recommendator = ChatOllama(
    model = WORKER_MODEL,
    temperature=0
    ).bind_tools([buscar_noticias])