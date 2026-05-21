from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .state import Tasks
from .config import ORCHESTRATOR_MODEL, WORKER_MODEL, LLM_PROVIDER, LLM_URL
from .tools import buscar_noticias, clasificar_noticia

if LLM_PROVIDER.lower() == "openai":
    base_orchestrator = ChatOpenAI(model=ORCHESTRATOR_MODEL, temperature=0.3)
    base_worker_t0 = ChatOpenAI(model=WORKER_MODEL, temperature=0)
    base_worker_t03 = ChatOpenAI(model=WORKER_MODEL, temperature=0.3)
else:
    # Por defecto Ollama
    base_orchestrator = ChatOllama(model=ORCHESTRATOR_MODEL, temperature=0.3, base_url=LLM_URL)
    base_worker_t0 = ChatOllama(model=WORKER_MODEL, temperature=0, base_url=LLM_URL)
    base_worker_t03 = ChatOllama(model=WORKER_MODEL, temperature=0.3, base_url=LLM_URL)

# LLM con esquema de tasks
llm_tasker = base_orchestrator.with_structured_output(Tasks)

# LLM con acceso a tools de recomendador
llm_recommendator = base_worker_t0.bind_tools([buscar_noticias])

# LLM con acceso a tools de clasificador
llm_clasificator = base_worker_t0.bind_tools([clasificar_noticia])

# LLM encargado de reestructurar toda la información recogida por los workers
llm_synthesizer = base_worker_t03