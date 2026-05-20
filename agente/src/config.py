import os
from pathlib import Path


# Variables de PATHS
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
_AGENT_ROOT = _CURRENT_FILE.parent.parent

PATH_DATA = Path(os.getenv("PATH_DATA", _PROJECT_ROOT / "data"))
PATH_VECTOR_DB = Path(os.getenv("PATH_VECTOR_DB", _AGENT_ROOT / "vector_db"))

def ensure_dirs():
    PATH_VECTOR_DB.mkdir(parents=True, exist_ok=True)
    PATH_DATA.mkdir(parents=True, exist_ok=True)


# Variables de clases
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
LABEL_TO_CLASS = {
    1: "World",
    2: "Sports", 
    3: "Business",
    4: "Sci/Tech"
}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}

# Variables de modelos
LLM_MODEL = os.getenv("LLM_MODEL","granite4:micro")
EMBEDDING_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"

LLM_PROVIDER = os.getenv("LLM_PROVIDER","ollama")
LLM_URL = os.getenv("LLM_URL",None)
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL","granite4:micro")
WORKER_MODEL = os.getenv("WORKER_MODEL","granite4:micro")