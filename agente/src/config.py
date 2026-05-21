import os
from pathlib import Path

# Variables de PATHS
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
_AGENT_ROOT = _CURRENT_FILE.parent.parent

PATH_DATA = Path(os.getenv("PATH_DATA", _PROJECT_ROOT / "data")) #: Ubicación de los datos
PATH_VECTOR_DB = Path(os.getenv("PATH_VECTOR_DB", _AGENT_ROOT / "vector_db")) #: Dirección en la que guardar la vector store en almacenamiento

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
LLM_MODEL = os.getenv("LLM_MODEL","granite4:micro") #: Modelo general para testing de capacidad predictiva en categorización (granite4:micro por velocidad)
EMBEDDING_MODEL = "sentence-transformers/msmarco-distilbert-base-v4" #: 'sentence-transformers/msmarco-distilbert-base-v4' Modelo base para embeddings de la Vector Store. Ventaja principal: útil para retrieval con texto corto para texto largo

LLM_PROVIDER = os.getenv("LLM_PROVIDER","ollama") #: Provider elegido para conexión con LLM. Permitido ollama o openai
LLM_URL = os.getenv("LLM_URL",None) #: Dirección URL para conexión con el modelo en caso de usar provider ollama
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL","granite4:micro") #: Modelo usado por el nodo orquestador del agente (mixtral7:8b overkill para la tarea, granite4:micro es más que suficiente)
WORKER_MODEL = os.getenv("WORKER_MODEL","granite4:micro") #: Modelo usado por los workers del agente (grantie4:micro, solo necesita saber que tiene que usar tools, no necesitan razonamiento más allá de extracción de argumentos)