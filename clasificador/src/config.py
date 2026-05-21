import os
from pathlib import Path

_CURRENT_FILE = Path(__file__).resolve()
_CLASSIFICATOR_ROOT = _CURRENT_FILE.parent.parent
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent


PATH_MODELS = Path(os.getenv("PATH_MODELS", _CLASSIFICATOR_ROOT / "models"))
PATH_DATA = Path(os.getenv("PATH_DATA", _PROJECT_ROOT / "data"))

def ensure_dirs():
    PATH_MODELS.mkdir(parents=True, exist_ok=True)
    PATH_DATA.mkdir(parents=True, exist_ok=True)

W2V_MODEL = "word2vec-google-news-300"
BERT_MODEL = "bert-base-uncased"

FINE_TUNING_MODEL = "distilbert-base-uncased"
FINE_TUNING_PATH = PATH_MODELS / FINE_TUNING_MODEL