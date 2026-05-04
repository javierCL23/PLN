from pathlib import Path


_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CURRENT_FILE.parent.parent

PATH_MODELS = PROJECT_ROOT / "models"
PATH_DATA = PROJECT_ROOT / "data"
# PATH_SRC = PROJECT_ROOT / "src"

def ensure_dirs():
    PATH_MODELS.mkdir(parents=True, exist_ok=True)
    PATH_DATA.mkdir(parents=True, exist_ok=True)

