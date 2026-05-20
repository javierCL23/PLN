from typing import Literal
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path

from src.config import PATH_DATA, CLASS_NAMES

def preprocesado_minimo(series_de_texto: pd.Series) -> pd.Series:
    """
    Aplica limpieza común a toda una Serie de Pandas usando métodos vectorizados .str
    """
    # El dataset contiene '\' que deben ser eliminadas manualmente
    texto_limpio = series_de_texto.str.replace(r'\\', ' ', regex=True)
    
    # Eliminar espacios múltiples y aplicar strip()
    texto_limpio = texto_limpio.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return texto_limpio

def cargar_datos(path: Path = PATH_DATA, conjunto: Literal["train","test"] = "train") -> pd.DataFrame:
    """
    Devuelve el conjunto de datos en forma de dataframe. Permite ajustar el path a una ruta concreta.
    Las opciones de tipo de conjunto son: 'train' o 'test'
    """

    # Descarga de datos en caso de ser necesario
    data_path = path / f"{conjunto}.jsonl"
    if not data_path.exists():
        print("Datos no encontrados. Descargando...")
        hf_hub_download(
            repo_id="sh0416/ag_news",
            filename=f"{conjunto}.jsonl",
            repo_type="dataset",
            local_dir=PATH_DATA,
            local_dir_use_symlinks=False
        )
    
    # Lectura y preprocesa
    df = pd.read_json(data_path, lines=True)
    df = df.dropna(subset=["description", "title"])
    df = df.drop_duplicates(subset=["title"])
    df["description"] = preprocesado_minimo(df["description"])
    
    return df

def fallback_postprocessing(resultado: str) -> str:
    """
    Función para asegurar devolver la categoría concreta.
    """
    prediccion = resultado.replace(".", "")
    for categoria in CLASS_NAMES:
        if categoria.lower() in prediccion.lower():
            return categoria
    return prediccion
