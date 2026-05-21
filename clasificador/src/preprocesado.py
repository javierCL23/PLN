import string
import html
from typing import Literal

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

from .config import PATH_DATA

# Preprocesados
import spacy
import en_core_web_sm


translator = str.maketrans('', '', string.punctuation)
nlp = en_core_web_sm.load()


def limpieza_comun(series_de_texto: pd.Series) -> pd.Series:
    """
    Aplica limpieza común a toda una Serie de Pandas usando métodos vectorizados .str
    """

    # Decodificar entidades HTML (&#39; a ', &amp; a &, etc.)
    texto_limpio = series_de_texto.fillna("").apply(html.unescape)

    # El dataset contiene '\' que deben ser eliminadas manualmente
    texto_limpio = series_de_texto.str.replace(r'\\', ' ', regex=True)
    
    # Eliminar espacios múltiples y aplicar strip()
    texto_limpio = texto_limpio.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return texto_limpio

def preprocessing_minimal(series_de_texto: pd.Series) -> pd.Series:
    """
    Mantiene puntuación, mayúsculas y estructura.
    Útil para sistemas más complejos como LLMs y Transformers.
    """
    return limpieza_comun(series_de_texto)

def preprocessing_basico(series_de_texto: pd.Series) -> pd.Series:
    """
    Limpia el texto convirtiéndolo a minúsculas, eliminando signos de puntuación
    y lematizando todas las palabras. Conserva las stopwords.
    Útil para TF-IDF o Word2Vec.
    """ 
    textos_limpios = []
    
    for doc in nlp.pipe(limpieza_comun(series_de_texto), 
                        batch_size=1000, 
                        disable=["ner", "parser"]):
        
        # Lematizamos, pasamos a minúscula y quitamos puntuación.
        tokens = [word.lemma_.lower() for word in doc if not word.is_punct]
        
        textos_limpios.append(" ".join(tokens))
        
    # Devolvemos un pd.Series con los índices originales intactos
    return pd.Series(textos_limpios, index=series_de_texto.index)

def preprocessing_avanzado(series_de_texto: pd.Series) -> list:
    """
    Limpia el texto convirtiéndolo todo a minúsculas y eliminando signos de puntuación.
    También elimina las palabras que sean stopWords.
    Útil para modelos estadísticos básados en frecuencias. 
    """
    textos_limpios = []
    
    for doc in nlp.pipe(limpieza_comun(series_de_texto), 
                        batch_size=1000, 
                        disable=["ner", "parser"]):
        
        # Lematización, pasar a minúsculas, eliminar stopwords, puntuaciones.
        tokens = [word.lemma_.lower() for word in doc 
                  if not word.is_stop 
                  and not word.is_punct 
                  and word.is_alpha]
        
        textos_limpios.append(" ".join(tokens))
        
    return textos_limpios

def preprocesar_datos(df: pd.DataFrame, conjunto:Literal["train","test"], guardar: bool = False) -> pd.DataFrame:
    """
    Añade columnas con los preprocesados de texto al conjunto df.
    'conjunto' puede tomar valores "train" o "test".
    Por defecto no guarda el dataframe resultante en almacenamiento persistente.
    En caso contrario, lo guarda en PATH_DATA. 
    """
    print("Minimo iniziando...",end=" ")
    df['desc_minimal'] = preprocessing_minimal(df.description)
    print("Minimo Hecho!!")
    print("Básico iniziando...",end=" ")
    df['desc_basico'] = preprocessing_basico(df.description)
    print("Básico Hecho!!")
    print("Avanzado iniziando...",end=" ")
    df['desc_avanzado'] = preprocessing_avanzado(df.description)
    print("Avanzado Hecho!!")

    # Ya no se va a usar la descripción original. La descripción orignal ahora será la minimal.
    df = df.drop(["description"], axis=1)

    # Guardar datos
    if guardar:
        save_path = PATH_DATA / f"{conjunto}_preprocesado.jsonl"
        df.to_json(
            save_path, 
            orient='records', 
            lines=True, 
            force_ascii=False
        )
    
    return df

def cargar_datos(test: bool = False, guardar: bool = True) -> pd.DataFrame:
    """
    Carga los datos de un path, revisando antes si existe la versión preprocesada.
    En caso de no existir, preprocesa los datos y los guarda en 'path_prep'.
    Si no se especifica si los datos son los de test o train, se usará train por defecto.
    Si no se especifica si se quiere guardar datos preprocesados, por defecto lo hará.
    """

    # Define si se cargará train o test
    if test:
        conjunto = "test"
    else:
        conjunto = "train"

    path_raw = PATH_DATA / f"{conjunto}.jsonl"
    path_prep = PATH_DATA / f"{conjunto}_preprocesado.jsonl"

    df = None

    # Datos preprocesados no existen
    if not path_prep.exists():
        print("Cargando datos...")
        # Datos crudos no existen
        if not path_raw.exists():
            print("Datos no encontrados. Descargando...")
            hf_hub_download(
                repo_id="sh0416/ag_news",
                filename=f"{conjunto}.jsonl",
                repo_type="dataset",
                local_dir=PATH_DATA,
                local_dir_use_symlinks=False
            )
        df = pd.read_json(str(path_raw), lines=True)
        df = df.dropna(subset=["description", "title"])
        df = df.drop_duplicates(subset=["title"])
        df = preprocesar_datos(df,conjunto, guardar)
    else:
        print("Cargando datos preprocesados...")
        df = pd.read_json(str(path_prep), lines=True)

    return df
