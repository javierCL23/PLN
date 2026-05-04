import string

import pandas as pd
import numpy as np

# Preprocesados
import spacy
import en_core_web_sm


translator = str.maketrans('', '', string.punctuation)
nlp = en_core_web_sm.load()


def limpieza_comun(series_de_texto: pd.Series) -> pd.Series:
    """
    Aplica limpieza común a toda una Serie de Pandas usando métodos vectorizados .str
    """
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

def preprocessing_basico(series_de_texto: pd.Series, nlp) -> pd.Series:
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

def preprocessing_avanzado(series_de_texto: pd.Series, nlp) -> list:
    """
     Limpia el texto convirtiéndolo todo a minúsculas y eliminando signos de puntuación.
    También elimina las palabras que sean stopWords.
    Útil para modelos estadísticos básados en frecuencias. 
    """
    textos_limpios = []
    
    for doc in nlp.pipe(limpieza_comun(series_de_texto), 
                        batch_size=1000, 
                        disable=["ner", "parser"]):
        
        tokens = [word.lemma_.lower() for word in doc 
                  if not word.is_stop 
                  and not word.is_punct 
                  and word.is_alpha]
        
        textos_limpios.append(" ".join(tokens))
        
    return textos_limpios
