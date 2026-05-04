import os

import pandas as pd
import numpy as np

# Vectorizaciones
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch

from .config import PATH_MODELS



# Descarga de Word2Vec

# Variable global de módulo para actuar como caché
_W2V_MODEL = None

def get_word2vec_model():
    """
    Carga el modelo Word2Vec si ya existe, o lo descarga si no.
    Retorna el modelo cargado.
    """
    global _W2V_MODEL
    
    # Si ya lo cargamos en esta sesión, no repetimos el proceso
    if _W2V_MODEL is not None:
        return _W2V_MODEL

    model_name = "word2vec-google-news-300"
    # Construimos la ruta usando tu objeto Path de config.py
    ruta_modelo_gz = PATH_MODELS / model_name / f"{model_name}.gz"

    if ruta_modelo_gz.exists():
        print(f"Modelo {model_name} ya detectado en local. Cargando...")
        # Convertimos el Path a string para Gensim
        _W2V_MODEL = KeyedVectors.load_word2vec_format(str(ruta_modelo_gz), binary=True)
    else:
        print(f"Modelo no encontrado. Descargando {model_name}...")
        # Configuramos las variables de entorno para que Gensim use la carpeta de PATH_MODELS
        os.environ["GENSIM_DATA_DIR"] = str(PATH_MODELS)
        api.BASE_DIR = str(PATH_MODELS)
        
        _W2V_MODEL = api.load(model_name)
        print(f"{model_name} descargado y cargado.")

    return _W2V_MODEL



# Vectorización Dispersa vs Densa

# Dispersas

def BoW(textos: pd.Series) -> pd.DataFrame:
    """
    Convierte todos los textos en un Bag Of Words
    limitado a las palabras más frecuentes de los textos.
    """
    # max_features limita a las N palabras más frecuentes
    vectorizador = CountVectorizer(max_features=5000, min_df=2, stop_words='english')
    
    X = vectorizador.fit_transform(textos)
    
    return pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())

def TF_IDF(textos: pd.Series) -> pd.DataFrame:
    """
    Convierte todos los textos en una matriz con TF-IDF.
    """
    vectorizador = TfidfVectorizer(max_features=5000, min_df=2, stop_words='english')
    
    X = vectorizador.fit_transform(textos)
    
    return pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())


# Densas

def Word2Vec_Embeddings(textos: pd.Series, modelo_w2v) -> pd.DataFrame:
    """
    Calcula el vector promedio de cada documento usando un modelo Word2Vec preentrenado.
    Recibe la serie de textos y el modelo ya cargado en memoria.
    Recomendación: Pasar 'desc_basico'.
    """
    descripciones_tokenizadas = textos.str.split().tolist()

    dimensiones = modelo_w2v.vector_size 
    
    def obtener_vector_documento(tokens):
        # Filtrar palabras existentes en el vocabulario preentrenado
        palabras_conocidas = [word for word in tokens if word in modelo_w2v]
        
        if not palabras_conocidas:
            # Si ninguna palabra del texto está en el modelo
            return np.zeros(dimensiones)
        
        # Media de los embeddings de las palabras
        return np.mean(modelo_w2v[palabras_conocidas], axis=0)

    # Aplicar a todas las descripciones
    vectores_documentos = [obtener_vector_documento(tokens) for tokens in descripciones_tokenizadas]
    
    nombres_columnas = [f"w2v_google_{i}" for i in range(dimensiones)]
    return pd.DataFrame(vectores_documentos, index=textos.index, columns=nombres_columnas)


def BERT_Embeddings(series_de_texto: pd.Series) -> pd.DataFrame:
    """
    Utiliza BertModel nativo para extraer embeddings contextuales.
    Toma el token especial [CLS] como representación semántica del texto completo.
    """
    print("Cargando modelo BERT y tokenizador...")
    # Cargamos la versión uncased (todo minúsculas por defecto en BERT)
    nombre_modelo = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(nombre_modelo, cache_dir=PATH_MODELS)
    modelo = BertModel.from_pretrained(nombre_modelo, cache_dir=PATH_MODELS)
    
    # Cargar modelo a GPU si hay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()
    
    todos_los_vectores = []
    batch_size = 32
    
    print(f"Generando embeddings en {device.type.upper()}...")
    
    # Procesamiento en batches
    for i in range(0, len(series_de_texto), batch_size):
        batch_textos = series_de_texto.iloc[i : i + batch_size].tolist()
        
        # Tokenización
        # max_length=128 recorta textos largos
        inputs = tokenizer(batch_textos, 
                           padding=True, 
                           truncation=True, 
                           max_length=128, 
                           return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = modelo(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            todos_los_vectores.append(cls_embeddings.cpu().numpy())
            
    # Juntar todos los batches
    matriz_final = np.vstack(todos_los_vectores)
    
    nombres_columnas = [f"bert_hf_{i}" for i in range(matriz_final.shape[1])]
    return pd.DataFrame(matriz_final, index=series_de_texto.index, columns=nombres_columnas)
