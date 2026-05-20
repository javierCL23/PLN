from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import pandas as pd

from src.config import PATH_VECTOR_DB, EMBEDDING_MODEL, LABEL_TO_CLASS


def obtener_vs(path: str | None = None) -> Chroma:
    """
    Devuelve un objeto Chroma. 
    Lo puede crear desde 0 o tomando un Path
    """
    if path is None:
        raise ValueError("Debes proporcionar un objeto 'vector_store' o un 'path' válido para la VS.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vector_store = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vector_store

def obtener_retriever(k: int = 3, vector_store: Chroma | None = None, path: str | None = None):
    """
    Devuelve un objeto Retriever configurado con un top-k específico.
    Permite pasar un objeto vector_store ya instanciado. 
    Si no se pasa, es obligatorio proporcionar un 'path' para cargarlo desde el disco.
    """

    # Cargar VS desde el path si no se pasa como argumentos
    if vector_store is None:
        vector_store = obtener_vs(path)
    
    # Cargar retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    
    return retriever


def generate_vector_database(df: pd.DataFrame, space: str = "l2", save: bool = True) -> (Chroma | None):
    """
    Genera una base de datos vectorial usando para ello el conjunto de datos que se le pase.
    El formato de los datos debe estar definido con los campos 'description', 'title' y 'label'.
    
    Args:
        df (pd.DataFrame): Datos a indexar.
        space (str): Medida de distancia a usar ('cosine', 'l2', o 'ip').
        save (bool): Si es True, guarda en disco y omite la generación si ya existe. 
                     Si es False, genera en memoria saltándose la comprobación y devuelve el objeto.
    """
    # Check de existencia de la BBDD
    if save and PATH_VECTOR_DB.exists() and any(PATH_VECTOR_DB.iterdir()):
        print(f"La base de datos en {PATH_VECTOR_DB} ya existe. Omitiendo generación.")
        return None

    # Check de consistencia en el df
    if df is None:
        raise ValueError("El DataFrame no puede ser None.")
    if not {"description", "title", "label"}.issubset(df.columns):
        raise ValueError(f"Faltan columnas requeridas. Encontradas: {list(df.columns)}")
        
    print(f"Total de descripciones a indexar: {len(df)}")
    
    # Creación de documentos y embeddings
    documents = [
        Document(
            page_content=row.description,
            metadata={"row_id": row.Index, "original_title": row.title, "category": LABEL_TO_CLASS[row.label]}
        ) 
        for row in df[["description", "title", "label"]].itertuples(name="Row")
    ]
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Generando embeddings e indexando en ChromaDB usando métrica '{space}'...")
    
    # Creación de vs
    persist_dir = str(PATH_VECTOR_DB) if save else None
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": space}
    )
    
    # Si no se quiere guardar, se carga como variable
    if save:
        print(f"Base de datos vectorial guardada en: {PATH_VECTOR_DB}")
        return None
    else:
        print("Base de datos vectorial generada en memoria (no persistente).")
        return vector_store