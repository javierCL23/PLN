from langchain_core.tools import tool

from src.rag import obtener_retriever
from src.config import PATH_VECTOR_DB


@tool
def buscar_noticias(query: str, k: int = 3) -> str:
    """
    Busca noticias similares en la base de datos (RAG) según la temática.
    
    Args:
        query: La temática, palabras clave o descripción de las noticias a buscar.
        k: El número de noticias a devolver. Si el usuario especifica una cantidad, 
           usa ese número exacto. Si no especifica ninguno, el valor por defecto es 3.
    """
    retriever = obtener_retriever(k = k, path=PATH_VECTOR_DB)
    resultados = retriever.invoke(query)

    noticias = ""
    for i,noticia in enumerate(resultados):
        noticias += f"- {i+1}. {noticia.metadata.get('original_title')}\n"
    return noticias