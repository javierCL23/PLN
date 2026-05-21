from langchain_core.tools import tool

from .rag import obtener_retriever, obtener_vs
from .config import PATH_VECTOR_DB
from .classifiers import predecir_categoria_knn

VECTOR_STORE_GLOBAL = obtener_vs(PATH_VECTOR_DB)

@tool
def buscar_noticias(query: str, k: int = 3) -> str:
    """
    Busca noticias similares en la base de datos (RAG) según la temática.
    
    Args:
        query: La temática, palabras clave o descripción de las noticias a buscar.
        k: El número de noticias a devolver. Si el usuario especifica una cantidad, 
           usa ese número exacto. Si no especifica ninguno, el valor por defecto es 3.
    """
    retriever = obtener_retriever(k = k, vector_store=VECTOR_STORE_GLOBAL)
    resultados = retriever.invoke(query)

    noticias = ""
    for i,noticia in enumerate(resultados):
        noticias += f"- {i+1}. {noticia.metadata.get('original_title')}\n"
    return noticias

@tool
def clasificar_noticia(query: str) -> str:
    """
    Clasifica una noticia gracias a un RAG según las noticias similares recuperadas.

    Args:
        - query: Texto o titular que se buscará en el RAG para encontrar matches similares.
        En base a la clase de los textos recuperados es que se decide la temática del la query.
    """
    retriever = obtener_retriever(k = 5, vector_store=VECTOR_STORE_GLOBAL)
    return predecir_categoria_knn(query,retriever)
    