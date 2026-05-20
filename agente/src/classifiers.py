from collections import Counter

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils import fallback_postprocessing
from src.config import CLASS_NAMES

def predecir_categoria(titulo: str, fallback: bool = True) -> str:
    """
    Dado un título, usa un LLM para predecir la categoría.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a news classifier. Your task is to classify the given title into one of these categories: {categorias}.

Answer ONLY with the exact category name from the list. Do not add explanations or formatting.

Given this title: "{titulo}"
Category: """)

    # Temperatura = 0 para evitar que el modelo se pase de explicativo
    llm = ChatOllama(model="granite4:micro", temperature=0)
    chain = prompt | llm
    
    resultado = chain.invoke({
        "categorias": ", ".join(CLASS_NAMES),
        "titulo": titulo
    })
    resultado = resultado.content.strip()

    if fallback:
        return fallback_postprocessing(resultado)
    return resultado

def predecir_categoria_rag(titulo: str, retriever, fallback: bool = True) -> str:
    """
    Dado un título, recupera las descripciones más relevantes del train
    y usa un LLM para predecir la categoría.
    """
    # Recuperación
    docs = retriever.invoke(titulo)
    
    # Contexto con categorías
    contexto = "\n".join([
        f"{i+1}. Category: {doc.metadata['category']} - Description: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are an advanced news classifier. Your primary task is to classify a given title based STRICTLY on the patterns and categories shown in the provided context.

Context (Similar examples and their classifications):
{contexto}

CRITICAL INSTRUCTIONS:
1. Analyze the context above. The examples provided are highly relevant and share direct semantic meaning with the title you need to classify.
2. Rely on the context to guide your decision. If a similar description in the context belongs to a specific category, the new title most likely belongs to that same category.
3. Answer ONLY with the category.
Given this title: "{titulo}"
Category: """)
    
    # print("======================================")
    # print("Descripciones de Contexto Recuperadas:")
    # print("======================================")
    # print(contexto)
    # print()
    # print("Predicción:")

    # Temperatura = 0 para evitar que el modelo se pase de explicativo
    llm = ChatOllama(model="granite4:micro", temperature=0)
    chain = prompt | llm
    
    resultado = chain.invoke({
        "categorias": ", ".join(CLASS_NAMES),
        "contexto": contexto,
        "titulo": titulo
    })
    resultado = resultado.content.strip()

    if fallback:
        return fallback_postprocessing(resultado)
    return resultado

def predecir_categoria_knn(titulo: str, retriever) -> str:
    """
    Dado un título, recupera las descripciones más relevantes del train
    y usa un clasificador basado en KNN para hacer la predicción.
    """

    docs = retriever.invoke(titulo)

    categorias = [doc.metadata['category'] for doc in docs]
    
    conteo = Counter(categorias)
    
    #conteo.most_common(1) = [(categoria_ganadora, votos)]
    # En caso de empate, se coge el primero que aparezca en la lista (el primero que entró, que tiene prioridad en el retrieval)
    categoria_ganadora = conteo.most_common(1)[0][0]
    
    return categoria_ganadora