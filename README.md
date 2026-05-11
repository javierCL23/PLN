# Clasificación y Recuperación de Información de Noticias (Práctica Final)

Este repositorio contiene la implementación de la Práctica Final, enfocada en resolver dos tareas principales de Procesamiento de Lenguaje Natural (NLP) sobre un corpus de noticias.
1. **Clasificación temática de textos.**
2. **Sistema RAG (Retrieval-Augmented Generation) con un agente conversacional.**

---

## Estructura del Proyecto

### Ejercicio 1: Clasificación Temática

En esta sección se explora la evolución de los sistemas de clasificación de texto, escalando en complejidad desde métodos estadísticos clásicos hasta modelos generativos modernos. El flujo de trabajo incluye:

* **Pipelines de Preprocesado:** Implementación de tres niveles de limpieza de texto (Mínimo, Básico y Avanzado) para alimentar adecuadamente a cada tipo de arquitectura.
* **Modelos Clásicos (Vectores Dispersos):**
    * **Bag of Words (BoW)** + Clasificador Multinomial Naive Bayes.
    * **TF-IDF** + Regresión Logística.
* **Modelos de Embeddings (Vectores Densos):**
    * **Word2Vec** (`word2vec-google-news-300`) + Random Forest.
    * **BERT** (Extracción de features estáticas) + Perceptrón Multicapa (MLP).
* **Fine-Tuning de Transformers:** Adaptación de pesos de un modelo preentrenado utilizando **DistilBERT** (`distilbert-base-uncased`) para evaluar la mejora de rendimiento al especializar el modelo en nuestro dataset específico.
* **Clasificación con LLMs en local:** Implementación de **Mixtral 8x7B** sin reentrenamiento. Se hace uso de técnicas estrictas de *Prompt Engineering* (paradigma RTF: Rol, Tarea, Formato) evaluando su capacidad de razonamiento en escenarios:
    * *Zero-Shot Prompting*
    * *One-Shot Prompting*
    * *Few-Shot Prompting*

---

**Autor:** Javier Carreño Luque