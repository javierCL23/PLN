
# Ejercicio 1: Clasificación Temática

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


## Estructura del código

**Estructura principal**:
```
├── main.ipynb
├── Dockerfile
├── data/
├── models/
├── scripts/
|   └── fineTunning.py
├── src/
|   ├── config.py
|   ├── preprocesado.py
|   ├── vectorizacion.py
|   └── server.py
└── pyproject.toml
```

El código de este ejercicio está pensado para tener dos funciones:

1. Función del ejercicio normal, es decir, todo lo explicado anteriormente. Para visualizar todo lo hecho se puede revisar el main.ipynb y desde ahí, las funciones desarrolladas en el src. (a excepción del server.py)
2. Función de servidor MCP para el uso de herramientas descentralizadas del agente del ejercicio 2. Para este segundo caso de uso es necesario revisar únicamente el server.py y la imagen de Docker.
---

**Autor:** Javier Carreño Luque