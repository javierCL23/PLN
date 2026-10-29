
# Sistema RAG y agente de IA

#### Sistema RAG

En esta sección se detalla la implementación de un sistema de Generación Aumentada por Recuperación (RAG) basado en las descripciones de los titulares de noticias.

Para la construcción de la base de datos vectorial, se procesó el conjunto de entrenamiento (train) en su totalidad, tratando cada descripción como un documento individual. Con el objetivo de garantizar una exploración robusta y aprovechar al máximo la información, se enriquecieron los metadatos de cada documento incluyendo:
* Titular original y la clase/categoría a la que pertenecen (para maximizar el contexto temático en la recuperación y habilitar el uso de modelos estadísticos clásicos como K-Nearest Neighbors).
* Índice original de la muestra (para asegurar la trazabilidad completa de los datos).

Esto se hizo con el fin de poder agregar la mayor cantidad temática posible al retrieval y mejorar con ello la predicción del modelo, o dar opción a usar un modelo estadístico/clásico como puede serlo un K-NN.

Además, por temas de trazabilidad, se añadió adicionalmente el índice de la muestra a la que pertenece una descripción.

El proceso de recuperación busca identificar los k documentos más cercanos a un titular de prueba (test) evaluando las siguientes métricas de distancia:
* Distancia L2 al cuadrado (L² Squared): Mide la distancia geométrica absoluta entre los vectores (norma-L² al cuadrado).
* Producto Interno (IP): Evalúa la alineación y magnitud combinada de los vectores.
* Similitud de Coseno: Mide exclusivamente el ángulo entre los vectores, ignorando sus magnitudes.

Para validar la calidad del sistema RAG, se realizarán tres niveles de comparación basados en la métrica de accuracy:

| Evaluación                | Descripción                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Comparación intra K-NN    | Se evalúa el impacto de las distintas métricas de distancia en el accuracy de la predicción del clasificador K-NN.        |
| Comparación entre modelos | Se contrasta el rendimiento del modelo clásico K-NN frente a la capacidad predictiva de un LLM con información aumentada. |
| Comparación intra LLM     | Se analiza el beneficio real del RAG comparando el rendimiento del mismo LLM con y sin contexto aumentado.                |

#### Agente de IA

En esta sección se detalla la implementación de un agente conversacional básico sin memoria conversacional, diseñado para interactuar con el usuario y cumplir dos funciones principales:
1. Clasificación de titulares: Evaluar y categorizar títulos de noticias en base a cuatro clases predefinidas: World, Sports, Business y Sci/Tech.
2. Sistema de recomendación: Sugerir noticias similares a partir de una descripción o temática introducida por el usuario, apoyándose en el sistema RAG previamente indexado.

La capacidad de resolución del agente se sustenta en el acceso estructurado al sistema RAG. Para la tarea de clasificación, el agente emplea la información recuperada como contexto de apoyo para fundamentar su toma de decisiones. Por su parte, en la tarea de recomendación, el acceso al RAG resulta indispensable para extraer titulares genuinos y mitigar cualquier riesgo de alucinación por parte del modelo.

Para gestionar este flujo de ejecución, la interacción con el sistema RAG no es directa, sino que se expone mediante herramientas (Tools) que el agente puede invocar según lo necesite. Este diseño incorpora además un mecanismo de control de flujo clave:
* Tool de Recuperación (RAG): Permite al LLM consultar la base de datos vectorial.
* Tool de Clasificación Clásica: Ofrece la posibilidad (ya sea por decisión del sistema o elección del usuario) de delegar la predicción. De este modo, se puede confiar en el criterio analítico del LLM, o invocar el modelo estadístico clásico entrenado previamente para determinar la categoría final.

Esta implementación está basada en este ejemplo de [patrón de orquestación](https://youtu.be/aHCDrAbH_go?si=JG6OjMfZtm-obUcg&t=840).

## Estructura del código

**Estructura principal**:
```
.
├── agente.py               # Punto de entrada principal para ejecutar el agente conversacional (API).
├── index.html              # Interfaz web básica de usuario para interactuar con el agente.
├── docker-compose.yml      # Configuración de servicios Docker (levantar la app y servicio de Ollama).
├── Dockerfile              # Imagen con la aplicación para su uso con docker-compose.
├── pyproject.toml
├── RAGSystem.ipynb         # Notebook con la parte 1 del ejercicio.
├── README.md
├── scripts
│   └── build_vs.py         # Script de ingesta: lee los datos, genera embeddings y construye la base de datos vectorial.
├── src                     # Código fuente principal de la aplicación.
│   ├── __init__.py
│   ├── classifiers.py      # Contiene la lógica de los modelos para predecir las categorías.
│   ├── config.py           # Variables de entorno, rutas a directorios y configuraciones globales.
│   ├── graph.py            # Definición de los nodos con el flujo de LangGraph (orquestador, workers, sintetizador y enrutamiento).
│   ├── llm.py              # Inicialización del LLM, parámetros del modelo y configuración de salidas estructuradas.
│   ├── prompts.py          # Centralización de los SystemMessages y directrices para el agente.
│   ├── rag.py              # Lógica interna de creación/conexión con el VectorStore y Retriever.
│   ├── state.py            # Definición de los esquemas de datos (State, WorkerState, Task, Tasks).
│   ├── tools.py            # Definición de las @tools (buscar_noticias, clasificar_noticia) que los workers invocan.
│   └── utils.py            # Funciones auxiliares genéricas (limpieza de texto, carga de datos...).
└── vector_db/              # Directorio local donde se persisten los índices de la base de datos vectorial (Chroma).

../
└── data/                   # Carpeta (en la raíz del repositorio) que almacena los datasets originales de noticias.
```

## Ejecución

El código de este segundo ejercicio está dividido en dos. 

Por un lado, la implementación del RAG con todas las comparativas realizadas se encuentran en el Notebook 'RAGSystem.ipynb'.

Está planteado de esta manera para poder visualizar de manera rápida todas las comparativas con los resultados comentados sin tener que cambiar entre memoria y código constantemente o ejecutar procesos tan demandantes como los relacionados con el uso de Transformers.

Y por otro lado, el agente está pensado para ser usado en un entorno aislado con Docker-compose. Por un lado se tiene la imagen con el código del agente, y por otro lado la imágen con ollama, que es el punto de entrada al modelo LLM.

Este segundo está planteado de esta manera para imitar una puesta en marcha real donde se tenga el sistema mínimamente distribuido.

Para ejecutarlo, se debe ejecutar únicamente:
``` bash
# En caso de usar docker
docker-compose up

# En caso de usar podman
podman-compose up
```
Si se quiere crear la imagen de manera independiente al docker-compose:

``` bash
podman build -t NOMBRE_IMAGEN .
```

Y desde un navegador web entrar a este [enlace](http://127.0.0.1:9090/)

> **Warning**
> Para que funcione todo correctamente, debe existir un `vector_db` dentro de esta misma carpeta.
> Para ello, se puede ejecutar el script build_vs.py
``` bash
uv run python -m scripts.build_vs
```

Otra forma de usar el agente pero con modelos de OpenAI sería ejecutando el contenedor con las flags adecuadas:
``` bash
docker run -d \
  --name agente_ai \
  -e LLM_PROVIDER="openai" \
  -e OPENAI_API_KEY="API_KEY_HERE" \
  -e ORCHESTRATOR_MODEL="gpt-4o" \
  -e WORKER_MODEL="gpt-4o-mini" \
  NOMBRE_IMAGEN

# También se puede invocar con un archivo .env en el que se definan estas
docker run -d --name agente_ai --env-file .env NOMBRE_IMAGEN
```
---

**Autor:** Javier Carreño Luque
