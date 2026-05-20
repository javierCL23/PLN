
# Ejercicio 2: Sistema RAG y agente de IA

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

## Estructura del código

**Estructura principal**:


## Ejecución

El código de este segundo ejercicio está dividido en dos. 

Por un lado, la implementación del RAG con todas las comparativas realizadas se encuentran en el Notebook 'RAGSystem.ipynb'.

Está planteado de esta manera para poder visualizar de manera rápida todas las comparativas con los resultados comentados sin tener que cambiar entre memoria y código constantemente o ejecutar procesos tan demandantes como los relacionados con el uso de Transformers.

Y por otro lado, el agente está pensado para ser usado en un entorno aislado con Docker-compose. Por un lado se tiene la imagen con el código del agente, y por otro lado las imágenes tanto del servidor al que se hacen las solicitudes a las Tools por medio de MCP (Model Context Protocol) y la imágen con el ollama, que es el punto de entrada al modelo LLM.

Este segundo está planteado de esta manera para imitar al máximo una puesta en marcha real donde se tenga el sistema lo mejor distribuido posible para así evitar sobrecargar un mismo ordenador/servidor/Instancia.

---

**Autor:** Javier Carreño Luque