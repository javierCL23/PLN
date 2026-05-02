# Clasificación y recuperación de información de noticias (Práctica Final)

En este repositorio se almacena el código necesario para abordar dos tareas:

1. Una tarea de clasificación temática
2. Un sistema de recuperación de información basado en RAG + un agente sencillo.


## Ejercicio 1.

### Preprocesado de texto

3 Tipos de preprocesado:
* Mínimo: mantiene puntuación, mayúsculas y estructura. Solo limpia cosas básicas necesarias.
* Básico: limpia el texto convirtiéndolo todo a minúsculas y eliminando signos de puntuación.
* Avanzado: hace lo mismo que el básico además de eliminar las palabras que sean stopwords.

Cada uno de estos preprocesados son más útiles para distintos métodos.
* Mínimo es útil para LLMs y Transformers.
* Básico es útil para TF-IDF o Word2Vec.
* Avanzado es útil para modelos estadísticos basados en frecuencias.

### Vectorización (Dispersa vs Densa)

Para las vectorizaciones dispersas se han elegido dos de los métodos más comunes:
* Bag of Words (BoW): para este método es buena idea aplicar un preprocesado avanzado.
* TF-IDF: para este método es buena idea aplicar un preprocesado básico.


---
Autor: Javier Carreño Luque
