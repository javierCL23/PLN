# Clasificación y recuperación de información de noticias (Práctica Final)

En este repositorio se almacena el código necesario para abordar dos tareas:

1. Una tarea de clasificación temática
2. Un sistema de recuperación de información basado en RAG + un agente sencillo.


## Ejercicio 1.

### Preprocesado de texto

3 Tipos de preprocesado:
* **Mínimo:** mantiene puntuación, mayúsculas y estructura. Solo limpia cosas básicas necesarias.
* **Básico:** limpia el texto convirtiéndolo todo a minúsculas y eliminando signos de puntuación.
* **Avanzado:** hace lo mismo que el básico además de eliminar las palabras que sean stopwords.

Cada uno de estos preprocesados son más útiles para distintos métodos.
* Mínimo es útil para LLMs y Transformers.
* Básico es útil para TF-IDF o Word2Vec.
* Avanzado es útil para modelos estadísticos basados en frecuencias.

### Vectorización (Dispersa vs Densa)

Para las vectorizaciones dispersas se han elegido dos de los métodos más comunes:

* **Bag of Words (BoW):** para este método es buena idea aplicar un preprocesado avanzado (eliminación exhaustiva de ruido, lematización, etc.) ya que se basa únicamente en el conteo de frecuencias.
* **TF-IDF:** para este método es buena idea aplicar un preprocesado básico, permitiendo que la propia fórmula matemática penalice las palabras irrelevantes y resalte las más importantes.

Para las vectorizaciones densas se han elegido dos de los métodos más comunes:

* **Word2Vec:** Basado en embeddings estáticos, en este caso 'word2vec-google-news-300' que transforma el texto a un vector denso continuo de tamaño 300.
* **BERT:** Basado en embeddings contextuales gracias a la arquitectura Transformer, para convertir el texto a un vector semántico profundo de tamaño 768.

Cada una de estas vectorizaciones será usada para implementar un tipo diferente de clasificador.

### Clasificadores

Se han explorado tantos tipos de clasificadores como métodos de vectorización, cada uno planteado para aprovechar al máximo el tipo de representación matemática que usa.

En el caso de **BoW**, se ha utilizado un clasificador **Multinomial Naive Bayes**. Este algoritmo probabilístico clásico es la pareja ideal para BoW, ya que está diseñado matemáticamente para trabajar de forma nativa con distribuciones de frecuencias discretas (conteos enteros de palabras), resultando en un modelo extremadamente rápido y eficiente en consumo de memoria.

Para **TF-IDF** se ha usado una **Regresión Logística**. Al aplicar TF-IDF obtenemos una matriz de alta dimensionalidad, dispersa, pero con valores continuos (decimales de "importancia"). Los modelos lineales como la regresión logística son excepcionalmente robustos en estos espacios de alta dimensionalidad, trazando fronteras de decisión claras entre temáticas y siendo poco propensos al overfitting.

Por otro lado, en **Word2Vec** se aumenta la complejidad usando un **Random Forest**. Al pasar a un espacio vectorial denso de 300 dimensiones, ya no tenemos palabras aisladas, sino coordenadas latentes. Los algoritmos no lineales basados en árboles son excelentes para encontrar agrupaciones semánticas de manera eficaz.

Finalmente, para el embedding de **BERT** se añade un **MLP** aprovechando la profundidad semántica de sus vectores. Al conectar una pequeña red neuronal densa directamente a las 768 dimensiones de BERT, simulamos con gran exactitud el comportamiento de la "cabeza de clasificación" de un Transformer original, permitiendo que la red extraiga las relaciones contextuales capturadas por el modelo de lenguaje para separar las cuatro categorías con precisión.

### Fine-Tuning con Transformers

Para la fase final del proyecto, se ha implementado un proceso completo de Fine-Tuning utilizando **DistilBERT** (distilbert-base-uncased).

El objetivo de esta sección es demostrar el salto de calidad que supone adaptar los pesos de un modelo de lenguaje preentrenado específicamente a nuestra tarea, en lugar de usarlo únicamente como un extractor de características estático (como se hizo en la sección de vectorización densa).

Se ha elegido la variante destilada de BERT por motivos de eficiencia computacional. Al reducir el número de parámetros en un 40%, conseguimos tiempos de entrenamiento significativamente más rápidos y un menor consumo de memoria VRAM, reteniendo a su vez el gran parte de las capacidades de comprensión del lenguaje del modelo original.

Este enfoque permite comparar directamente el rendimiento de un modelo profundo "ligero" pero especializado en los datos de AG News, frente a las arquitecturas clásicas y al uso de un BERT completo sin fine-tuning.

---
Autor: Javier Carreño Luque
