
# Clasificación Temática

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
./
├── main.ipynb              # Jupyter Notebook con todo runeado
├── models/                 # Carpeta con los modelos usados para los BERT y Word2Vec
├── scripts/
|   └── fineTunning.py      # Script para el Fine-Tuning del modelo BERT
├── src/
|   ├── config.py           # Configuración de Paths (no hace falta modificarlo, se pueden usar ENV)
|   ├── preprocesado.py     # Funciones de preprocesado
|   └── vectorizacion.py    # Funciones de vectorizaciones densas y esparsas
└── pyproject.toml

../
└── data/                   # Carpeta donde se leen los datos por defecto (En la carpeta padre; raiz del repo)
```

## Ejecución

En caso de querer ejecutarse el código será necesario tener construido el venv del repositorio con al menos los paquetes del workspace 'clasificacion'.

Para ejecutar el notebook, solo es necesario acceder a este desde un IDE de preferencia con extensiones de Jupyter seleccionando el kernel del entorno virtual.

Si se quiere ejecutar el script de fine-tuning:

``` bash
uv run python -m scripts.FineTuning
```

---

**Autor:** Javier Carreño Luque
