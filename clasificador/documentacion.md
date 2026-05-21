<a id="src"></a>

# src

<a id="src.preprocesado"></a>

# src.preprocesado

<a id="src.preprocesado.limpieza_comun"></a>

#### limpieza\_comun

```python
def limpieza_comun(series_de_texto: pd.Series) -> pd.Series
```

Aplica limpieza común a toda una Serie de Pandas usando métodos vectorizados .str

<a id="src.preprocesado.preprocessing_minimal"></a>

#### preprocessing\_minimal

```python
def preprocessing_minimal(series_de_texto: pd.Series) -> pd.Series
```

Mantiene puntuación, mayúsculas y estructura.
Útil para sistemas más complejos como LLMs y Transformers.

<a id="src.preprocesado.preprocessing_basico"></a>

#### preprocessing\_basico

```python
def preprocessing_basico(series_de_texto: pd.Series) -> pd.Series
```

Limpia el texto convirtiéndolo a minúsculas, eliminando signos de puntuación
y lematizando todas las palabras. Conserva las stopwords.
Útil para TF-IDF o Word2Vec.

<a id="src.preprocesado.preprocessing_avanzado"></a>

#### preprocessing\_avanzado

```python
def preprocessing_avanzado(series_de_texto: pd.Series) -> list
```

Limpia el texto convirtiéndolo todo a minúsculas y eliminando signos de puntuación.
También elimina las palabras que sean stopWords.
Útil para modelos estadísticos básados en frecuencias.

<a id="src.preprocesado.preprocesar_datos"></a>

#### preprocesar\_datos

```python
def preprocesar_datos(df: pd.DataFrame,
                      conjunto: Literal["train", "test"],
                      guardar: bool = False) -> pd.DataFrame
```

Añade columnas con los preprocesados de texto al conjunto df.
'conjunto' puede tomar valores "train" o "test".
Por defecto no guarda el dataframe resultante en almacenamiento persistente.
En caso contrario, lo guarda en PATH_DATA.

<a id="src.preprocesado.cargar_datos"></a>

#### cargar\_datos

```python
def cargar_datos(test: bool = False, guardar: bool = True) -> pd.DataFrame
```

Carga los datos de un path, revisando antes si existe la versión preprocesada.
En caso de no existir, preprocesa los datos y los guarda en 'path_prep'.
Si no se especifica si los datos son los de test o train, se usará train por defecto.
Si no se especifica si se quiere guardar datos preprocesados, por defecto lo hará.

<a id="src.vectorizacion"></a>

# src.vectorizacion

<a id="src.vectorizacion.get_word2vec_model"></a>

#### get\_word2vec\_model

```python
def get_word2vec_model()
```

Carga el modelo Word2Vec si ya existe, o lo descarga si no.
Retorna el modelo cargado.

<a id="src.vectorizacion.BoW"></a>

#### BoW

```python
def BoW(textos: pd.Series, max_features=5000, vectorizador=None)
```

Convierte textos en Bag Of Words.
Si no se pasa vectorizador, asume que es Train: lo crea, lo entrena y lo devuelve.
Si se pasa vectorizador, asume que es Test: solo transforma usando el vocabulario aprendido.

<a id="src.vectorizacion.TF_IDF"></a>

#### TF\_IDF

```python
def TF_IDF(textos: pd.Series, max_features=5000, vectorizador=None)
```

Convierte textos en matriz TF-IDF.
Si no se pasa vectorizador, asume que es Train: lo crea, lo entrena y lo devuelve.
Si se pasa vectorizador, asume que es Test: solo transforma usando el vocabulario aprendido.

<a id="src.vectorizacion.Word2Vec_Embeddings"></a>

#### Word2Vec\_Embeddings

```python
def Word2Vec_Embeddings(textos: pd.Series, modelo_w2v) -> pd.DataFrame
```

Calcula el vector promedio de cada documento usando un modelo Word2Vec preentrenado.
Recibe la serie de textos y el modelo ya cargado en memoria.
Recomendación: Pasar 'desc_basico'.

<a id="src.vectorizacion.BERT_Embeddings"></a>

#### BERT\_Embeddings

```python
def BERT_Embeddings(series_de_texto: pd.Series) -> pd.DataFrame
```

Utiliza BertModel nativo para extraer embeddings contextuales.
Toma el token especial [CLS] como representación semántica del texto completo.

<a id="src.config"></a>

# src.config

<a id="src.config.W2V_MODEL"></a>

#### W2V\_MODEL

Modelo de embeddings a usar para w2v

<a id="src.config.BERT_MODEL"></a>

#### BERT\_MODEL

Modelo de Transformer para clasificar

<a id="src.config.FINE_TUNING_MODEL"></a>

#### FINE\_TUNING\_MODEL

Modelo a fine-tunear

<a id="src.config.FINE_TUNING_PATH"></a>

#### FINE\_TUNING\_PATH

Path donde guardar el modelo fine-tuneado

