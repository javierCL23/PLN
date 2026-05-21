<a id="src"></a>

# src

<a id="src.rag"></a>

# src.rag

<a id="src.rag.obtener_vs"></a>

#### obtener\_vs

```python
def obtener_vs(path: str | None = None) -> Chroma
```

Devuelve un objeto Chroma. 
Lo puede crear desde 0 o tomando un Path

<a id="src.rag.obtener_retriever"></a>

#### obtener\_retriever

```python
def obtener_retriever(k: int = 3,
                      vector_store: Chroma | None = None,
                      path: str | None = None)
```

Devuelve un objeto Retriever configurado con un top-k específico.
Permite pasar un objeto vector_store ya instanciado. 
Si no se pasa, es obligatorio proporcionar un 'path' para cargarlo desde el disco.

<a id="src.rag.generate_vector_database"></a>

#### generate\_vector\_database

```python
def generate_vector_database(df: pd.DataFrame,
                             space: str = "l2",
                             save: bool = True) -> (Chroma | None)
```

Genera una base de datos vectorial usando para ello el conjunto de datos que se le pase.
El formato de los datos debe estar definido con los campos 'description', 'title' y 'label'.

**Arguments**:

- `df` _pd.DataFrame_ - Datos a indexar.
- `space` _str_ - Medida de distancia a usar ('cosine', 'l2', o 'ip').
- `save` _bool_ - Si es True, guarda en disco y omite la generación si ya existe.
  Si es False, genera en memoria saltándose la comprobación y devuelve el objeto.

<a id="src.classifiers"></a>

# src.classifiers

<a id="src.classifiers.predecir_categoria"></a>

#### predecir\_categoria

```python
def predecir_categoria(titulo: str, fallback: bool = True) -> str
```

Dado un título, usa un LLM para predecir la categoría.

<a id="src.classifiers.predecir_categoria_rag"></a>

#### predecir\_categoria\_rag

```python
def predecir_categoria_rag(titulo: str,
                           retriever,
                           fallback: bool = True) -> str
```

Dado un título, recupera las descripciones más relevantes del train
y usa un LLM para predecir la categoría.

<a id="src.classifiers.predecir_categoria_knn"></a>

#### predecir\_categoria\_knn

```python
def predecir_categoria_knn(titulo: str, retriever) -> str
```

Dado un título, recupera las descripciones más relevantes del train
y usa un clasificador basado en KNN para hacer la predicción.

<a id="src.state"></a>

# src.state

<a id="src.state.State"></a>

## State Objects

```python
class State(TypedDict)
```

<a id="src.state.State.tareas_completadas"></a>

#### tareas\_completadas

Todos los workers escriben a la vez sobre esta lista

<a id="src.state.WorkerState"></a>

## WorkerState Objects

```python
class WorkerState(TypedDict)
```

<a id="src.state.WorkerState.tareas_completadas"></a>

#### tareas\_completadas

MISMA que la del State general (overlapping keys en langgraph funciona así)

<a id="src.graph"></a>

# src.graph

<a id="src.graph.orquestador"></a>

#### orquestador

```python
def orquestador(state: State)
```

Orquestador que genera un plan para llevar a cabo la petición del usuario

<a id="src.graph.llm_call"></a>

#### llm\_call

```python
def llm_call(state: WorkerState)
```

Worker ejecuta la tarea designada llamando a las herramientas correspondientes

<a id="src.graph.manejo_sin_tareas"></a>

#### manejo\_sin\_tareas

```python
def manejo_sin_tareas(state: State)
```

Nodo de Fallback

<a id="src.graph.sintetizar_tareas"></a>

#### sintetizar\_tareas

```python
def sintetizar_tareas(state: State)
```

Sintetiza los resultados de las tareas completadas para dar una respuesta final al usuario.

<a id="src.tools"></a>

# src.tools

<a id="src.tools.buscar_noticias"></a>

#### buscar\_noticias

```python
@tool
def buscar_noticias(query: str, k: int = 3) -> str
```

Busca noticias similares en la base de datos (RAG) según la temática.

**Arguments**:

- `query` - La temática, palabras clave o descripción de las noticias a buscar.
- `k` - El número de noticias a devolver. Si el usuario especifica una cantidad,
  usa ese número exacto. Si no especifica ninguno, el valor por defecto es 3.

<a id="src.tools.clasificar_noticia"></a>

#### clasificar\_noticia

```python
@tool
def clasificar_noticia(query: str) -> str
```

Clasifica una noticia gracias a un RAG según las noticias similares recuperadas.

**Arguments**:

  - query: Texto o titular que se buscará en el RAG para encontrar matches similares.
  En base a la clase de los textos recuperados es que se decide la temática del la query.

<a id="src.utils"></a>

# src.utils

<a id="src.utils.preprocesado_minimo"></a>

#### preprocesado\_minimo

```python
def preprocesado_minimo(series_de_texto: pd.Series) -> pd.Series
```

Aplica limpieza común a toda una Serie de Pandas usando métodos vectorizados .str

<a id="src.utils.cargar_datos"></a>

#### cargar\_datos

```python
def cargar_datos(path: Path = PATH_DATA,
                 conjunto: Literal["train", "test"] = "train") -> pd.DataFrame
```

Devuelve el conjunto de datos en forma de dataframe. Permite ajustar el path a una ruta concreta.
Las opciones de tipo de conjunto son: 'train' o 'test'

<a id="src.utils.fallback_postprocessing"></a>

#### fallback\_postprocessing

```python
def fallback_postprocessing(resultado: str) -> str
```

Función para asegurar devolver la categoría concreta.

<a id="src.prompts"></a>

# src.prompts

Módulo de configuración de prompts para el sistema multi-agente.

Este archivo almacena las instrucciones base (system prompts) que definen 
el comportamiento de los diferentes agentes encargados del análisis, 
clasificación y recomendación de noticias.

<a id="src.llm"></a>

# src.llm

<a id="src.config"></a>

# src.config

<a id="src.config.PATH_DATA"></a>

#### PATH\_DATA

Ubicación de los datos

<a id="src.config.PATH_VECTOR_DB"></a>

#### PATH\_VECTOR\_DB

Dirección en la que guardar la vector store en almacenamiento

<a id="src.config.LLM_MODEL"></a>

#### LLM\_MODEL

Modelo general para testing de capacidad predictiva en categorización (granite4:micro por velocidad)

<a id="src.config.EMBEDDING_MODEL"></a>

#### EMBEDDING\_MODEL

'sentence-transformers/msmarco-distilbert-base-v4' Modelo base para embeddings de la Vector Store. Ventaja principal: útil para retrieval con texto corto para texto largo

<a id="src.config.LLM_PROVIDER"></a>

#### LLM\_PROVIDER

Provider elegido para conexión con LLM. Permitido ollama o openai

<a id="src.config.LLM_URL"></a>

#### LLM\_URL

Dirección URL para conexión con el modelo en caso de usar provider ollama

<a id="src.config.ORCHESTRATOR_MODEL"></a>

#### ORCHESTRATOR\_MODEL

Modelo usado por el nodo orquestador del agente (mixtral7:8b overkill para la tarea, granite4:micro es más que suficiente)

<a id="src.config.WORKER_MODEL"></a>

#### WORKER\_MODEL

Modelo usado por los workers del agente (grantie4:micro, solo necesita saber que tiene que usar tools, no necesitan razonamiento más allá de extracción de argumentos)

