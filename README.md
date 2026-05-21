# Clasificación y Recuperación de Información de Noticias (Práctica Final)

Este repositorio contiene la implementación de la Práctica Final, enfocada en resolver dos tareas principales de Procesamiento de Lenguaje Natural (NLP) sobre un corpus de noticias.
1. [**Clasificación temática de textos.**](./clasificador/)
2. [**Sistema RAG (Retrieval-Augmented Generation) con un agente conversacional.**](./agente/)

---

## Estructura del Proyecto

```
.
├── agente
│   ├── RAGSystem.ipynb
│   ├── agente.py
│   ├── index.html
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── scripts/
│   ├── src/
│   ├── vector_db/
│   ├── pyproject.toml
│   └── README.md
│
├── clasificador
│   ├── main.ipynb
│   ├── src/
│   ├── scripts/
│   ├── models/
│   ├── pyproject.toml
│   └── README.md
│
├── data/
├── pyproject.toml
├── README.md
└── uv.lock
```

El código está dividido en 2 subcarpetas con el contenido de cada uno de los ejercicios propuestos. Dentro de estos viene más información específica de cada uno de ellos. 

En rasgos generales siguen la misma estructura:
* Carpeta src/ con la definición de funciones necesarias.
* Carpeta scripts/ con código ejecutable necesario para alguna función del código principal.
* pyproject.toml con las dependencias aisldas de cada proyecto.

En el caso del agente, se incluye un DockerFile y Docker-Compose para poder ejecutar el agente conversacional.


## Ejecución del Código

Este repositorio está pensado para ser ejecutado con la herramienta de *uv*. Si no se dispone de esta se puede descargar:

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

En caso de querer ejecutar el código será necesario crear el entorno virtual de manera total o parcial:

``` bash
# Instalción total de todas las dependencias
uv sync

# Instalación parcial de las dependencias de solo el ejercicio 1
uv sync --package clasificador

# Instalación parcial de las dependencias de solo el ejercicio 2
uv sync --package agente
```

---
**Autor:** Javier Carreño Luque