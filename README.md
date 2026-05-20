# Clasificación y Recuperación de Información de Noticias (Práctica Final)

Este repositorio contiene la implementación de la Práctica Final, enfocada en resolver dos tareas principales de Procesamiento de Lenguaje Natural (NLP) sobre un corpus de noticias.
1. **Clasificación temática de textos.**
2. **Sistema RAG (Retrieval-Augmented Generation) con un agente conversacional.**

---

## Estructura del Proyecto

## Ejecución del Código

Este repositorio está pensado para ser ejecutado con la herramienta de *uv*. Si no se dispone de esta se puede descargar:

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

En caso de querer ejecutar el código del ejercicio 1 será necesario crear el entorno virtual de manera total o parcial:

``` bash
# Instalción total de todas las dependencias
uv sync

# Instalación parcial de las dependencias de solo el ejercicio 1
uv sync --package clasificador

# Instalación parcial de las dependencias de solo el ejercicio 2
uv sync --package agente
```

**Autor:** Javier Carreño Luque