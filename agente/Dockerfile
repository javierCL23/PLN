FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Descargar dependencias
COPY pyproject.toml ./

RUN uv venv /app/.venv
RUN uv pip install --python /app/.venv .


# Copiar código
COPY agente.py index.html ./
COPY src/ ./src/

EXPOSE 9090

# Usamos la ruta ABSOLUTA al Python del entorno virtual. 
# Esto fuerza a que use las librerías de /app/.venv pase lo que pase.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "agente:app_api", "--host", "0.0.0.0", "--port", "9090"]