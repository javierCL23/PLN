import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import orquestador,llm_call, manejo_sin_tareas, enrutar_tareas, sintetizar_tareas
from langgraph.graph import StateGraph, START, END
from src.state import State


workflow = StateGraph(State)

# Nodos
workflow.add_node("orquestador", orquestador)
workflow.add_node("manejo_sin_tareas", manejo_sin_tareas)
workflow.add_node("llm_call", llm_call)
workflow.add_node("sintetizador",sintetizar_tareas)

# Aristas
workflow.add_edge(START, "orquestador")
workflow.add_conditional_edges(
    "orquestador",
    enrutar_tareas,
    ["llm_call", "manejo_sin_tareas"]
)
workflow.add_edge("llm_call", "sintetizador")

# Flujo finalizado
workflow.add_edge("manejo_sin_tareas", END) # Si no hay tareas, devuelve mensaje de aviso.
workflow.add_edge("sintetizador",END) # Si hay tareas, un LLM resume todas las tareas realizadas

app = workflow.compile()


# Código para el servidor
app_api = FastAPI(title="Agente de Noticias API")

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    mensaje: str

@app_api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Recibe el mensaje del frontend y devuelve el flujo de ejecución paso a paso."""
    
    async def generar_stream():
        estado_inicial = {"input": request.mensaje}
        estado_final = None
        
        # 1. Ejecutamos el grafo en modo stream
        for event in app.stream(estado_inicial):
            for nombre_nodo, actualizaciones in event.items():
                
                # Enviamos al frontend qué nodo se acaba de ejecutar
                yield json.dumps({
                    "tipo": "progreso", 
                    "contenido": f"Nodo ejecutado: {nombre_nodo}"
                }) + "\n"
                
                estado_final = actualizaciones
                
        # 2. Cuando termina el bucle, enviamos el resultado final
        if estado_final and "output" in estado_final:
            yield json.dumps({
                "tipo": "resultado", 
                "contenido": estado_final["output"]
            }) + "\n"
        else:
            yield json.dumps({
                "tipo": "error", 
                "contenido": "Flujo terminado sin respuesta final."
            }) + "\n"

    # Devolvemos la respuesta como un flujo continuo (NDJSON)
    return StreamingResponse(generar_stream(), media_type="application/x-ndjson")

@app_api.get("/")
async def servir_web():
    """Sirve la interfaz web al entrar a la raíz del servidor."""
    return FileResponse("index.html")