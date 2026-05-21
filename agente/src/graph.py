from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from .state import State,WorkerState
from .llm import llm_tasker, llm_recommendator, llm_clasificator, llm_synthesizer

from .tools import buscar_noticias, clasificar_noticia
from .prompts import ORCHESTRATOR_PROMPT, RECOMENDATOR_PROMPT, CLASIFICATOR_PROMPT, SYNTHESIZER_PROMPT


# Nodo encargado de asignar tareas
def orquestador(state: State):
    """Orquestador que genera un plan para llevar a cabo la petición del usuario"""

    # Genera tareas
    tareas = llm_tasker.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=state["input"])
    ])

    # Asigna al campo tareas del state el listado de tasks de la clase Tasks que devuelve el llm
    return {"tareas": tareas.tasks}

# Nodos que ejecutan tools
def llm_call(state: WorkerState):
    """Worker ejecuta la tarea designada llamando a las herramientas correspondientes"""
    tarea = state["task"]
    
    if tarea.task == "news_recommendation":
        config = {
            "nombre": "recomendación",
            "llm": llm_recommendator,
            "prompt": RECOMENDATOR_PROMPT,
            "tool": buscar_noticias,
            "prefijo": "Recomendaciones encontradas:\n"
        }
    elif tarea.task == "classification":
        config = {
            "nombre": "clasificación",
            "llm": llm_clasificator,
            "prompt": CLASIFICATOR_PROMPT,
            "tool": clasificar_noticia,
            "prefijo": "Clase: "
        }
    
    print(f"Ejecutando tarea de {config['nombre']}...")
    respuesta_llm = config["llm"].invoke([
        SystemMessage(config["prompt"]),
        HumanMessage(content=tarea.description)
    ])
    
    resultado_parcial = ""
    
    if respuesta_llm.tool_calls:
        for tool_call in respuesta_llm.tool_calls:
            print("LLAMADA A TOOL:", tool_call)
            args = tool_call["args"] 

            try:                
                resultado_tool = config["tool"].invoke(args)
                resultado_parcial += f"{config['prefijo']}{resultado_tool}\n"
            except Exception as e:
                print(f"Error interno en tool: {e}")
                resultado_parcial += "Error: Hubo un error inesperado al ejecutar la Tool.\n"
    else:
        resultado_parcial = "Error: El agente no pudo formular la búsqueda en el RAG."
    
    texto_completado = (
        f"Descripción de la tarea: {tarea.description}\n"
        f"Resultado:\n{resultado_parcial.strip()}"
    )
    
    return {"tareas_completadas": [texto_completado]}

# Nodo especial en caso de no haber tareas
def manejo_sin_tareas(state: State):
    """Nodo de Fallback"""
    mensaje = "No he detectado ninguna petición de clasificación o recomendación de noticias en tu mensaje."
    return {"output": mensaje}

# Nodo enrutador
def enrutar_tareas(state: State):
    tareas = state.get("tareas", [])
    if not tareas:
        return "manejo_sin_tareas"
    
    enviados = []
    for tarea in tareas:
        enviados.append(Send("llm_call", {"task": tarea}))
    return enviados

# Nodo sintetizador
def sintetizar_tareas(state: State):
    """Sintetiza los resultados de las tareas completadas para dar una respuesta final al usuario."""
    
    peticion_original = state.get("input", "")
    tareas_completadas = state.get("tareas_completadas", [])
    
    # Unión de las tareas completadas
    bloques_texto = []
    for i, resultado in enumerate(tareas_completadas, start=1):
        bloques_texto.append(f"--- Tarea {i} ---\n{resultado}")
    resultados_texto = "\n\n".join(bloques_texto)

    # Prompt final
    mensaje_humano = (
        f"Petición original del usuario: '{peticion_original}'\n\n"
        f"Resultados de los agentes intermedios:\n{resultados_texto}"
    )
    
    respuesta_llm = llm_synthesizer.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=mensaje_humano)
    ])
    
    return {"output": respuesta_llm.content}