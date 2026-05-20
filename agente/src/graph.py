from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from src.state import State,WorkerState
from src.state import Task, Tasks
from src.llm import llm_tasker, llm_recommendator

from src.tools import buscar_noticias
from src.prompts import ORCHESTRATOR_PROMPT, RECOMENDATOR_PROMPT
# Nodos
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
    
    # Recomendación
    if tarea.task == "news_recommendation":
        print("Ejecutando tarea de recomendación...")
        # Responde con intención de llamar a una tool en base al prompt
        respuesta_llm = llm_recommendator.invoke([
            SystemMessage(RECOMENDATOR_PROMPT),
            HumanMessage(content=tarea.description)
        ])
        resultado_final = ""
        if respuesta_llm.tool_calls:
            for tool_call in respuesta_llm.tool_calls:
                # Query y K
                args = tool_call["args"] 
                
                # Invocamos la función de Python real con esos argumentos
                resultado_tool = buscar_noticias.invoke(args)
                resultado_final += f"Recomendaciones encontradas:\n{resultado_tool}"
        else:
            resultado_final = "Error: El agente no pudo formular la búsqueda en el RAG."
            
        # Añadir a la lista compartida de completadas
        return {"tareas_completadas": [resultado_final]}

    # Clasificación
    elif tarea.task == "classification":
        print("Ejecutando tarea de clasificación...")
        return {"tareas_completadas": ["Lógica de clasificación pendiente de implementar."]}


# Nodo especial en caso de no haber tareas
def manejo_sin_tareas(state: State):
    """Nodo de Fallback"""
    mensaje = "No he detectado ninguna petición de clasificación o recomendación en tu mensaje."
    return {"output": mensaje, "tareas_completadas": [mensaje]}

# Nodo enrutador
def enrutar_tareas(state: State):
    tareas = state.get("tareas", [])
    if not tareas:
        return "manejo_sin_tareas"
    
    enviados = []
    for tarea in tareas:
        enviados.append(Send("llm_call", {"task": tarea}))
    return enviados