import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import Literal


class Task(BaseModel):
    task: Literal["classification","news_recommendation"] = Field(description="El tipo de acción a realizar")
    description: str = Field(description="Instrucción clara y autocontenida de la tarea.")

class Tasks(BaseModel):
    tasks: list[Task] = Field(description="Lista de tareas extraídas de la petición del usuario.")


class State(TypedDict):
    input: str
    tareas: list[Task]
    tareas_completadas: Annotated[list, operator.add] # Todos los workers escriben a la vez sobre esta lista
    output: str

class WorkerState(TypedDict):
    task: Task
    tareas_completadas: Annotated[list, operator.add] # MISMA que la del State general (overlapping keys en langgraph funciona así)



