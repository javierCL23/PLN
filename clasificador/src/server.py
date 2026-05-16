from mcp.server.fastmcp import FastMCP
from src.preprocesado import *
from src.vectorizacion import *


# Iniciar servidor de MCP
mcp = FastMCP("Clasificador de Texto")

# Creación de tools (TODO)
@mcp.tool()
def clasificar_texto(texto: str) -> str:
    """Preprocesa y clasifica un texto utilizando técnicas de NLP."""
    if not texto.strip():
        return "Error: El texto está vacío."
    try:
	# TODO        
        return "Texto clasificado como positivo"
    except Exception as e:
        return f"Error en la clasificación: {str(e)}"

if __name__ == "__main__":
    # Conexión pensada para uso remoto por envío de mensajes por puerto 5000
    mcp.run(transport="sse", host="0.0.0.0", port=5000)
