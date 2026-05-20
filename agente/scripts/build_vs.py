from src.utils import cargar_datos
from src.rag import generate_vector_database


def main():
    # Importar datos
    df = cargar_datos()

    # Creación de la base de datos
    generate_vector_database(df)

if __name__ == "__main__":
    main()