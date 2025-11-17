import pandas as pd
from langchain_core.tools import tool

# Variable global para simular la base de datos
db = None

def cargar_base_de_datos():
    """Carga el CSV en un DataFrame de Pandas."""
    global db
    try:
        df = pd.read_csv("products_asos.csv")
        # Renombramos para que coincida con los datos
        df = df.rename(columns={"nombre": "name", "precio": "price", "talle": "size"})
        # Usamos el 'name' (nombre del producto) como índice para buscar rápido
        df = df.set_index("name")
        print(f"Base de datos cargada con {len(df)} productos.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'products_asos.csv'.")
        exit()
    except Exception as e:
        print(f"Error al cargar la base de datos: {e}")
        exit()

# --- ¡NUEVA HERRAMIENTA UNIFICADA DE BÚSQUEDA! ---
@tool
def find_products(search_term: str = None, talle: str = None, sort_by_price: str = None) -> dict:
    """
    Busca productos en el catálogo.
    Puede filtrar por término de búsqueda (search_term) y talle.
    Puede ordenar por precio (sort_by_price='asc' o 'desc').
    """
    print(f"[Debug: Ejecutando find_products(search={search_term}, talle={talle}, sort={sort_by_price})]")
    global db
    if db is None:
        return {"error": "La base de datos no está cargada."}

    try:
        results_df = db.copy()

        # 1. Filtrar por término de búsqueda (search_term)
        if search_term:
            # Limpiar el término y buscarlo (case-insensitive)
            term_limpio = search_term.strip().strip('"').strip("'")
            results_df = results_df[results_df.index.str.contains(term_limpio, case=False)]

        # 2. Filtrar por talle
        if talle:
            talle_limpio = talle.strip().strip('"').strip("'")
            # Filtra solo los que contienen el talle
            results_df = results_df[results_df['size'].str.contains(talle_limpio, case=False, na=False)]

        # 3. Verificar si no hay resultados
        if results_df.empty:
            return {"status": "No encontrado", "message": f"No se encontraron productos para '{search_term}'"}

        # 4. Ordenar por precio
        if sort_by_price == 'asc':
            results_df = results_df.sort_values(by='price', ascending=True)
        elif sort_by_price == 'desc':
            results_df = results_df.sort_values(by='price', ascending=False)
        
        # 5. Devolver los 3 mejores resultados
        top_results = results_df.head(3)
        productos = []
        for index, row in top_results.iterrows():
            productos.append({
                "nombre": index,
                "precio": row['price'],
                "talles_disponibles": row['size']
            })
        
        return {"status": "Encontrado", "productos": productos}

    except Exception as e:
        print(f"[Debug: ERROR INTERNO find_products] {str(e)}")
        return {"error": f"Error interno en la base de datos: {str(e)}"}


# --- Herramientas Estáticas (¡con el decorador @tool!) ---

@tool
def get_opening_hours() -> str:
    """Devuelve los horarios de atención al público del negocio."""
    print("[Debug: Ejecutando get_opening_hours]")
    return "Estamos abiertos de lunes a viernes de 9:00 a 18:00 hs, y los sábados de 9:00 a 13:00 hs."

@tool
def get_location() -> str:
    """Devuelve la dirección física y la ubicación del negocio."""
    print("[Debug: Ejecutando get_location]")
    return "Nos podés encontrar en Av. Rivadavia 123. También podés ver el mapa en nuestro sitio web: www.mi-negocio.com/ubicacion"

@tool
def get_general_recommendations(topic: str) -> str:
    """Da recomendaciones generales sobre un tema, como 'abrigo', 'lluvia', 'cuidado', etc."""
    print(f"[Debug: Ejecutando get_general_recommendations(topic={topic})]")
    topic_lower = topic.lower()
    if "abrigo" in topic_lower or "frio" in topic_lower:
        return "Para el frío, te recomendamos un 'wool coat' (abrigo de lana) o una 'biker jacket' (campera de cuero) para un look más canchero."
    elif "lluvia" in topic_lower:
        return "Para la lluvia, lo mejor es un 'trench coat', ya que la tela repele el agua y te mantiene seco."
    else:
        return "Te recomendamos siempre revisar la etiqueta de 'Look After Me' (Cuidado) en la descripción del producto antes de lavarlo."

@tool
def list_sample_products(count: int = 5) -> dict:
    """Obtiene una lista de 'count' nombres de productos al azar del catálogo."""
    print(f"[Debug: Ejecutando list_sample_products(count={count})]")
    global db
    try:
        safe_count = int(count)
        if safe_count > 10: safe_count = 10
        
        sample_names = db.sample(n=safe_count).index.tolist()
        return {"productos_ejemplo": sample_names}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_return_policy() -> str:
    """Devuelve la política de devoluciones."""
    print("[Debug: Ejecutando get_return_policy]")
    return "Tenés 30 días para cambiar cualquier producto, siempre que esté en las mismas condiciones en que lo recibiste y con la etiqueta puesta."

@tool
def chat_response(message: str) -> str:
    """Se usa para responder a saludos, despedidas o chat general que no requiere una herramienta."""
    print(f"[Debug: Ejecutando chat_response(message={message})]")
    # Esta herramienta en realidad no se ejecutará, el LLM debería
    # usar el 'message' para responder directamente.
    # Pero necesitamos definirla para que LangChain la conozca.
    return message