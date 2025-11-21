import pandas as pd
from langchain_core.tools import tool

# Variable global
db = None

def cargar_base_de_datos():
    """Carga el CSV en un DataFrame de Pandas."""
    global db
    try:
        df = pd.read_csv("products_asos.csv")
        df = df.rename(columns={"nombre": "name", "precio": "price", "talle": "size"})
        
        # Sanitizar datos (Evita el error de NaN)
        df['name'] = df['name'].fillna('').astype(str)
        df['size'] = df['size'].fillna('').astype(str)
        
        df = df.set_index("name")
        db = df
        print(f"Base de datos cargada con {len(db)} productos.")
    except Exception as e:
        print(f"Error fatal DB: {e}")
        exit()

@tool
def find_products(search_term: str = None, talle: str = None, sort_by_price: str = None) -> dict:
    """
    Busca productos en el catálogo usando palabras clave.
    search_term: Palabras a buscar (ej: 'black jeans').
    talle: Talle a filtrar.
    sort_by_price: 'asc' o 'desc'.
    """
    print(f"[Debug: Ejecutando find_products(search={search_term}, talle={talle})]")
    global db
    if db is None: return {"error": "DB no cargada."}

    try:
        results_df = db.copy()

        # 1. BÚSQUEDA INTELIGENTE (Palabras por separado)
        if search_term:
            term_limpio = search_term.strip().strip('"').strip("'")
            # Dividimos "black jeans" en ["black", "jeans"]
            palabras = term_limpio.split()
            
            # Filtramos: El nombre debe contener TODAS las palabras
            for palabra in palabras:
                results_df = results_df[results_df.index.str.contains(palabra, case=False)]

        # 2. Filtrar por talle
        if talle:
            talle_limpio = talle.strip().strip('"').strip("'")
            results_df = results_df[results_df['size'].str.contains(talle_limpio, case=False)]

        # 3. Resultados
        if results_df.empty:
            return {"status": "No encontrado", "message": f"No encontré productos que tengan todas las palabras: '{search_term}'"}

        # 4. Ordenar y Limpiar duplicados
        if sort_by_price == 'asc': results_df = results_df.sort_values('price', ascending=True)
        elif sort_by_price == 'desc': results_df = results_df.sort_values('price', ascending=False)
        
        # Eliminar duplicados por nombre para no mostrar el mismo producto 5 veces
        results_df = results_df.reset_index().drop_duplicates(subset='name').set_index('name')
        
        top_results = results_df.head(3)
        
        productos = []
        for index, row in top_results.iterrows():
            productos.append({"nombre": index, "precio": row['price']})
        
        return {"status": "Encontrado", "productos": productos}

    except Exception as e:
        return {"error": f"Error técnico: {str(e)}"}

@tool
def get_opening_hours() -> str:
    """Devuelve los horarios de atención."""
    return "Lunes a Viernes 9-18hs, Sábados 9-13hs."

@tool
def get_location() -> str:
    """Devuelve la dirección del local."""
    return "Av. Rivadavia 123."

@tool
def get_general_recommendations(topic: str) -> str:
    """Da recomendaciones generales sobre ropa."""
    return f"Para '{topic}', te recomendamos buscar materiales de calidad y revisar nuestra tabla de talles."

@tool
def list_sample_products(count: int = 5) -> str: # <--- Cambiamos a str
    """Obtiene una lista de 'count' nombres de productos al azar del catálogo."""
    print(f"[Debug: Ejecutando list_sample_products(count={count})]")
    global db
    
    if db is None:
        return "Error: La base de datos no está cargada."

    try:
        safe_count = int(count)
        if safe_count > 10: safe_count = 10
        
        sample = db.sample(n=safe_count)
        
        # --- FORMATO LISTO PARA LEER ---
        txt = "Aquí tienes algunos ejemplos de nuestro catálogo:\n"
        for name, row in sample.iterrows():
            txt += f"• {name} (${row['price']})\n"
            
        return txt
        
    except Exception as e:
        return f"Error al muestrear: {str(e)}"

@tool
def get_return_policy() -> str:
    """Devuelve la política de cambios y devoluciones."""
    return "30 días para cambios con etiqueta."

@tool
def chat_response(message: str) -> str:
    """Herramienta para charlas generales."""
    return message