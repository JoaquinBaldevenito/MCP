import pandas as pd
import re
from langchain_core.tools import tool

# Variable global para el DataFrame
db = None

def cargar_base_de_datos():
    """Carga el CSV en un DataFrame de Pandas y prepara los datos."""
    global db
    try:
        df = pd.read_csv("products_asos.csv")
        # Renombrar columnas para estandarizar
        df = df.rename(columns={"nombre": "name", "precio": "price", "talle": "size", "color": "color_col"})
        
        # 1. Limpieza de Texto
        df['name'] = df['name'].fillna('').astype(str)
        df['size'] = df['size'].fillna('').astype(str)
        
        # 2. Limpieza de Precio (CRÍTICO para ordenar bien)
        def limpiar_precio(val):
            try:
                if isinstance(val, str):
                    val = val.replace('$', '').replace(',', '').strip()
                    match = re.search(r"(\d+\.?\d*)", val)
                    if match:
                        return float(match.group(1))
                return float(val)
            except:
                return 0.0

        df['price'] = df['price'].apply(limpiar_precio)

        db = df
        print(f"Base de datos cargada con {len(db)} productos.")
        
    except Exception as e:
        print(f"Error fatal cargando DB: {e}")
        # Creamos una DB vacía para no romper el programa
        db = pd.DataFrame(columns=['name', 'price', 'size'])

@tool
def find_products(search_term: str = "", talle: str = None, color: str = None, sort_by_price: str = None) -> dict:
    """
    Busca productos en la base de datos.
    - search_term: Nombre del producto en inglés (ej: 'T-shirt').
    - talle: Filtro exacto de talle (ej: 'S', 'L').
    - color: Filtro de color (busca la palabra en el nombre).
    - sort_by_price: 'asc' (barato) o 'desc' (caro).
    """

    print(f"[Debug: find_products(search='{search_term}', talle='{talle}', color='{color}', sort='{sort_by_price}')]")

    global db
    if db is None or db.empty: 
        return {"status": "Error", "productos": [], "mensaje": "Base de datos vacía."}

    try:
        # Trabajamos sobre una copia
        results = db.copy()

        # 1. FILTRO: Búsqueda de Texto
        if search_term:
            term_limpio = search_term.lower().strip()
            results = results[results['name'].str.lower().str.contains(term_limpio, na=False)]

        # 2. FILTRO: Color
        if color:
            color_limpio = color.lower().strip()
            results = results[results['name'].str.lower().str.contains(color_limpio, na=False)]

        # 3. FILTRO: Talle
        if talle:
            talle_limpio = talle.lower().strip()
            results = results[results['size'].str.lower().str.contains(talle_limpio, na=False)]

        # 4. ORDENAMIENTO (Sorting)
        if sort_by_price:
            es_ascendente = (sort_by_price.lower() == 'asc')
            results = results.sort_values(by='price', ascending=es_ascendente)
        
        # 5. RESULTADO FINAL
        if results.empty:
            return {
                "status": "No encontrado", 
                "productos": [], 
                "mensaje": f"No encontré productos con: {search_term} {color if color else ''} {talle if talle else ''}"
            }

        # Tomamos los top 5 resultados
        top_results = results.head(5)
        
        lista_productos = []
        for _, row in top_results.iterrows():
            lista_productos.append({
                "nombre": row['name'],
                "precio": row['price'],
                "talle_disponible": row['size']
            })
        
        return {"status": "Encontrado", "productos": lista_productos}

    except Exception as e:
        return {"status": "Error", "productos": [], "mensaje": f"Error técnico: {str(e)}"}

@tool
def get_opening_hours() -> str:
    """Devuelve los horarios de atención de la tienda física."""
    return "Nuestro horario es de Lunes a Sábados de 10:00 a 20:00hs."

@tool
def get_location() -> str:
    """Devuelve la ubicación y dirección de la tienda."""
    return "Estamos en Av. Corrientes 1234, Buenos Aires."

@tool
def get_return_policy() -> str:
    """Devuelve la política de cambios y devoluciones."""
    return "Tienes 30 días para realizar cambios. La prenda debe estar sin uso y con etiqueta."

@tool
def get_general_recommendations(topic: str) -> str:
    """
    Brinda recomendaciones generales de moda o uso.
    topic: El tema sobre el cual aconsejar (ej: 'lluvia', 'fiesta', 'talles').
    """
    return f"Para '{topic}', te sugerimos buscar prendas cómodas y verificar la guía de talles en nuestra web."

@tool
def list_sample_products(count: int = 5) -> dict:
    """
    Lista productos al azar para inspirar al usuario.
    count: Cantidad de productos a mostrar (max 10).
    """
    global db
    if db is None: return {"error": "DB off"}
    
    try:
        n = min(int(count), 10)
        sample = db.sample(n)
        lista = []
        for _, row in sample.iterrows():
            lista.append({"nombre": row['name'], "precio": row['price']})
        return {"status": "Muestra", "productos": lista}
    except:
        return {"status": "Error", "productos": []}

@tool
def chat_response(message: str) -> str:
    """
    Úsala SOLO para saludar, despedirse o responder cosas que NO requieren buscar en la base de datos.
    """
    return message