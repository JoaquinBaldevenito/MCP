import pandas as pd

# Variable global para simular la base de datos
db = None

def cargar_base_de_datos():
    """Carga el CSV en un DataFrame de Pandas."""

    global db
    df = pd.read_csv("products_asos.csv")
    # Usamos el 'nombre' como índice para buscar rápido
    db = df.set_index("nombre")
    print(f"Base de datos cargada con {len(db)} productos.")


def get_price(producto: str) -> dict:
    """Obtiene el precio de un producto específico."""

    print(f"[Debug: Ejecutando get_price({producto})]")    
    global db # Para acceder a la base de datos
    
    try:
        # 1. Limpiar el string que viene del LLM
        #    (quitamos espacios al inicio/final y comillas que pueda agregar)
        producto_limpio = producto.strip().strip('"').strip("'")
        
        # 2. Intentar un match exacto (case-insensitive)
        match_exacto = db[db.index.str.lower() == producto_limpio.lower()]
        
        if not match_exacto.empty:
            item = match_exacto.iloc[0] # Tomamos el primero
            print("[Debug: Encontrado por match exacto]")
            return {"producto": item.name, "precio": item["precio"]}

        # 3. Si no hay match exacto, probar con "contains" (como antes)
        match_parcial = db[db.index.str.contains(producto_limpio, case=False)]
        
        if not match_parcial.empty:
            item = match_parcial.iloc[0] # Tomamos el primero
            print("[Debug: Encontrado por match parcial]")
            return {"producto": item.name, "precio": item["precio"]}
        
        # 4. Si nada funciona, es un error
        return {"error": f"Producto no encontrado: '{producto_limpio}'"}
        
    except Exception as e:
        # Damos un error más específico si Pandas falla
        print(f"[Debug: ERROR INTERNO get_price] {str(e)}")
        return {"error": f"Error interno en la base de datos: {str(e)}"}


def check_stock(producto: str, talle: str) -> dict:
    """Verifica el talle de un producto específico."""
    
    print(f"[Debug: Ejecutando check_stock({producto}, {talle})]")
    
    global db
    
    try:
        # Limpiar strings
        producto_limpio = producto.strip().strip('"').strip("'")
        talle_limpio = talle.strip().strip('"').strip("'")
        
        # Intentar match exacto
        match_exacto = db[db.index.str.lower() == producto_limpio.lower()]
        
        item_encontrado = None
        if not match_exacto.empty:
            item_encontrado = match_exacto.iloc[0]
            print("[Debug: Encontrado por match exacto]")
        else:
            # Si no, probar con "contains"
            match_parcial = db[db.index.str.contains(producto_limpio, case=False)]
            if not match_parcial.empty:
                item_encontrado = match_parcial.iloc[0]
                print("[Debug: Encontrado por match parcial]")
        
        if item_encontrado is not None:
            
            talles_del_producto = item_encontrado["talle"]
            
            # Chequeo de seguridad: ¿Está vacía la columna de talle? (NaN)
            # (pd.isna es la forma correcta de chequear esto)
            if pd.isna(talles_del_producto):
                return {"producto": item_encontrado.name, "talle": talle_limpio, "disponible": False, "motivo": "Este producto no tiene talles listados."}

            # La lógica de 'talle in str(talles)' es simple y funciona
            if talle_limpio.lower() in str(talles_del_producto).lower():
                return {"producto": item_encontrado.name, "talle": talle_limpio, "disponible": True}
            else:
                return {"producto": item_encontrado.name, "talle": talle_limpio, "disponible": False, "motivo": "Talle no disponible en la lista."}
        
        else:
            # Si 'item_encontrado' sigue siendo None, no lo encontramos
            return {"error": f"Producto no encontrado: '{producto_limpio}'"}
            
    except Exception as e:
        print(f"[Debug: ERROR INTERNO check_stock] {str(e)}")
        return {"error": f"Error interno en la base de datos: {str(e)}"}



def list_sample_products(count: float = 10.0) -> str:
    """
    Obtiene una lista de 'count' nombres de productos al azar del catálogo para
    mostrarle ejemplos al usuario.
    
    Esta función convierte 'count' a int, ya que el LLM puede pasarlo como float.
    """
    
    print(f"[Debug: Ejecutando list_sample_products(count={count})]")
    
    try:
        safe_count = int(count)
    except ValueError:
        return "Error: Por favor, dame un número válido."
    
    if safe_count > 30: # Ponemos un límite para no inundar la consola
        safe_count = 30
    elif safe_count < 1:
        safe_count = 1
        
    try:
        # Ahora usamos 'safe_count' (que es un int)
        sample_names = db.sample(n=safe_count).index.tolist()
        
        # Devuelve los nombres como un solo string, separados por saltos de línea
        return "\n".join(sample_names)
        
    except Exception as e:
        return f"Error al obtener productos de muestra: {str(e)}"