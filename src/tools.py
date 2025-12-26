import pandas as pd
from langchain.tools import tool

df = pd.read_csv("data\products_asos.csv")

@tool
def search_products(query: str):
    """
    Busca productos por texto libre (nombre, descripción o categoría)
    """
    q = query.lower()
    results = df[
        df["nombre"].str.lower().str.contains(q, na=False) |
        df["description"].str.lower().str.contains(q, na=False) |
        df["category"].str.lower().str.contains(q, na=False)
    ]
    return results.head(5).to_dict(orient="records")

@tool
def refine_products(
    color: str = None,
    talle: str = None,
    max_precio: float = None,
    sort_by_price: str = None
) -> dict:
    """Refina la última búsqueda del usuario."""
    results = df.copy()

    if color:
        results = results[results["color"] == color]
    if talle:
        results = results[results["talle"] == talle]
    if max_precio:
        results = results[results["precio"] <= max_precio]
    if sort_by_price:
        results = results.sort_values(
            by="precio",
            ascending=(sort_by_price == "asc")
        )

    return {
        "productos": results.head(5).to_dict(orient="records")
    }

@tool
def get_product_by_sku(sku: str):
    """
    Devuelve un producto exacto por SKU
    """
    product = df[df["sku"] == sku]
    if product.empty:
        return "No encontré ese producto"
    return product.iloc[0].to_dict()

@tool
def get_similar_products(sku: str):
    """
    Devuelve productos similares por categoría
    """
    base = df[df["sku"] == sku]
    if base.empty:
        return []

    category = base.iloc[0]["category"]
    results = df[df["category"] == category].head(5)
    return results.to_dict(orient="records")

@tool
def recommend_products():
    """
    Recomienda productos populares (precio medio)
    """
    avg_price = df["precio"].mean()
    results = df[
        (df["precio"] >= avg_price * 0.8) &
        (df["precio"] <= avg_price * 1.2)
    ]
    return results.head(5).to_dict(orient="records")

@tool
def summarize_product(sku: str):
    """
    Resume la descripción de un producto
    """
    product = df[df["sku"] == sku]
    if product.empty:
        return "Producto no encontrado"

    p = product.iloc[0]
    return f"{p['nombre']} en color {p['color']}, ideal para uso diario. Precio ${p['precio']}."

@tool
def business_info(topic: str):
    """
    Información general del negocio
    """
    info = {
        "envios": "Enviamos a todo el país",
        "pagos": "Aceptamos tarjetas y transferencias",
        "cambios": "Cambios hasta 30 días"
    }
    return info.get(topic, "No tengo esa información")

@tool
def chat_response(message: str) -> str:
    """Responder cuando NO se necesita acceder a datos."""
    return message
