import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from tools import (
    search_products,
    refine_products,
    get_product_by_sku,
    get_similar_products,
    recommend_products,
    summarize_product,
    business_info,
    chat_response,
)

# =========================
# CONFIGURACIÓN MODELO
# =========================

MODEL_NAME = os.getenv("MODELO", "prueba")
print(f"🤖 Usando modelo afinado: {MODEL_NAME}")

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0  # CRÍTICO para que no rompa el JSON
)

print("✅ Sistema listo\n")

# =========================
# PROMPT DEL SISTEMA
# =========================

SYSTEM_PROMPT = """
Eres un asistente de una tienda de ropa.

Cuando el usuario necesite información del negocio o productos,
DEBES responder únicamente con un JSON válido, sin texto extra.

Formato OBLIGATORIO:
{
  "name": "<tool_name>",
  "arguments": { ... }
}

Tools disponibles:
- search_products(query)
- refine_products(color?, talle?, max_precio?, sort_by_price?)
- get_product_by_sku(sku)
- get_similar_products(sku)
- recommend_products()
- summarize_product(sku)
- business_info(topic)
- chat_response(message)

Reglas:
- NO expliques el JSON
- NO escribas texto fuera del JSON
- Si es charla normal, usa chat_response
"""

# =========================
# EJECUTOR DE TOOLS
# =========================

def ejecutar_tool(name, args):
    if name == "search_products":
        return search_products.invoke(args)

    if name == "refine_products":
        return refine_products.invoke(args)

    if name == "get_product_by_sku":
        return get_product_by_sku.invoke(args)

    if name == "get_similar_products":
        return get_similar_products.invoke(args)

    if name == "recommend_products":
        return recommend_products.invoke({})

    if name == "summarize_product":
        return summarize_product.invoke(args)

    if name == "business_info":
        return business_info.invoke(args)

    if name == "chat_response":
        return chat_response.invoke(args)

    return "Tool no reconocida"

# =========================
# FORMATEO RESULTADOS
# =========================

def mostrar_resultado(resultado):
    if isinstance(resultado, list):
        if not resultado:
            print("Bot: No encontré resultados.")
            return

        print("Bot:")
        for p in resultado:
            nombre = p.get("nombre", "Producto")
            precio = p.get("precio", "")
            print(f"• {nombre} → ${precio}")

    elif isinstance(resultado, dict) and "productos" in resultado:
        productos = resultado["productos"]
        if not productos:
            print("Bot: No encontré productos con esos filtros.")
            return

        print("Bot:")
        for p in productos:
            print(f"• {p['nombre']} → ${p['precio']}")

    else:
        print(f"Bot: {resultado}")

# =========================
# LOOP PRINCIPAL
# =========================

while True:
    user_input = input("Tú: ").strip()
    if user_input.lower() == "salir":
        break

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    try:
        # Intentar extraer JSON
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("No JSON")

        data = json.loads(match.group(0))
        tool_name = data.get("name")
        args = data.get("arguments", {})

        print(f"🛠️ Tool elegida: {tool_name} {args}")

        resultado = ejecutar_tool(tool_name, args)
        mostrar_resultado(resultado)

    except Exception:
        # No era JSON → charla normal
        print(f"Bot: {content}")
