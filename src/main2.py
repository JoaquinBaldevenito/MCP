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
    temperature=0.0,
    format="json"
)

print("✅ Sistema listo\n")

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
        print("Bot: pase el not resultado")
        for p in resultado:
            nombre = p.get("nombre", "Producto")
            precio = p.get("precio", "")
            print(f"• {nombre} → ${precio}")

    elif isinstance(resultado, dict) and "productos" in resultado:
        productos = resultado["productos"]
        if not productos:
            print("Bot: No encontré productos con esos filtros.")
            return
        print("Bot: pase el not productos")
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

    response = llm.invoke(user_input)
    content = response.content.strip()
    print(f"DEBUG - Lo que llegó del modelo:\n{content}\n----------------")

    if not content:
        print("Bot: (El modelo no generó respuesta JSON válida. Intenta reformular).")
        continue

    try:
        # Como forzamos JSON, no necesitamos Regex complejos, parseamos directo
        data = json.loads(content)
        
        tool_name = data.get("name")
        args = data.get("arguments", {})

        print(f"🛠️ Tool elegida: {tool_name}")

        resultado = ejecutar_tool(tool_name, args)
        print(f"🧠 Resultado: {resultado}")
        mostrar_resultado(resultado)
    except json.JSONDecodeError:
        print("Bot: (Error interno: El modelo no generó una instrucción válida)")
        print(f"DEBUG ERROR: Contenido recibido: '{content}'")
    except Exception as e:
        print(f"Error: {e}")
