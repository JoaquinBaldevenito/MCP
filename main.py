import json
import os
import re
import ast
import inspect

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import negocio 
from negocio import (
    find_products, get_opening_hours, get_location,
    get_general_recommendations, list_sample_products,
    get_return_policy, chat_response 
)

negocio.cargar_base_de_datos()

MODEL_NAME = os.getenv("MODELO", "prueba") 
print(f"🤖 Conectando al modelo: {MODEL_NAME}")
llm = ChatOllama(model=MODEL_NAME, temperature=0.0)

# --- 1. MEMORIA DE SESIÓN ---
SESSION_CONTEXT = {
    "last_search_term": None,
    "last_product_list": ""
}

# --- 2. SANITIZADOR ---
def forzar_texto_plano(content):
    if not isinstance(content, str): return str(content)
    
    if re.search(r'"\w+":\s*".+"', content) and "{" not in content:
        return "" 

    if "{" in content:
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    for k in ["text", "message", "chat_response", "response"]:
                        if k in data: return data[k]
                    if "arguments" in data:
                        args = data["arguments"]
                        for k in ["text", "message", "chat_response"]:
                            if k in args: return args[k]
        except: pass
        clean = re.sub(r'["\']name["\']\s*:\s*["\'].*?["\'],?', '', content)
        clean = re.sub(r'[{}]', '', clean)
        return clean.strip()
    return content

# --- 3. CEREBRO LÓGICO (DICCIONARIO) ---
def procesar_intencion_con_memoria(texto_usuario):
    texto = texto_usuario.lower()
    
    # A. SALUDOS
    saludos = ["hola", "buenas", "qué tal", "hello", "hi", "buenos días", "buenas noches"]
    if any(texto.startswith(s) for s in saludos) or (len(texto.split()) < 4 and any(s in texto for s in saludos)):
        return {
            "tipo": "chat",
            "respuesta": "¡Hola! Bienvenido. ¿Buscas ropa (remeras, jeans) o necesitas información del local?"
        }

    # B. HERRAMIENTAS INFORMATIVAS
    if re.search(r'\b(hora|horarios?|abierto|cerrado|abren|cierran)\b', texto):
        return { "tipo": "tool", "json": '{ "name": "get_opening_hours", "arguments": {} }' }
    
    if re.search(r'\b(ubicacion|donde|direccion|local|queda|calle)\b', texto):
        return { "tipo": "tool", "json": '{ "name": "get_location", "arguments": {} }' }
    
    if re.search(r'\b(devolucion|cambio|politica|reembolso|devolver)\b', texto):
        return { "tipo": "tool", "json": '{ "name": "get_return_policy", "arguments": {} }' }

    # C. MAPEO DE PRODUCTOS (SOLUCIÓN AQUÍ)
    # Agregamos plurales, singulares y versiones sin acento
    mapa = {
        # Remeras
        "remera": "T-shirt", "remeras": "T-shirt", 
        "camiseta": "T-shirt", "camisetas": "T-shirt", 
        "chomba": "T-shirt", "chombas": "T-shirt",
        
        # Pantalones (Clave: 'pantalon' sin acento)
        "pantalón": "Jeans", "pantalon": "Jeans", "pantalones": "Jeans",
        "jeans": "Jeans", "jean": "Jeans", "vaquero": "Jeans", "vaqueros": "Jeans",
        
        # Zapatillas (Probamos 'Sneakers' para mayor compatibilidad)
        "zapatillas": "Sneakers", "zapatilla": "Sneakers", 
        "botas": "Boots", "bota": "Boots",
        "calzado": "Shoes", "zapatos": "Shoes", "zapato": "Shoes",
        
        # Abrigos
        "campera": "Jacket", "camperas": "Jacket",
        "chaqueta": "Jacket", "chaquetas": "Jacket",
        "abrigo": "Coat", "abrigos": "Coat",
        
        # Otros
        "vestido": "Dress", "vestidos": "Dress"
    }
    
    # Buscamos coincidencias
    prod_encontrado = None
    for k, v in mapa.items():
        # Usamos regex con \b para asegurar que coincida la palabra exacta
        if k in texto: 
            prod_encontrado = v
            break
    
    # D. MAPEO DE PRECIO
    orden = "None"
    if "caro" in texto or "cara" in texto: orden = "desc"
    if "barato" in texto or "barata" in texto: orden = "asc"

    # CASO 1: BÚSQUEDA
    if prod_encontrado:
        SESSION_CONTEXT["last_search_term"] = prod_encontrado
        return {
            "tipo": "tool",
            "json": f"""{{ "name": "find_products", "arguments": {{ "search_term": "{prod_encontrado}", "sort_by_price": "{orden}" }} }}"""
        }

    # CASO 2: REFINAMIENTO (Soloprecio, usa memoria)
    if orden != "None" and SESSION_CONTEXT["last_search_term"]:
        prod_memoria = SESSION_CONTEXT["last_search_term"]
        return {
            "tipo": "tool",
            "json": f"""{{ "name": "find_products", "arguments": {{ "search_term": "{prod_memoria}", "sort_by_price": "{orden}" }} }}"""
        }

    # CASO 3: DESCONOCIDO
    return {
        "tipo": "chat",
        "respuesta": "Disculpa, no entendí bien qué producto buscas. Prueba con: 'remeras', 'jeans', 'zapatillas'..."
    }

# --- BUCLE PRINCIPAL ---

print("--- SISTEMA LISTO ---")
chat_history = []

while True:
    try:
        user_input = input("\nTú: ")
        if user_input.lower() == "salir": break

        decision = procesar_intencion_con_memoria(user_input)
        
        if decision["tipo"] == "chat":
            print(f"Bot: {decision['respuesta']}")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=decision['respuesta']))
            continue 

        json_comando = decision["json"]
        match = re.search(r'\{.*\}', json_comando)
        data_json = json.loads(match.group(0))
        tool_name = data_json.get("name")
        args = data_json.get("arguments", {})
        
        print(f"🛠️  Herramienta detectada: {tool_name} {args if args else ''}...")
        
        func_to_call = None
        if tool_name == "find_products": func_to_call = find_products
        elif tool_name == "get_opening_hours": func_to_call = get_opening_hours
        elif tool_name == "get_location": func_to_call = get_location
        elif tool_name == "get_return_policy": func_to_call = get_return_policy
        
        listado_historia = ""
        
        if func_to_call:
            res = func_to_call.invoke(args)
            
            if isinstance(res, dict) and "productos" in res:
                productos = res['productos']
                if not productos:
                    listado_historia = "Lo siento, no encontré productos con esa descripción exacta en stock."
                else:
                    for p in productos:
                        listado_historia += f"• {p['nombre']} -> ${p['precio']}\n"
            else:
                listado_historia = str(res)
        else:
            listado_historia = "Error: Herramienta no encontrada."

        SESSION_CONTEXT["last_product_list"] = listado_historia

        print("✅ Datos obtenidos. Redactando...")
        
        ctx_limpio = [
            SystemMessage(content="Eres un vendedor. Tu única tarea es mostrar los datos al usuario en español. NO inventes precios."),
            HumanMessage(content=f"Usuario pregunta: {user_input}"),
            SystemMessage(content=f"RESULTADOS DE LA BÚSQUEDA:\n{listado_historia}\n\nINSTRUCCIÓN: Si hay una lista, copiala tal cual. Si hay error, dilo en español. NO uses JSON.")
        ]
        
        final_res = llm.invoke(ctx_limpio)
        texto_final = forzar_texto_plano(final_res.content)
        
        if len(texto_final) < 5 or "arguments" in texto_final:
            texto_final = listado_historia

        print(f"Bot: {texto_final}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        SESSION_CONTEXT["last_search_term"] = None