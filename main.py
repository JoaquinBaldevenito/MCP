import json
import os
import re
import ast

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import negocio 
from negocio import (
    find_products, get_opening_hours, get_location,
    get_general_recommendations, list_sample_products,
    get_return_policy, chat_response 
)

# --- 1. Cargar Datos ---
negocio.cargar_base_de_datos()

# --- 2. Herramientas ---
tools = [
    find_products, get_opening_hours, get_location,
    get_general_recommendations, list_sample_products,
    get_return_policy, chat_response 
]

# --- 3. Modelo ---
MODEL_NAME = os.getenv("MODELO", "mi_bot_v1")
print(f"🤖 Conectando al modelo: {MODEL_NAME}")

# Temperatura un poquito más alta (0.1) para que hable más fluido, pero baja.
llm = ChatOllama(model=MODEL_NAME, temperature=0.1) 
llm_with_tools = llm.bind_tools(tools)

# --- MAPEO DE SINÓNIMOS ---
TOOL_ALIASES = {
    "get_products": "find_products",
    "search_products": "find_products",
    "product_search": "find_products",
    "greetings": "chat_response",
    "message_response": "chat_response",
    "chat": "chat_response",
    "get_sample_response": "list_sample_products",
    "list_products": "list_sample_products"
}

# --- FUNCIONES AUXILIARES ---

def safe_parse_dict(text):
    try: return json.loads(text)
    except:
        try: return ast.literal_eval(text)
        except: return None

def try_parse_json_tool(content):
    try:
        if re.search(r'[\"\']name[\"\']\s*:\s*[\"\'][^\"\']+[\"\']\s*,\s*\{', content):
            content = re.sub(r'(,)\s*(\{)', r', "arguments": {', content, count=1)
        match = re.search(r'\{.*[\"\']name[\"\']:.*\}', content, re.DOTALL)
        if match:
            data = safe_parse_dict(match.group(0))
            if data and isinstance(data, dict):
                name = data.get("name")
                args = data.get("arguments", data.get("args", {}))
                return [{"name": name, "args": args, "id": "manual"}]
    except: pass
    return None

def clean_final_response(content):
    try:
        if re.search(r'[\"\']name[\"\']:', content) and '{' in content: return None 
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = safe_parse_dict(match.group(0))
            if data:
                if "arguments" in data and "text" in data["arguments"]: return data["arguments"]["text"]
                if "text" in data: return data["text"]
                if "message" in data: return data["message"]
            return None 
    except: pass
    return content

# --- 4. Chat Loop ---
print("--- SISTEMA LISTO. ESCRIBE 'salir' PARA TERMINAR ---")
chat_history = []

while True:
    user_input = input("\nTú: ")
    if user_input.lower() == "salir":
        break

    # Guardamos el mensaje del usuario
    chat_history.append(HumanMessage(content=user_input))

    # 1. Primera llamada
    ai_response = llm_with_tools.invoke(chat_history)
    tool_calls = ai_response.tool_calls
    
    if not tool_calls:
        tool_calls = try_parse_json_tool(ai_response.content)

    if tool_calls:
        # Guardamos la intención de herramienta en el historial
        chat_history.append(ai_response)
        
        last_tool_result = ""
        is_chat_tool = False
        
        for call in tool_calls:
            raw_name = call['name']
            tool_args = call['args']
            tool_name = TOOL_ALIASES.get(raw_name, raw_name)
            
            print(f"🛠️  Ejecutando: {tool_name}...")
            
            if "chat_response" in tool_name.lower():
                is_chat_tool = True
                if isinstance(tool_args, dict):
                    if "text" in tool_args: tool_args["message"] = tool_args.pop("text")
                    if "chat_response" in tool_args: tool_args["message"] = tool_args.pop("chat_response")

            if "find_products" in tool_name.lower():
                if isinstance(tool_args, dict) and not tool_args.get("search_term"):
                    if "query" in tool_args: tool_args["search_term"] = tool_args.pop("query")
                    elif "product" in tool_args: tool_args["search_term"] = tool_args.pop("product")
                    else: tool_args["search_term"] = ""

            selected_tool = next((t for t in tools if t.name.lower() == tool_name.lower()), None)
            
            if selected_tool:
                try:
                    tool_result = selected_tool.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error técnico: {e}"
            else:
                tool_result = f"Herramienta '{tool_name}' no encontrada."
            
            if isinstance(tool_result, dict) and "productos" in tool_result:
                items = tool_result["productos"]
                txt = "Opciones encontradas:\n"
                for p in items:
                    txt += f"• {p['nombre']} (${p['precio']})\n"
                tool_result = txt

            print(f"✅ Dato: {str(tool_result)[:100]}...") 
            last_tool_result = str(tool_result)
            
            # Guardamos el resultado TÉCNICO en el historial (esto sí es útil)
            chat_history.append(AIMessage(content=last_tool_result))

        # --- ESTRATEGIA DE RESPUESTA FINAL (MEJORADA) ---
        if is_chat_tool:
            texto_final = last_tool_result
        else:
            # --- CAMBIO CLAVE AQUÍ ---
            # Creamos una lista temporal para esta llamada. 
            # NO ensuciamos 'chat_history' con la instrucción de redacción.
            prompt_redaccion = f"""
            DATOS OBTENIDOS:
            {last_tool_result}

            TU TAREA:
            Actúa como un vendedor amable. Responde al usuario EXCLUSIVAMENTE EN ESPAÑOL basándote en los datos de arriba.
            NO uses JSON. NO inventes datos.
            """
            
            # Copia temporal + Instrucción fantasma
            messages_for_llm = chat_history + [SystemMessage(content=prompt_redaccion)]
            
            final_response = llm.invoke(messages_for_llm)
            texto_limpio = clean_final_response(final_response.content)
            
            if not texto_limpio:
                texto_final = str(last_tool_result)
            else:
                texto_final = texto_limpio

        print(f"Bot: {texto_final}")
        # Guardamos solo la respuesta bonita final
        chat_history.append(AIMessage(content=texto_final))

    else:
        texto = clean_final_response(ai_response.content)
        if not texto: texto = ai_response.content
        print(f"Bot: {texto}")
        chat_history.append(AIMessage(content=texto))