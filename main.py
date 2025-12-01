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

#--- Herramientas para el LLM ---
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
    """
    Intenta convertir una cadena de texto en un diccionario de Python utilizando
    una estrategia de doble validación (Dual-Parsing Strategy).

    Esta utilidad es fundamental para trabajar con LLMs, ya que los modelos a menudo
    generan "JSONs inválidos" que en realidad son diccionarios de Python sintácticamente
    correctos (uso de comillas simples en lugar de dobles).

    Estrategia de ejecución:
    ------------------------
    1. **Intento Estricto (JSON Standard)**:
        Prueba primero con `json.loads`. Es más rápido y es el estándar esperado.
        Requiere comillas dobles estrictas (`{"key": "value"}`).

    2. **Intento Permisivo (AST Literal Evaluation)**:
        Si falla el JSON, recurre a `ast.literal_eval`.
       - **Ventaja**: Acepta sintaxis de Python (comillas simples `{'key': 'value'}`),
            comas finales (trailing commas) y `None` en lugar de `null`.
       - **Seguridad**: A diferencia de `eval()`, esta función es segura ya que solo
            procesa literales (strings, números, tuplas, listas, dicts, booleanos, None)
            y no puede ejecutar código arbitrario ni llamadas a funciones.

    Args:
        text (str): La cadena de texto que representa la estructura de datos.

    Returns:
        dict | None:
            - El diccionario parseado si tiene éxito.
            - `None` si la cadena no representa una estructura válida o está corrupta.
    """
    try: return json.loads(text)
    except:
        try: return ast.literal_eval(text)
        except: return None

def try_parse_json_tool(content):
    """
    Intenta detectar, reparar y estructurar una llamada a herramienta desde texto crudo (Fallback Mechanism).

    Esta función se activa cuando el modelo "alucina" una llamada a herramienta escribiendo
    JSON en el texto en lugar de usar la API nativa de `tool_calls`. Actúa como un
    puente para normalizar estas respuestas y evitar que el flujo se rompa.

    Fases del procesamiento:
    ------------------------
    1. **Micro-cirugía de JSON (Regex Injection)**:
        Detecta un patrón de error común en modelos pequeños donde omiten la clave "arguments"
        pero abren las llaves de los argumentos inmediatamente después del nombre.
       * Patrón detectado: `{"name": "foo", {"arg": 1}}`
       * Acción: Inyecta `"arguments":` para convertirlo en JSON válido.

    2. **Extracción y Validación**:
        Busca un bloque JSON que contenga explícitamente la clave `"name"`.
        Utiliza `safe_parse_dict` para convertir el string en un diccionario de Python de forma segura.

    3. **Normalización (Mapping)**:
        Estandariza las claves de argumentos (acepta `arguments` o `args`) y envuelve el resultado
        en una lista con la estructura exacta que espera el bucle principal (`name`, `args`, `id`),
        simulando ser una llamada nativa.

    Args:
        content (str): El texto completo de la respuesta del LLM que podría contener el JSON.

    Returns:
        list[dict] | None:
            - Lista con un objeto de herramienta estandarizado: `[{'name': ..., 'args': ..., 'id': 'manual'}]`.
            - `None` si no se encuentra un JSON válido o no parece ser una llamada a herramienta.
    """
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
    """
    Realiza un saneamiento heurístico de la respuesta cruda del LLM para extraer
    texto legible por humanos y filtrar estructuras de datos internas.

    Esta función actúa como una capa de defensa (safety layer) contra "alucinaciones de formato",
    donde el modelo podría devolver JSON o llamadas a herramientas residuales en una etapa
    donde se espera texto plano.

    Lógica de procesamiento:
    ------------------------
    1. **Filtrado de Tool Calls (Guard Clause)**:
        Analiza si el contenido contiene patrones de llamadas a funciones (ej. clave `"name":`).
        Si se detecta, retorna `None` inmediatamente para evitar que el usuario vea
        código interno o definiciones de herramientas.

    2. **Desempaquetado de JSON (JSON Unwrapping)**:
        Utiliza Regex (`re.DOTALL`) para encontrar y extraer bloques JSON `{...}` incrustados
        dentro de texto basura o wrappers.

    3. **Normalización de Campos**:
        Si se parsea un objeto JSON válido, intenta extraer el mensaje final buscando en
        orden de prioridad:
        - `arguments["text"]`: Caso donde el LLM intenta usar una tool de chat como JSON.
        - `text`: Clave estándar de respuesta.
        - `message`: Clave alternativa común.

    4. **Fallback (Modo a prueba de fallos)**:
        Ante cualquier excepción de parsing o si no se encuentran estructuras,
        asume que el contenido original es texto válido y lo devuelve intacto.

    Args:
        content (str): La cadena cruda (raw string) generada por el modelo.

    Returns:
        str | None:
            - Retorna el texto limpio extraído.
            - Retorna `None` si el contenido se identifica como puramente técnico/estructural.
            - Retorna `content` original si no aplicó ninguna regla de limpieza.
    """
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
    """
    Bucle principal de orquestación del ciclo de vida del chat (Loop REPL).

    Este ciclo gestiona la interacción continua entre el usuario y el Asistente AI,
    implementando un patrón de ejecución de herramientas (Tool Calling) con una
    estrategia de generación de respuesta refinada.

    Flujo de ejecución:
    -------------------
    1. **Captura de Input**: Lee la entrada del usuario y verifica la condición de salida ('salir').
    2. **Inferencia Inicial**: Invoca al modelo (`llm_with_tools`) para determinar la intención
        y si se requieren herramientas.
    3. **Detección de Herramientas**:
        - Verifica `ai_response.tool_calls` nativos.
        - Si falla, intenta un fallback con `try_parse_json_tool` para modelos que devuelven JSON raw.
    4. **Ejecución de Herramientas (Si aplica)**:
       - **Normalización**: Sanea nombres de herramientas (con `TOOL_ALIASES`) y argumentos
            (ej. convierte 'text' a 'message' o 'query' a 'search_term') para coincidir con la firma de la función.
       - **Invocación**: Ejecuta la herramienta seleccionada y captura el resultado o errores.
       - **Formateo**: Si el resultado es una lista de productos, la convierte a un string legible.
    5. **Estrategia de Respuesta Final (Synthesis)**:
        - Si la herramienta fue conversacional (`is_chat_tool`), usa el resultado directo.
        - Si fue una herramienta de datos (ej. búsqueda), utiliza una técnica de "Ghost Prompt":
            Crea una lista de mensajes temporal (`messages_for_llm`) inyectando el resultado técnico
            y una instrucción de sistema ("Actúa como vendedor...") para generar una respuesta natural
            *sin contaminar* el historial principal con instrucciones de formateo.
    6. **Persistencia**: Actualiza `chat_history` con el input humano, los resultados técnicos
        de las herramientas y la respuesta final del bot.

    Excepciones manejadas:
    ----------------------
    - Errores de ejecución de herramientas son capturados y devueltos como strings de error
        para que el LLM pueda informar al usuario en lugar de romper el ciclo.

    Salida:
    -------
    Imprime en consola el progreso (🛠️, ✅, Bot:) y actualiza la lista global `chat_history`.
    """
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

        # 2. Ejecutamos las herramientas
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

        if is_chat_tool:
            texto_final = last_tool_result
        else:
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