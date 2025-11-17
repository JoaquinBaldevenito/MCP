import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Importa TODAS tus herramientas y el cargador de DB
import negocio 
from negocio import (
    find_products,
    get_opening_hours,
    get_location,
    get_general_recommendations,
    list_sample_products,
    get_return_policy,
    chat_response
)

# --- 1. Cargar la Base de Datos ---
negocio.cargar_base_de_datos()

# --- 2. Definir Herramientas ---
tools = [
    find_products,
    get_opening_hours,
    get_location,
    get_general_recommendations,
    list_sample_products,
    get_return_policy,
    chat_response # Importante para saludos/despedidas
]

# --- 3. Inicializar el modelo (APUNTANDO AL MODELO BASE PARA PROBAR) ---
llm = ChatOllama(model="mistral:instruct")

# --- 4. Vincular herramientas al modelo ---
llm_with_tools = llm.bind_tools(tools)

# --- 5. Lógica del Chat ---
print("Iniciando chat con MODELO BASE (para pruebas). Escribí 'salir' para terminar.")
print("---")
print("NOTA: El modelo base 'mistral:instruct' NO está afinado.")
print("Puede que intente chatear en vez de usar las herramientas. ¡El objetivo aquí es probar la lógica!")
print("---")

# Historial del chat
chat_history = []

while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        break

    # Agregar el input al historial
    chat_history.append(HumanMessage(content=user_input))

    # Invocar al modelo
    # El modelo recibe todo el historial + el último mensaje
    ai_response = llm_with_tools.invoke(chat_history)

    # Si la respuesta CONTIENE llamadas a herramientas
    if ai_response.tool_calls:
        print(f"[Debug: LLM quiere llamar a: {ai_response.tool_calls}]")
        
        # Agregar la decisión de la IA al historial
        chat_history.append(ai_response)
        
        # Ejecutar cada herramienta que el LLM pidió
        for call in ai_response.tool_calls:
            tool_name = call['name'].lower()
            tool_args = call['args']
            
            # Buscar la función correspondiente en 'negocio.py'
            selected_tool = next((t for t in tools if t.name.lower() == tool_name), None)
            
            tool_result = None
            if selected_tool:
                try:
                    # ¡Ejecutar la función!
                    tool_result = selected_tool.invoke(tool_args)
                except Exception as e:
                    tool_result = {"error": f"Error al ejecutar la herramienta: {e}"}
            else:
                tool_result = {"error": f"Herramienta '{tool_name}' desconocida."}
            
            print(f"[Debug: Resultado de la herramienta: {tool_result}]")
            
            # Agregar el resultado de la herramienta al historial
            chat_history.append(AIMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=call['id']
            ))

        # --- SEGUNDA LLAMADA AL LLM ---
        # Ahora que el historial contiene el resultado de la herramienta,
        # llamamos al modelo de nuevo para que genere una respuesta final.
        
        print("[Debug: Llamando al LLM de nuevo para la respuesta final...]")
        
        # Usamos el modelo SIN herramientas esta vez
        final_response = llm.invoke(chat_history)
        print(f"Bot: {final_response.content}")
        chat_history.append(final_response)

    else:
        # Si no llamó a una herramienta, solo chateó
        print(f"Bot: {ai_response.content}")
        chat_history.append(ai_response)