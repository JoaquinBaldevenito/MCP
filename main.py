import google.generativeai as genai
import os
from dotenv import load_dotenv

import negocio 

# 1. Cargar la API Key del archivo .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Cargar tu base de datos en memoria
negocio.cargar_base_de_datos()

# 3. Definir el "Protocolo":
#    Le decimos al modelo qué herramientas tiene disponibles.
tools = [
    negocio.get_price,
    negocio.check_stock,
    negocio.list_sample_products
    # TODO: Agregar mas cosas
]

# 4. Inicializar el modelo
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=tools
)

print("Iniciando chat. Escribí 'salir' para terminar.")

# 5. Iniciar la sesión de chat con el modelo
chat = model.start_chat(enable_automatic_function_calling=True)

# 6. Bucle principal de la consola
while True:
    # 1. Obtener input del usuario
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        break

    # 2. Enviar el input al modelo
    response = chat.send_message(user_input)
    print(f"Bot: {response.text}")