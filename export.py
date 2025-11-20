from unsloth import FastLanguageModel

# 1. Cargar tus adaptadores YA entrenados
# Nota: Usamos la carpeta donde se guardó el entrenamiento anterior
model_name = "mi_modelo_afinado_lora" 
max_seq_length = 1024
dtype = None
load_in_4bit = True

print(f"--- Cargando adaptadores desde: {model_name} ---")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Convertir a GGUF
print("--- Iniciando conversión a GGUF (Esto compilará llama.cpp) ---")
# Esto puede tardar unos minutos la primera vez mientras compila
model.save_pretrained_gguf("mi_modelo_ollama", tokenizer, quantization_method = "q4_k_m")

print("✅ ¡Conversión completada! Ahora puedes crear el modelo en Ollama.")