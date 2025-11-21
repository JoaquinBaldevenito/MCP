from unsloth import FastLanguageModel
import os
import shutil

# --- CONFIGURACIÓN ---
model_name = "mi_modelo_afinado_lora" 
output_dir = "mi_modelo_ollama"
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
# El método guarda el archivo, pero a veces lo deja en la raíz
model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m")

# --- 3. CORRECCIÓN AUTOMÁTICA DE ARCHIVOS ---
print("--- Verificando ubicación del archivo GGUF ---")

# Nombre estándar que usa Unsloth
expected_filename = "unsloth.Q4_K_M.gguf" 
# A veces usa el nombre del modelo base
alt_filename = "phi-3-mini-4k-instruct.Q4_K_M.gguf" 

# Posibles ubicaciones donde puede haber quedado el archivo
possible_paths = [
    f"{output_dir}/{expected_filename}",   # Dentro de la carpeta (lo ideal)
    expected_filename,                     # En la raíz (lo común)
    alt_filename,                          # En la raíz con otro nombre
    f"{output_dir}/{alt_filename}"         # Dentro con otro nombre
]

found = False
final_path = f"{output_dir}/{expected_filename}"

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Archivo encontrado en: {path}")
        
        # Si no está donde queremos, lo movemos
        if path != final_path:
            print(f"🚚 Moviendo archivo a: {final_path} ...")
            shutil.move(path, final_path)
        
        found = True
        break

if found:
    print(f"🎉 ¡Éxito! Tu modelo está listo en: {final_path}")
    print("Ahora asegúrate de que tu Modelfile apunte a:")
    print(f"FROM ./{final_path}")
else:
    print("❌ ERROR: No encuentro el archivo .gguf generado.")
    print("Revisa la carpeta manualmente.")