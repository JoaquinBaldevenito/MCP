from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

"""
Script de Entrenamiento (Fine-Tuning) con Unsloth optimizado para 4GB VRAM.

Descripción:
    Este script realiza un ajuste fino supervisado (SFT) del modelo 'Phi-3-mini-4k-instruct'
    utilizando técnicas de LoRA (Low-Rank Adaptation) y cuantización de 4 bits.
    
    Está configurado específicamente para funcionar en una grafica NVIDIA con 4GB VRAM,
    utilizando un tamaño de lote (batch size) de 1 y acumulación de gradientes para
    evitar errores de 'Out of Memory' (OOM).

Requisitos de Entrada:
    - Archivo 'dataset.jsonl': Debe contener objetos JSON con una lista "messages"
        estructurada (usuario/asistente).

Configuración Clave (Hardware Limitado):
    - Modelo base: unsloth/Phi-3-mini-4k-instruct (ligero y potente).
    - Max Seq Length: 1024 tokens (reducido para ahorrar VRAM).
    - Batch Size: 1 (con gradient_accumulation_steps = 4).
    - Optimizador: adamw_8bit (menor consumo de memoria).

Salida:
    - Guarda los adaptadores LoRA entrenados en la carpeta 'mi_modelo_afinado_lora'.
"""

# --- CONFIGURACIÓN PARA 4GB VRAM ---
max_seq_length = 1024 # Bajamos un poco el contexto para ahorrar memoria
dtype = None # Detecta automáticamente (float16)
load_in_4bit = True # OBLIGATORIO: Carga en 4 bits para que entre en 4GB

# Con una mejor gráfica, cambia esto a: 
# model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
model_name = "unsloth/Phi-3-mini-4k-instruct" 

print(f"--- Cargando modelo: {model_name} ---")

# 1. Cargar Modelo y Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Configurar PEFT (LoRA)
# Esto entrena solo el 1-5% del modelo, haciéndolo posible en tu GPU
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rango de atención
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Optimización extrema de memoria
    random_state = 3407,
)

# 3. Preparar el Dataset
# Función para convertir tu formato JSONL a texto plano que el modelo entienda
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

EOS_TOKEN = tokenizer.eos_token # Token de fin de texto

def formatting_prompts_func(examples):
    instructions = examples["messages"]
    outputs = []
    for conversation in instructions:
        # Extraemos el mensaje del usuario y del asistente
        user_msg = conversation[0]["content"]      # Primer mensaje (usuario)
        assistant_msg = conversation[1]["content"] # Segundo mensaje (bot con el JSON)
        
        # Formateamos el texto
        text = alpaca_prompt.format(
            instruction = user_msg,
            response = assistant_msg,
        ) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }

# Cargar dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

print("--- Iniciando Entrenamiento ---")

# 4. Configurar Entrenador
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        # --- AJUSTES CRÍTICOS PARA 4GB VRAM ---
        per_device_train_batch_size = 1, # Solo 1 ejemplo a la vez
        gradient_accumulation_steps = 4, # Acumula 4 pasos (simula batch de 4)
        # --------------------------------------
        warmup_steps = 5,
        max_steps = 60, # Número de pasos de entrenamiento
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Entrenar
trainer_stats = trainer.train()

print("--- Entrenamiento Finalizado ---")

# 6. Guardar el Adaptador (LoRA)
model.save_pretrained("mi_modelo_afinado_lora")
tokenizer.save_pretrained("mi_modelo_afinado_lora")