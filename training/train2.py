from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

"""
Fine-Tuning de un Agente Conversacional con Tools
=================================================

Modelo base:
    unsloth/Phi-3-mini-4k-instruct

Objetivo:
    Entrenar un asistente de tienda de ropa que:
    - Converse naturalmente
    - Decida cuándo llamar tools
    - Devuelva llamadas a tools en formato JSON

Características:
    - Optimizado para GPUs de 4GB VRAM
    - Entrenamiento LoRA + cuantización 4-bit
    - Dataset en formato JSONL con conversaciones
"""

# -------------------------------------------------------------------
# CONFIGURACIÓN GENERAL (OPTIMIZADA PARA 4GB VRAM)
# -------------------------------------------------------------------

MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
DTYPE = None  # auto (float16)

OUTPUT_DIR = "outputs"
LORA_OUTPUT_DIR = "modelo_lora_agente_ropa"

DATASET_PATH = "../data/dataset2.jsonl"

# -------------------------------------------------------------------
# CARGA DEL MODELO Y TOKENIZER
# -------------------------------------------------------------------

print(f"🔹 Cargando modelo base: {MODEL_NAME}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

EOS_TOKEN = tokenizer.eos_token

# -------------------------------------------------------------------
# CONFIGURACIÓN LoRA (PEFT)
# -------------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -------------------------------------------------------------------
# FORMATEO DEL DATASET (CHAT + TOOLS)
# -------------------------------------------------------------------
"""
Formato esperado en dataset.jsonl:

{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{ \"name\": \"search_products\", \"arguments\": {...} }"}
  ]
}

o conversaciones más largas.
"""

# Definimos el System Prompt que queremos que el modelo aprenda de memoria
SYSTEM_PROMPT = """Eres un asistente experto en ventas.
TU OBJETIVO: Responder SIEMPRE ejecutando una herramienta (Tool) en formato JSON.
NO hables texto plano."""

def format_chat(example):
    conversation = example["messages"]

    text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
    for msg in conversation:
        if msg["role"] == "user":
            text += f"<|user|>\n{msg['content']}<|end|>\n"
        elif msg["role"] == "assistant":
            text += f"<|assistant|>\n{msg['content']}<|end|>\n"

    text += EOS_TOKEN
    return {"text": text}


print("🔹 Cargando dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names,
    num_proc=1,  # más estable en Windows / WSL
)

# -------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTRENADOR
# -------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # 🔥 CRÍTICO PARA 4GB VRAM
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    max_steps=80,
    warmup_steps=5,
    learning_rate=2e-4,

    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,

    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=training_args,
)

# -------------------------------------------------------------------
# ENTRENAMIENTO
# -------------------------------------------------------------------

print("🚀 Iniciando entrenamiento...")
trainer.train()
print("✅ Entrenamiento finalizado")

# -------------------------------------------------------------------
# GUARDADO DEL MODELO LoRA
# -------------------------------------------------------------------

print(f"💾 Guardando adaptadores LoRA en '{LORA_OUTPUT_DIR}'")

model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

print("🎉 Listo. Modelo afinado correctamente.")
