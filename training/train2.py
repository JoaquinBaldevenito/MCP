from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

# -------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------------------

MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
DTYPE = None 

OUTPUT_DIR = "outputs"
LORA_OUTPUT_DIR = "modelo_lora_agente_ropa_v2"
DATASET_PATH = "../data/dataset_v2.jsonl"


import warnings
from typing import Any, Dict, List, Optional, Union

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False, ignore_index=-100):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.ignore_index = ignore_index
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        
        # Enmascarar (poner -100) todo lo que está antes de la respuesta del asistente
        for i in range(len(batch["labels"])):
            response_token_ids_start_idx = None
            
            # Buscamos dónde empieza la respuesta del asistente en los tokens
            for idx in range(len(batch["labels"][i]) - len(self.response_token_ids) + 1):
                if batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist() == self.response_token_ids:
                    response_token_ids_start_idx = idx
                    break
            
            if response_token_ids_start_idx is None:
                # Si no encuentra el tag, ignoramos toda la fila (seguridad)
                batch["labels"][i, :] = self.ignore_index
            else:
                # Ignoramos (mask) desde el principio hasta justo después del tag del asistente
                response_start_idx = response_token_ids_start_idx + len(self.response_token_ids)
                batch["labels"][i, :response_start_idx] = self.ignore_index

        return batch

# -------------------------------------------------------------------
# CARGA DEL MODELO
# -------------------------------------------------------------------

print(f"🔹 Cargando modelo base: {MODEL_NAME}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# -------------------------------------------------------------------
# CONFIGURACIÓN LoRA
# -------------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -------------------------------------------------------------------
# FORMATEO DE DATOS (FLEXIBLE)
# -------------------------------------------------------------------

def format_chat(example):
    """
    Convierte la lista de mensajes al formato ChatML de Phi-3.
    NO inyecta System Prompt (se asume que vendrá del Modelfile o del JSONL).
    """
    conversation = example["messages"]
    text = ""

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            text += f"<|system|>\n{content}<|end|>\n"
        elif role == "user":
            text += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}<|end|>\n"

    text += tokenizer.eos_token
    return {"text": text}

print("🔹 Cargando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names, num_proc=1)

# -------------------------------------------------------------------
# CONFIGURACIÓN DEL DATA COLLATOR (MASKING)
# -------------------------------------------------------------------
# Esto es vital: Asegura que el modelo solo calcule el error en la respuesta del asistente.
# Ignora el prompt del usuario durante el entrenamiento.

response_template = "<|assistant|>\n"

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

# -------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTRENADOR
# -------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # 3 Épocas completas para asegurar que aprenda tus tools
    num_train_epochs=3,
    
    warmup_steps=20,
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
    data_collator=data_collator, 
    args=training_args,
)

# -------------------------------------------------------------------
# ENTRENAMIENTO Y GUARDADO
# -------------------------------------------------------------------

print("🚀 Iniciando entrenamiento...")
trainer.train()

print(f"💾 Guardando en '{LORA_OUTPUT_DIR}'")
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

# Si vas a usar GGUF/Ollama directamente, puedes descomentar esto:

print("💾 Guardando GGUF (q4_k_m)...")

model.save_pretrained_gguf(LORA_OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")

print("🎉 Listo.")