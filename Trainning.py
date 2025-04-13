#Trainning.py
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import json
import torch
import os
from transformers import DataCollatorForLanguageModeling

def load_and_prepare_dataset(file_path):
    """Carga y prepara el dataset para entrenamiento."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Preparar conversaciones
    conversations = []
    for conv in data['conversaciones']:
        conversations.append({
            "text": f"Pregunta: {conv['input']}\nRespuesta: {conv['output']}"
        })
    
    # Preparar información de hoteles
    for hotel, info in data['informacion_hoteles'].items():
        hotel_info = f"Información de {hotel}:\n"
        hotel_info += f"Ubicación: {info['ubicacion']}\n"
        hotel_info += f"Habitaciones: {info['habitaciones']}\n"
        hotel_info += f"Restaurantes: {info['restaurantes']}\n"
        hotel_info += f"Servicios: {', '.join(info['servicios_principales'])}\n"
        hotel_info += f"Categorías de habitaciones: {', '.join(info['categorias_habitaciones'])}"
        conversations.append({"text": hotel_info})
    
    # Preparar políticas
    policies = "Políticas Generales:\n"
    for section, details in data['politicas_generales'].items():
        policies += f"{section}:\n"
        if isinstance(details, dict):
            for key, value in details.items():
                policies += f"- {key}: {value}\n"
        elif isinstance(details, list):
            for item in details:
                policies += f"- {item}\n"
    conversations.append({"text": policies})
    
    return Dataset.from_dict({"text": [item["text"] for item in conversations]})

def tokenize_function(examples, tokenizer):
    """Tokeniza el texto para el entrenamiento."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

# Configuración de rutas
DATASET_PATH = "dataset/velas_dataset.json"
OUTPUT_DIR = "./velas_assistant"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar dataset
dataset = load_and_prepare_dataset(DATASET_PATH)

# Cargar modelo y tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# Tokenizar dataset
tokenized_dataset = dataset.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=dataset.column_names
)

# Configurar argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
)

# Crear data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Crear trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)