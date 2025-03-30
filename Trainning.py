#Trainning.py
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import json
import torch

# Cargar el dataset desde el archivo JSON
with open('dataset/velas_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preparar los datos de entrenamiento
train_data = []
for conv in data['conversaciones']:
    train_data.append({
        "input": conv['input'],
        "output": conv['output']
    })

# Convertir a formato de dataset
dataset = Dataset.from_dict({
    "input": [item["input"] for item in train_data],
    "output": [item["output"] for item in train_data]
})

# Cargar modelo y tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Configurar argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True if torch.cuda.is_available() else False,
)

# Crear trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo
model.save_pretrained("./velas_assistant")
tokenizer.save_pretrained("./velas_assistant")