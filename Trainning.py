#Trainning.py
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("deepseek-ai/deepseek-14b")
tokenizer = GPT2Tokenizer.from_pretrained("deepseek-ai/deepseek-14b")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Dataset de entrenamiento
)

trainer.train()