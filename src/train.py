import os
import json
import torch
from llama_factory import create_model
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

with open("config/config.json", "r") as f:
    config = json.load(f)

dataset = load_dataset("csv", data_files=config["train_data"])
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

base_model = "Qwen/Qwen2-7B"
print(f"⏳ Downloading base model {base_model}...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)
trainer.train()

model.save_pretrained(config["output_dir"])
print(" Fine-tuning completed! Model saved at:", config["output_dir"])
