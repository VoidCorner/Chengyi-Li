import os
import json
import torch
from llama_factory import create_model
from peft import LoraConfig
from datasets import load_dataset

with open("config/config.json", "r") as f:
    config = json.load(f)

dataset = load_dataset("csv", data_files=config["train_data"])
def tokenize_function(examples):
    return model.tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = create_model(config["base_model"], torch_dtype=torch.float16, device_map="auto")

peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"]
)
model.add_adapter(peft_config)

model.train(tokenized_datasets["train"], output_dir=config["output_dir"], epochs=config["epochs"])

model.save_pretrained(config["output_dir"])
