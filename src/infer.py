import torch
from transformers import AutoTokenizer
from llama_factory import create_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
args = parser.parse_args()

base_model = "Qwen/Qwen2-7B"
model = create_model(base_model, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)

model.load_adapter(args.model_path)

while True:
    text = input("Enter your movement question: ")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs)
    print("AI Response:", tokenizer.decode(output[0], skip_special_tokens=True))
