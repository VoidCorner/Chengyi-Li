import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
args = parser.parse_args()

model_name = "VoidCorner/Qwen2-7B_Vicon"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

print(" Model loaded successfully!")

df = pd.read_csv("data/motion_data.csv")
texts = df["text"].tolist()

for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    print(f" Input: {text}\n Model Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
