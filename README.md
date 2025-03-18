
# Qwen2-7B_Vicon Fine-tuning for Motion Capture Analysis

This repository contains the code and instructions for fine-tuning **Qwen2-7B_Vicon** using **LoRA and LlamaFactory** to analyze motion capture data from **Vicon**. The fine-tuned model provides **real-time feedback on movement patterns and injury prevention**.

---

## **1. Getting Started**

### **1.1 Clone the Repository**
```bash
git clone https://github.com/VoidCorner/Qwen2-7B_Vicon.git
cd Qwen2-7B_Vicon
```

### **1.2 Install Dependencies**
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## **2. Download the Pre-trained Qwen2-7B_Vicon Model**

Before fine-tuning, download the **pre-trained Qwen2-7B** from **Hugging Face**:

### **2.1 Download from Hugging Face**
Use the `transformers` library to download:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
```
Alternatively, you can manually download the model files from:
➡[Qwen2-7B_Vicon on Hugging Face](https://huggingface.co/Qwen/Qwen2-7B)

---

## **3. Fine-tune Qwen2-7B_Vicon using LoRA and LlamaFactory**

### **3.1 Configure Training Parameters**
Modify `config/config.json` as needed:
```json
{
    "base_model": "VoidCorner/Qwen2-7B_Vicon",
    "output_dir": "./models/Qwen2-7B_Vicon_finetuned",
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "train_data": "./data/motion_data.csv"
}
```

### **3.2 Run Fine-tuning**
Execute the training script:
```bash
python src/train.py --config config/config.json
```
This will fine-tune the model using **LoRA** and save the weights in `./models/Qwen2-7B_Vicon_finetuned`.

---

## **4. Run Inference with the Fine-tuned Model**

After training, use the following command to test the model:
```bash
python src/infer.py --model_path ./models/Qwen2-7B_Vicon_finetuned
```
You can then interact with the model:
```text
Enter your movement question: How should I adjust my squat posture?
AI Response: To optimize your squat, keep your knees aligned with your toes...
```

---

## **5. Download the Fine-tuned Model**

The fine-tuned model has been uploaded to **Hugging Face** for easy access.

**Download Fine-tuned Model**: [Qwen2-7B_Vicon Fine-tuned](https://huggingface.co/VoidCorner/Qwen2-7B_Vicon)

Use this Python script to load the fine-tuned model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "VoidCorner/Qwen2-7B_Vicon"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(model, "VoidCorner/Qwen2-7B_Vicon_finetuned")
```
