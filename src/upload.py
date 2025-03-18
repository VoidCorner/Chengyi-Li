from huggingface_hub import upload_folder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
parser.add_argument("--repo_id", type=str, required=True, help="Your Hugging Face repository ID")
args = parser.parse_args()

print(f" Uploading model to {args.repo_id}...")
upload_folder(folder_path=args.model_path, repo_id=args.repo_id)
print(f"Model uploaded successfully to Hugging Face: https://huggingface.co/{args.repo_id}")
