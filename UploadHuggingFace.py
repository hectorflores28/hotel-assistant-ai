#UploadHuggingFace.py
from huggingface_hub import HfApi

api = HfApi()
model_path = "./deepseek_model"
api.upload_folder(repo_id="hectorflores28/deepseek-ollama", folder_path=model_path)
