#SaveHuggingFace.py
model.save_pretrained("./deepseek_model")
tokenizer.save_pretrained("./deepseek_model")

api.upload_folder(repo_id="hectorflores28/deepseek-ollama", folder_path="./deepseek_model")