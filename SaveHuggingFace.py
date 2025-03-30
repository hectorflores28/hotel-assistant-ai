#SaveHuggingFace.py
from huggingface_hub import HfApi
import os

# Inicializar el API de Hugging Face
api = HfApi()

# Guardar el modelo localmente
model_path = "./velas_assistant"
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Subir el modelo a Hugging Face
api.upload_folder(
    repo_id="hectorflores28/velas-assistant",
    folder_path=model_path,
    ignore_patterns=["*.pyc", "__pycache__", "*.pyo", "*.pyd", ".Python", "env", "pip-log.txt", "pip-delete-this-directory.txt", "to_delete.txt", ".git", ".hg", ".mypy_cache", ".tox", ".coverage", ".coverage.*", ".cache", "nosetests.xml", "coverage.xml", "*.cover", "*.log", ".pytest_cache", ".env", ".venv", "venv", "ENV", "env.bak", "venv.bak"],
    commit_message="Actualización del modelo Velas Assistant"
)