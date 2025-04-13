#SaveHuggingFace.py
from huggingface_hub import HfApi, create_repo
import os
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_repository(api, repo_id):
    """Configura el repositorio en Hugging Face."""
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        logger.info(f"Repositorio {repo_id} configurado correctamente")
    except Exception as e:
        logger.error(f"Error al configurar el repositorio: {str(e)}")
        raise

def upload_model(api, repo_id, model_path):
    """Sube el modelo a Hugging Face."""
    try:
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El directorio del modelo {model_path} no existe")

        # Patrones de archivos a ignorar
        ignore_patterns = [
            "*.pyc", "__pycache__", "*.pyo", "*.pyd", ".Python",
            "env", "pip-log.txt", "pip-delete-this-directory.txt",
            "to_delete.txt", ".git", ".hg", ".mypy_cache", ".tox",
            ".coverage", ".coverage.*", ".cache", "nosetests.xml",
            "coverage.xml", "*.cover", "*.log", ".pytest_cache",
            ".env", ".venv", "venv", "ENV", "env.bak", "venv.bak"
        ]

        # Subir el modelo
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            ignore_patterns=ignore_patterns,
            commit_message="Actualización del modelo Velas Assistant"
        )
        logger.info(f"Modelo subido exitosamente a {repo_id}")
    except Exception as e:
        logger.error(f"Error al subir el modelo: {str(e)}")
        raise

def main():
    # Configuración
    REPO_ID = "hectorflores28/velas-assistant"
    MODEL_PATH = "./velas_assistant"
    
    try:
        # Inicializar API
        api = HfApi()
        
        # Configurar repositorio
        setup_repository(api, REPO_ID)
        
        # Subir modelo
        upload_model(api, REPO_ID, MODEL_PATH)
        
        logger.info("Proceso completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()