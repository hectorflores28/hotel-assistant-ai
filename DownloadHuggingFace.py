#DownloadHuggingFace.py
from huggingface_hub import snapshot_download

model_id = "deepseek-ai/deepseek-14b"
snapshot_download(repo_id=model_id, cache_dir="./deepseek_model")