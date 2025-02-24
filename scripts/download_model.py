import sys
from huggingface_hub import hf_hub_download, snapshot_download

name="BAAI/bge-base-en-v1.5"

downloaded_model_path = snapshot_download(
    repo_id=name
)

print(downloaded_model_path)

