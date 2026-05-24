
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# ==========================================
# Initialize Hugging Face API
# ==========================================

api = HfApi(token=os.getenv("HF_TOKEN"))

space_repo_id = "RahulGolande/tourism-package-prediction-app"

# ==========================================
# Create HF Space if it doesn't exist
# ==========================================

try:
    api.repo_info(repo_id=space_repo_id, repo_type="space")
    print(f"Space '{space_repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"Creating Space '{space_repo_id}'...")
    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=os.getenv("HF_TOKEN")
    )
    print(f"Space '{space_repo_id}' created successfully!")

# ==========================================
# Upload Deployment Folder to HF Space
# ==========================================

api.upload_folder(
    folder_path="deployment",
    repo_id=space_repo_id,
    repo_type="space",
    path_in_repo=""
)

print("Deployment files uploaded to Hugging Face Space successfully!")
