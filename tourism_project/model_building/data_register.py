
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# -----------------------------
# Hugging Face Dataset Details
# -----------------------------

repo_id = "RahulGolande/tourism-package-prediction"   # FIXED: was placeholder
repo_type = "dataset"

# -----------------------------
# Initialize API
# -----------------------------

api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Check/Create Dataset Repo
# -----------------------------

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists.")

except RepositoryNotFoundError:

    print(f"Dataset repository '{repo_id}' not found.")
    print("Creating new dataset repository...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=os.getenv("HF_TOKEN")
    )

    print(f"Dataset repository '{repo_id}' created successfully!")

# -----------------------------
# Upload Dataset Folder
# -----------------------------

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Dataset uploaded successfully!")
