
from huggingface_hub import HfApi
import os

# ==========================================
# Initialize Hugging Face API
# ==========================================

api = HfApi(
    token=os.getenv("HF_TOKEN")
)

# ==========================================
# Upload Deployment Folder to HF Space
# ==========================================

api.upload_folder(

    folder_path="tourism_project/deployment",

    repo_id="RahulGolande/tourism-package-prediction-app",

    repo_type="space",

    path_in_repo=""
)

print(
    "Deployment files uploaded "
    "to Hugging Face Space successfully!"
)
