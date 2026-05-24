
# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

import os
import pandas as pd

from huggingface_hub import HfApi
from huggingface_hub import create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# VERIFY HF TOKEN
# ==========================================

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise Exception(
        "HF_TOKEN is missing in GitHub Actions Secrets"
    )

# ==========================================
# INITIALIZE API
# ==========================================

api = HfApi(token=HF_TOKEN)

dataset_repo_id = "RahulGolande/tourism-package-prediction"

# ==========================================
# CREATE DATASET REPO IF NOT EXISTS
# ==========================================

try:

    api.repo_info(
        repo_id=dataset_repo_id,
        repo_type="dataset"
    )

    print("Dataset repository already exists")

except RepositoryNotFoundError:

    create_repo(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        private=False,
        token=HF_TOKEN
    )

    print("Dataset repository created")

# ==========================================
# VERIFY DATASET FILE
# ==========================================

dataset_path = "tourism_project/data/tourism.csv"

if not os.path.exists(dataset_path):

    raise Exception(
        f"Dataset file missing: {dataset_path}"
    )

print("Dataset file verified")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(dataset_path)

print(df.head())

# ==========================================
# UPLOAD DATASET
# ==========================================

api.upload_file(
    path_or_fileobj=dataset_path,
    path_in_repo="tourism.csv",
    repo_id=dataset_repo_id,
    repo_type="dataset"
)

print("Dataset uploaded successfully")
