
# ==========================================
# IMPORT LIBRARIES
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

print("===================================")
print("HF TOKEN CHECK")
print("===================================")

if HF_TOKEN is None:

    raise Exception(
        "HF_TOKEN missing in GitHub Actions Secrets"
    )

else:

    print("HF_TOKEN found successfully")

# ==========================================
# INITIALIZE HUGGING FACE API
# ==========================================

api = HfApi(token=HF_TOKEN)

# ==========================================
# DATASET REPOSITORY
# ==========================================

dataset_repo_id = (
    "RahulGolande/tourism-package-prediction"
)

print("===================================")
print("DATASET REPOSITORY")
print("===================================")

print(dataset_repo_id)

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

    print("Creating dataset repository...")

    create_repo(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        private=False,
        token=HF_TOKEN
    )

    print("Dataset repository created successfully")

# ==========================================
# VERIFY DATA FILES
# ==========================================

required_files = [

    "tourism_project/data/tourism.csv",

    "tourism_project/data/Xtrain.csv",

    "tourism_project/data/Xtest.csv",

    "tourism_project/data/ytrain.csv",

    "tourism_project/data/ytest.csv"
]

print("===================================")
print("VERIFY DATA FILES")
print("===================================")

for file in required_files:

    print(file, ":", os.path.exists(file))

    if not os.path.exists(file):

        raise Exception(
            f"Missing required file: {file}"
        )

print("All dataset files verified successfully")

# ==========================================
# UPLOAD FILES TO HUGGING FACE
# ==========================================

print("===================================")
print("UPLOADING FILES")
print("===================================")

for file in required_files:

    filename = os.path.basename(file)

    print(f"Uploading {filename}...")

    api.upload_file(

        path_or_fileobj=file,

        path_in_repo=filename,

        repo_id=dataset_repo_id,

        repo_type="dataset"
    )

    print(f"{filename} uploaded successfully")

print("===================================")
print("ALL DATASETS UPLOADED SUCCESSFULLY")
print("===================================")
