if HF_TOKEN is None:
    raise Exception("HF_TOKEN not found")

api = HfApi(token=HF_TOKEN)

space_repo_id = "RahulGolande/tourism-package-prediction-app"

# ==========================================
# Create Space
# ==========================================

try:
    api.repo_info(
        repo_id=space_repo_id,
        repo_type="space"
    )

    print("Space already exists")

except RepositoryNotFoundError:

    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=HF_TOKEN
    )

    print("Space created successfully")

# ==========================================
# Verify Deployment Files
# ==========================================

required_files = [
    "tourism_project/deployment/app.py",
    "tourism_project/deployment/Dockerfile",
    "tourism_project/deployment/requirements.txt"
]

for file in required_files:
    if not os.path.exists(file):
        raise Exception(f"Missing deployment file: {file}")

print("All deployment files verified")

# ==========================================
# Upload Deployment Folder
# ==========================================

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=space_repo_id,
    repo_type="space",
    path_in_repo=""
)

print("Deployment uploaded successfully")
