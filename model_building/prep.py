
# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# Hugging Face Details
# ==========================================

repo_id = "RahulGolande/tourism-package-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ==========================================
# Load Dataset Directly from Hugging Face
# (Rubric requirement: load from HF data space)
# ==========================================

hf_csv_url = (
    f"https://huggingface.co/datasets/{repo_id}"
    "/resolve/main/tourism.csv"
)

df = pd.read_csv(hf_csv_url)

print("Dataset loaded from Hugging Face successfully!")
print("Dataset Shape:", df.shape)

# ==========================================
# Remove Duplicates
# ==========================================

df.drop_duplicates(inplace=True)

# ==========================================
# Drop Unnecessary Columns
# (Unnamed: 0 is a leftover index; CustomerID is an ID column)
# ==========================================

cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# ==========================================
# Handle Missing Values
# ==========================================

# Numerical columns — fill with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns — fill with mode
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled successfully!")

# NOTE: Categorical columns are kept as strings.
# Encoding is handled inside the sklearn pipeline in train.py
# using OneHotEncoder — do NOT LabelEncode here.

# ==========================================
# Define Features and Target
# ==========================================

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================================
# Train-Test Split
# ==========================================

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed!")
print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# ==========================================
# Save Processed Files Locally
# ==========================================

os.makedirs("tourism_project/data", exist_ok=True)

Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print("Processed datasets saved locally!")

# ==========================================
# Upload Processed Files back to HF
# ==========================================

files = [
    "tourism_project/data/Xtrain.csv",
    "tourism_project/data/Xtest.csv",
    "tourism_project/data/ytrain.csv",
    "tourism_project/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"{file_path} uploaded successfully!")

print("All processed datasets uploaded to Hugging Face successfully!")
