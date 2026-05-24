
# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ==========================================
# Hugging Face Details
# ==========================================

repo_id = "RahulGolande/tourism-package-prediction"
api = HfApi(token=os.getenv("HF_TOKEN"))

# ==========================================
# Load Dataset Directly from Hugging Face
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
# ==========================================

cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# ==========================================
# Handle Missing Values
# ==========================================

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled successfully!")

# NOTE: NO LabelEncoder here.
# Categorical columns stay as strings.
# Encoding is handled inside the sklearn pipeline in train.py via OneHotEncoder.

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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# ==========================================
# Save Processed Files Locally
# ==========================================

os.makedirs("data", exist_ok=True)

Xtrain.to_csv("data/Xtrain.csv", index=False)
Xtest.to_csv("data/Xtest.csv", index=False)
ytrain.to_csv("data/ytrain.csv", index=False)
ytest.to_csv("data/ytest.csv", index=False)

print("Processed datasets saved locally!")

# ==========================================
# Upload Processed Files back to HF
# ==========================================

for file_path in ["data/Xtrain.csv", "data/Xtest.csv", "data/ytrain.csv", "data/ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"{file_path} uploaded to Hugging Face!")

print("All processed datasets uploaded successfully!")
