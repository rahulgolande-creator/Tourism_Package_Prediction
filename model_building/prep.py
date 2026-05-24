
# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from huggingface_hub import HfApi

# ==========================================
# Hugging Face Details
# ==========================================

repo_id = "RahulGolande/tourism-package-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ==========================================
# Load Dataset Locally
# ==========================================

DATASET_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")

print("Dataset Shape:", df.shape)

# ==========================================
# Remove Duplicates
# ==========================================

df.drop_duplicates(inplace=True)

# ==========================================
# Drop Unnecessary Columns
# ==========================================

df.drop(columns=['CustomerID'], inplace=True)

# ==========================================
# Handle Missing Values
# ==========================================

# Numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled successfully!")

# ==========================================
# Encode Categorical Variables
# ==========================================

label_encoder = LabelEncoder()

categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("Categorical encoding completed!")

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

# ==========================================
# Save Processed Files
# ==========================================

Xtrain.to_csv(
    "tourism_project/data/Xtrain.csv",
    index=False
)

Xtest.to_csv(
    "tourism_project/data/Xtest.csv",
    index=False
)

ytrain.to_csv(
    "tourism_project/data/ytrain.csv",
    index=False
)

ytest.to_csv(
    "tourism_project/data/ytest.csv",
    index=False
)

print("Processed datasets saved locally!")

# ==========================================
# Upload Processed Files to HF
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

print("All processed datasets uploaded successfully!")
