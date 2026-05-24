
# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model Building
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# MLflow Configuration
# ==========================================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction_Experiment")

# ==========================================
# Initialize Hugging Face API
# ==========================================

api = HfApi(token=os.getenv("HF_TOKEN"))

dataset_repo_id = "RahulGolande/tourism-package-prediction"

# ==========================================
# Load Processed Datasets Directly from Hugging Face
# (Rubric requirement: load from HF data space)
# ==========================================

hf_base_url = (
    f"https://huggingface.co/datasets/{dataset_repo_id}/resolve/main"
)

Xtrain = pd.read_csv(f"{hf_base_url}/Xtrain.csv")
Xtest  = pd.read_csv(f"{hf_base_url}/Xtest.csv")
ytrain = pd.read_csv(f"{hf_base_url}/ytrain.csv")
ytest  = pd.read_csv(f"{hf_base_url}/ytest.csv")

# Convert target to series
ytrain = ytrain.squeeze()
ytest  = ytest.squeeze()

print("Processed datasets loaded from Hugging Face successfully!")

# ==========================================
# Define Feature Categories
# (Categorical cols remain as strings — encoded in pipeline)
# ==========================================

numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# ==========================================
# Preprocessing Pipeline
# ==========================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ==========================================
# Define XGBoost Classifier
# ==========================================

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# ==========================================
# Hyperparameter Grid
# ==========================================

param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 1.0]
}

# ==========================================
# Create ML Pipeline
# ==========================================

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ==========================================
# MLflow Experiment Tracking
# ==========================================

with mlflow.start_run():

    # ======================================
    # Grid Search CV
    # ======================================

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1'
    )

    grid_search.fit(Xtrain, ytrain)

    # ======================================
    # Log All Parameter Combinations
    # ======================================

    results = grid_search.cv_results_

    for i in range(len(results['params'])):

        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1_score", mean_score)

    # ======================================
    # Best Model
    # ======================================

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # ======================================
    # Predictions
    # ======================================

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    # ======================================
    # Evaluation Metrics
    # ======================================

    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy  = accuracy_score(ytest, y_pred_test)
    precision      = precision_score(ytest, y_pred_test)
    recall         = recall_score(ytest, y_pred_test)
    f1             = f1_score(ytest, y_pred_test)

    # ======================================
    # Log Metrics in MLflow
    # ======================================

    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy":  test_accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1
    })

    # ======================================
    # Print Results
    # ======================================

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nModel Performance:")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(ytest, y_pred_test))

    # ======================================
    # Save Best Model
    # ======================================

    os.makedirs("tourism_project/models", exist_ok=True)

    model_path = "tourism_project/models/best_tourism_model.pkl"

    joblib.dump(best_model, model_path)
    print("\nBest model saved successfully!")

    # ======================================
    # Log Model Artifact in MLflow
    # ======================================

    mlflow.log_artifact(model_path, artifact_path="model")

    # ======================================
    # Create Hugging Face Model Repository
    # ======================================

    model_repo_id = "RahulGolande/tourism-package-prediction-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Model repository '{model_repo_id}' already exists.")

    except RepositoryNotFoundError:
        print(f"Model repository '{model_repo_id}' not found.")
        print("Creating new model repository...")

        create_repo(
            repo_id=model_repo_id,
            repo_type=repo_type,
            private=False,
            token=os.getenv("HF_TOKEN")
        )
        print(f"Model repository '{model_repo_id}' created successfully!")

    # ======================================
    # Upload Best Model to Hugging Face
    # ======================================

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model.pkl",
        repo_id=model_repo_id,
        repo_type=repo_type
    )

    print("\nBest model uploaded to Hugging Face successfully!")

print("\nExperiment tracking completed successfully!")
