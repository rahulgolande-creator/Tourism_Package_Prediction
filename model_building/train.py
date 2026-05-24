
# ==========================================
# Import Required Libraries
# ==========================================

import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
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
# Load Processed Datasets from Hugging Face
# ==========================================

hf_base_url = (
    f"https://huggingface.co/datasets/{dataset_repo_id}/resolve/main"
)

Xtrain = pd.read_csv(f"{hf_base_url}/Xtrain.csv")
Xtest  = pd.read_csv(f"{hf_base_url}/Xtest.csv")
ytrain = pd.read_csv(f"{hf_base_url}/ytrain.csv")
ytest  = pd.read_csv(f"{hf_base_url}/ytest.csv")

ytrain = ytrain.squeeze()
ytest  = ytest.squeeze()

print("Processed datasets loaded from Hugging Face successfully!")

# ==========================================
# Define Feature Categories
# ==========================================

numeric_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# ==========================================
# Preprocessing Pipeline
# ==========================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

param_grid = {
    'xgbclassifier__n_estimators':    [100, 200],
    'xgbclassifier__max_depth':       [3, 5],
    'xgbclassifier__learning_rate':   [0.01, 0.1],
    'xgbclassifier__subsample':       [0.8, 1.0],
    'xgbclassifier__colsample_bytree':[0.8, 1.0]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ==========================================
# MLflow Experiment Tracking
# ==========================================

with mlflow.start_run():

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3, n_jobs=-1, scoring='f1'
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_f1_score", results['mean_test_score'][i])

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy  = accuracy_score(ytest, y_pred_test)
    precision      = precision_score(ytest, y_pred_test)
    recall         = recall_score(ytest, y_pred_test)
    f1             = f1_score(ytest, y_pred_test)

    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy":  test_accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1
    })

    print(f"\nTrain Accuracy : {train_accuracy:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(ytest, y_pred_test))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_tourism_model.pkl"
    joblib.dump(best_model, model_path)
    print("Best model saved!")

    mlflow.log_artifact(model_path, artifact_path="model")

    # Create HF model repo and upload
    model_repo_id = "RahulGolande/tourism-package-prediction-model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
        print(f"Model repo '{model_repo_id}' already exists.")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=model_repo_id,
            repo_type="model",
            private=False,
            token=os.getenv("HF_TOKEN")
        )
        print(f"Model repo '{model_repo_id}' created!")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model.pkl",
        repo_id=model_repo_id,
        repo_type="model"
    )
    print("Model uploaded to Hugging Face successfully!")

print("Experiment tracking completed successfully!")
