import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- 1. CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# FIX: Use Absolute Path for MLflow to avoid the URI Scheme error
base_dir = os.path.abspath(os.getcwd())
mlflow_uri = f"file://{base_dir}/mlruns"
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("wellness-tourism-experiment")

# --- 2. LOAD DATA ---
repo_id_data = "VKblues2025/wellness-tourism-data"
print("Loading datasets from Hugging Face Hub...")

# Direct URL access for reliability in GitHub Actions
train_path = f"https://huggingface.co/datasets/{repo_id_data}/resolve/main/train_v1.csv"
test_path = f"https://huggingface.co/datasets/{repo_id_data}/resolve/main/test_v1.csv"

# Load dataframes
train_df = pd.read_csv(train_path, storage_options={'token': HF_TOKEN})
test_df = pd.read_csv(test_path, storage_options={'token': HF_TOKEN})

target_col = 'ProdTaken'
Xtrain = train_df.drop(columns=[target_col])
ytrain = train_df[target_col]
Xtest = test_df.drop(columns=[target_col])
ytest = test_df[target_col]

# --- 3. PIPELINE SETUP ---
# Automatically identify column types
numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- 4. HYPERPARAMETER TUNING ---
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

print("Starting Grid Search...")
with mlflow.start_run(run_name="Best_XGB_Classifier"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Evaluation
    y_pred = best_model.predict(Xtest)
    metrics = {
        "test_accuracy": accuracy_score(ytest, y_pred),
        "test_f1": f1_score(ytest, y_pred),
        "test_precision": precision_score(ytest, y_pred),
        "test_recall": recall_score(ytest, y_pred)
    }
    mlflow.log_metrics(metrics)
    print(f"✅ Training Complete. Best Accuracy: {metrics['test_accuracy']:.4f}")

    # --- 5. SAVE THE PIPELINE ---
    model_filename = "model.joblib"
    joblib.dump(best_model, model_filename)

    # This should now work without the URI error
    try:
        mlflow.log_artifact(model_filename)
    except Exception as e:
        print(f"⚠️ MLflow Artifact Log Warning: {e}")

# --- 6. UPLOAD TO HUGGING FACE MODEL HUB ---
repo_id_model = "VKblues2025/wellness-tourism-model"
try:
    api.repo_info(repo_id=repo_id_model, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id_model, repo_type="model", private=False, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo="model.joblib",
    repo_id=repo_id_model,
    repo_type="model",
    token=HF_TOKEN
)
print(f"✨ Model successfully registered to {repo_id_model}")
