import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

# --- 1. CONFIGURATION ---
DATA_REPO_ID = "VKblues2025/wellness-tourism-data"
RAW_FILE = "tourism.csv"
TRAIN_FILE = "train_v1.csv"
TEST_FILE = "test_v1.csv"
HF_TOKEN = os.environ.get("HF_TOKEN")

def run_preparation():
    print(f"--- Step 1: Downloading {RAW_FILE} ---")
    raw_data_path = hf_hub_download(
        repo_id=DATA_REPO_ID,
        filename=RAW_FILE,
        repo_type="dataset",
        token=HF_TOKEN
    )

    df = pd.read_csv(raw_data_path)

    # --- 2. CLEANING & SPLITTING ---
    print("--- Step 2: Cleaning and Splitting ---")
    # Basic cleaning: Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Split the data into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save locally
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    # --- 3. UPLOAD BOTH TO HUB ---
    print("--- Step 3: Pushing Train and Test sets to Hub ---")
    api = HfApi()
    for file in [TRAIN_FILE, TEST_FILE]:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    print("✅ Success: train_v1.csv and test_v1.csv are now on the Hub!")

if __name__ == "__main__":
    run_preparation()
