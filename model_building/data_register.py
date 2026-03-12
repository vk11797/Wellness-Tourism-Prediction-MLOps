from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "VKblues2025/wellness-tourism-data"

# 1. Check if repo exists, if not, create it
try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"✅ Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"🚀 Creating new dataset repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", private=False)

# 2. Path to your local file
file_path = "tourism_project/data/tourism.csv"

# 3. Upload the file
if os.path.exists(file_path):
    print(f"📤 Commencing upload for {file_path}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✨ Success! Your data is now on the Hugging Face Hub.")
else:
    print(f"❌ Error: File not found at {file_path}")
