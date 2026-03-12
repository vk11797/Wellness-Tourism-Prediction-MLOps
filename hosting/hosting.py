import os
from huggingface_hub import HfApi

# 1. Configuration
REPO_ID = "VKblues2025/wellness-tourism"
HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

print(f"🚀 Starting deployment to Hugging Face Space: {REPO_ID}")

try:
    # 2. Cleanup (Optional but Recommended)
    # This removes old cached files in the Space to prevent 'Stopping' loops
    try:
        api.delete_file(path_in_repo="Dockerfile", repo_id=REPO_ID, repo_type="space")
    except Exception:
        pass

    # 3. Upload the Deployment Folder
    # This pushes app.py, requirements.txt, and the Dockerfile to the root of the Space
    print("📤 Uploading application files...")
    api.upload_folder(
        folder_path="tourism_project/deployment",
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo=""  # Files go to the root of the Space
    )

    print("✅ Deployment Successful! Your Space is now rebuilding.")
    print(f"🔗 View your app here: https://huggingface.co/spaces/{REPO_ID}")

except Exception as e:
    print(f"❌ Deployment Failed: {e}")
    # This helps debug in the GitHub Action logs
    raise e
