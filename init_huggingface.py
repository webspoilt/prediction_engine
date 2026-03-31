import os
import logging
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
HF_REPO_ID = os.getenv("HF_REPO_ID", "zeroday01/predictionsingle")
HF_TOKEN = os.getenv("HF_TOKEN")

def upload_project_to_hub():
    """
    Syncs the entire project folder to Hugging Face Hub using the Python API.
    Replaces the broken 'hf' CLI launcher on the user's system.
    """
    token = HF_TOKEN
    if not token:
        logger.warning("⚠️ HF_TOKEN environment variable not set.")
        token = input("🔑 Please enter your Hugging Face Write Token: ").strip()
        if not token:
            logger.error("❌ ERROR: No token provided. Abortion.")
            return

    api = HfApi()
    
    # 1. Determine local folder (current directory)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"🚀 Starting Python-based Upload to: https://huggingface.co/{HF_REPO_ID}...")
    
    try:
        # 2. Perform Folder Upload
        # We manually exclude common noise files since we aren't using the CLI's --exclude parser
        logger.info("📡 Scanning project and uploading files... (This may take a minute)")
        
        api.upload_folder(
            folder_path=local_dir,
            repo_id=HF_REPO_ID,
            token=token,
            commit_message="Initial Deployment via Python Uploader",
            ignore_patterns=[
                ".venv*",
                "__pycache__",
                ".git",
                ".pytest_cache",
                "*.log",
                "pytest_*.txt",
                "dataset/archive*",
                "data/match_jsons/*"
            ]
        )
        
        logger.info(f"✅ SUCCESS! Your engine is now live on the Hub: https://huggingface.co/{HF_REPO_ID}")
        
    except Exception as e:
        logger.error(f"❌ UPLOAD FAILED: {e}")
        if "Repository not found" in str(e):
            logger.info("💡 Hint: Make sure the repository 'zeroday01/predictionsingle' exists on Hugging Face!")

if __name__ == "__main__":
    upload_project_to_hub()
