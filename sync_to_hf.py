import os
import sys
from huggingface_hub import HfApi
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sync_to_spaces():
    """
    Reliably synchronizes the local project to Hugging Face Spaces.
    Bypasses Git LFS and push restrictions by using the direct API.
    """
    # ─── Configuration ───
    # We prioritize environment variables for security (Hugging Face / GitHub scanning)
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID") or "zeroday01/ipl_prediction_engine"
    
    if not token:
        logger.error("❌ HF_TOKEN not found. Please set it in your environment or .env file.")
        sys.exit(1)

    api = HfApi()
    
    logger.info(f"🚀 Starting high-speed sync to: {repo_id}")
    
    try:
        # ignore_patterns ensures we don't upload heavy local envs or giant git history
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            token=token,
            delete_patterns=["*.pyc", "__pycache__/*"], # Clean up remote artifacts
            ignore_patterns=[
                ".venv/*",
                ".venv312/*",
                "__pycache__/*",
                ".git/*",
                ".pytest_cache/*",
                "*.log",
                "pytest_*.txt",
                "zai/*",
                "tests/*",
                "docs/data_flow_pipeline.png" # Explicitly block the binary file that caused Git issues
            ]
        )
        logger.info("✅ SUCCESS: Project successfully synchronized to Hugging Face Spaces!")
        logger.info(f"📍 View live at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during sync: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_to_spaces()
