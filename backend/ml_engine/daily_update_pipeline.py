import asyncio
import logging
import os
import json
from datetime import datetime
from backend.data_pipeline.cricsheet_fetcher import fetch_recent_matches
from backend.ml_engine.build_player_db import update_with_new_players
from backend.ml_engine.train_efficient import train_efficiently
from backend.ml_engine.hybrid_model import HybridEnsemble, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (Env Vars for production flexibility)
HF_REPO_ID = os.getenv("HF_REPO_ID", "zeroday01/predictionsingle")
MODEL_PATH = os.getenv("MODEL_PATH", "models/hybrid_ensemble")
DATA_DIR = os.getenv("DATA_DIR", "data/match_jsons")

async def run_daily_update():
    """
    Main pipeline for the daily maintenance of the model.
    1. Fetch historical JSONs for the last 48 hours.
    2. Update the common Player Stats DB for any new players.
    3. Re-train the model (Efficiently) using newly added data.
    4. Save and Upload weights to Hugging Face Model Hub.
    """
    logger.info("📅 Starting Daily Model Update Pipeline...")

    try:
        # Step 1: Fetch recent matches (via Cricsheet)
        logger.info("Step 1: Fetching recent match data from Cricsheet...")
        new_match_files = fetch_recent_matches(output_dir=DATA_DIR, last_n_days=2)
        
        if not new_match_files:
            logger.info("ℹ️ No new matches to process today. Skipping update.")
            # We skip training but we might still check discovery or something else
            return

        # Step 2: Update Player Stats DB
        logger.info(f"Step 2: Updating Player stats with {len(new_match_files)} matches...")
        player_db_json = "backend/ml_engine/player_stats_db.json"
        update_with_new_players(new_match_files, player_db_json)

        # Step 3: Trigger Efficient Retraining
        logger.info("Step 3: Starting efficient model retraining...")
        # Actually calling the training logic (this would be heavy)
        # We'll use a wrapper that trains on the contents of DATA_DIR
        await asyncio.to_thread(train_efficiently, data_dir=DATA_DIR)

        # Step 4: Save & Push to Hugging Face Hub
        logger.info("Step 4: Pushing updated model to Hugging Face Hub...")
        ensemble = HybridEnsemble()
        ensemble.save_to_hub(
            repo_id=HF_REPO_ID, 
            path_prefix=MODEL_PATH, 
            commit_message=f"Auto-update: {datetime.now().date()}"
        )
        logger.info("✅ Daily update pipeline complete!")

    except Exception as e:
        logger.error(f"❌ Error in daily update pipeline: {e}")

if __name__ == "__main__":
    asyncio.run(run_daily_update())
