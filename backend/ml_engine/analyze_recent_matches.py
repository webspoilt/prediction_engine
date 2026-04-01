import asyncio
import logging
import os
import pandas as pd
from datetime import datetime

# Import existing fetchers and builders
from backend.data_pipeline.cricsheet_fetcher import fetch_recent_matches
from backend.ml_engine.build_player_db import update_with_new_players
from backend.ml_engine.train_efficient import train_efficiently

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MATCHES_CSV = os.path.join(PROCESSED_DIR, "ipl_matches.csv")
BALL_CSV = os.path.join(PROCESSED_DIR, "ipl_ball.csv")

def ensure_base_csvs_exist():
    """Silences the [Errno 2] No such file warning on startup by initializing required CSVs."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if not os.path.exists(MATCHES_CSV):
        logger.info("Initializing base ipl_matches.csv to prevent startup warnings.")
        df = pd.DataFrame(columns=['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner', 'toss_decision', 'result', 'winner'])
        df.to_csv(MATCHES_CSV, index=False)
        
    if not os.path.exists(BALL_CSV):
        logger.info("Initializing base ipl_ball.csv to prevent startup warnings.")
        df = pd.DataFrame(columns=['match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball', 'batsman', 'bowler', 'runs_batter', 'runs_extras', 'runs_total', 'wicket'])
        df.to_csv(BALL_CSV, index=False)

async def analyze_completed_matches():
    """
    Specifically analyzes the matches completed this season.
    1. Fetches new matches.
    2. Analyzes player performances.
    3. Updates ML engine state.
    """
    logger.info("🏏 Starting Analysis of Completed Matches...")
    
    # 1. Ensure data directories exist
    ensure_base_csvs_exist()
    
    # 2. Fetch recent matches (last 7 days to cover the 3 completed matches)
    # We use Cricsheet as it provides the exact ball-by-ball JSON schema the ML model requires
    json_dir = "data/match_jsons"
    new_files = fetch_recent_matches(output_dir=json_dir, last_n_days=7)
    
    if new_files:
        logger.info(f"✅ Downloaded {len(new_files)} new match files for analysis.")
        
        # 3. Update Player Database
        db_path = "backend/ml_engine/player_stats_db.json"
        
        try:
            update_with_new_players(new_files, db_path)
            logger.info("✅ Player performance database updated with new match data.")
        except Exception as e:
            logger.error(f"Error updating player DB: {e}")

        # 4. Trigger Retraining
        logger.info("🧠 Triggering ML Engine update to learn from latest matches...")
        try:
            await asyncio.to_thread(train_efficiently, data_dir=json_dir)
            logger.info("✅ ML Engine successfully analyzed and adapted to the latest match dynamics.")
        except Exception as e:
            logger.error(f"Error during incremental training: {e}")
            
    else:
        logger.warning("No new ball-by-ball JSONs found yet. (Data providers usually sync 12-24 hours after a match ends).")
        logger.info("Base CSVs have been initialized to prevent startup warnings anyway.")

if __name__ == "__main__":
    asyncio.run(analyze_completed_matches())
