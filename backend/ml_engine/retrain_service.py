import os
import time
import logging
from datetime import datetime
from pathlib import Path
from backend.ml_engine.train_efficient import train_efficiently

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 3600))  # 1 hour
LAST_CHECK_FILE = "backend/ml_engine/last_retrain.txt"

def get_last_retrain_time():
    """Read the last successfull retrain timestamp"""
    if not os.path.exists(LAST_CHECK_FILE):
        return datetime.min
    try:
        with open(LAST_CHECK_FILE, 'r') as f:
            return datetime.fromisoformat(f.read().strip())
    except Exception as e:
        logger.warning(f"Could not read last_retrain file: {e}")
        return datetime.min

def update_last_retrain_time():
    """Save the current timestamp as the last retrain time"""
    with open(LAST_CHECK_FILE, 'w') as f:
        f.write(datetime.now().isoformat())

def check_for_new_data(last_time):
    """Scan the data directory for any files modified after last_time"""
    new_files = []
    # Search for JSON match files
    for file in Path(DATA_DIR).glob("**/*.json"):
        mod_time = datetime.fromtimestamp(file.stat().st_mtime)
        if mod_time > last_time:
            new_files.append(file)
    
    return new_files

def main_loop():
    logger.info(f"🚀 Starting Automated Retraining Service (Watching: {DATA_DIR})")
    
    while True:
        last_retrain = get_last_retrain_time()
        new_data = check_for_new_data(last_retrain)
        
        if new_data:
            logger.info(f"📈 Found {len(new_data)} new match files! Triggering retraining pipeline...")
            try:
                # Trigger the efficient training pipeline
                train_efficiently(data_dir=DATA_DIR)
                
                update_last_retrain_time()
                logger.info("✅ Retraining complete and model updated.")
            except Exception as e:
                logger.error(f"❌ Retraining failed: {e}")
        else:
            logger.info("😴 No new data found. Sleeping...")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
