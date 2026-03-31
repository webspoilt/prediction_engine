import os
import json
import zipfile
import urllib.request
import tempfile
import logging
import shutil
from datetime import datetime, timedelta
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"

def fetch_recent_matches(output_dir: str = "data", last_n_days: int = 2) -> List[str]:
    """
    Downloads the IPL JSON zip from Cricsheet and extracts matches 
    played within the last N days to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    fetched_files = []
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=last_n_days)
    logger.info(f"Looking for matches played on or after {cutoff_date.date()}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "ipl.zip")
        
        logger.info(f"Downloading cricsheet zip from {CRICSHEET_URL}...")
        try:
            urllib.request.urlretrieve(CRICSHEET_URL, zip_path)
        except Exception as e:
            logger.error(f"Failed to download Cricsheet data: {e}")
            return []
            
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        # Parse extracted JSON files
        json_files = [f for f in os.listdir(tmp_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} match files. Filtering recent matches...")
        
        for file in json_files:
            file_path = os.path.join(tmp_dir, file)
            # Skip README and other non-match json if any
            if file == "README.json":
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                dates = data.get('info', {}).get('dates', [])
                if not dates:
                    continue
                    
                # Usually, dates[0] is the match start date (YYYY-MM-DD)
                match_date_str = dates[0]
                match_date = datetime.strptime(match_date_str, "%Y-%m-%d")
                
                # Check if it falls within our targeted timeframe
                if match_date.date() >= cutoff_date.date():
                    dest_path = os.path.join(output_dir, file)
                    shutil.copy2(file_path, dest_path)
                    fetched_files.append(dest_path)
                    logger.info(f"✅ Fetched recent match: {file} (Date: {match_date_str})")
            except Exception as e:
                logger.warning(f"Could not parse match file {file}: {e}")
                
    logger.info(f"Successfully fetched {len(fetched_files)} recent matches.")
    return fetched_files

if __name__ == "__main__":
    new_files = fetch_recent_matches(output_dir="data", last_n_days=3)
    print("New files added:", new_files)
