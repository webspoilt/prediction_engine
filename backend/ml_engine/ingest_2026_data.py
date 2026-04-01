import pandas as pd
import json
import os
import logging
import numpy as np
from typing import List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_2026_data(data_dir: str, db_path: str):
    """
    Ingests 2026 IPL player and match data from CSVs into the internal stats DB.
    """
    logger.info(f"🚀 Starting 2026 Data Ingestion from {data_dir}...")
    
    # 1. Update Player Database
    player_csv = os.path.join(data_dir, "IPL_2026_All_Players.csv")
    if os.path.exists(player_csv):
        _update_player_db(player_csv, db_path)
    else:
        logger.warning(f"⚠️ Player CSV not found: {player_csv}")

    # 2. Process Ball-by-Ball and Results for Model Training (Future phase)
    # For now, we update the player stats using the ball-by-ball CSVs
    ball_by_ball_files = list(Path(data_dir).glob("*BallByBall*.csv"))
    if ball_by_ball_files:
        _process_match_data(ball_by_ball_files, db_path)
    else:
        logger.warning("⚠️ No ball-by-ball CSVs found.")

def _update_player_db(csv_path: str, db_path: str):
    """Integrates 2026 player info into the stats database."""
    logger.info(f"📝 Updating player info from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(db_path):
        logger.info("Initializing new player database...")
        db = {"batsmen": {}, "bowlers": {}, "global_baselines": {}}
    else:
        with open(db_path, 'r') as f:
            db = json.load(f)

    # Baselines for new players
    baselines = db.get("global_baselines", {
        "bat_sr": 130.0, "bat_avg": 25.0, "bat_bound_pct": 12.0,
        "bowl_econ": 8.5, "bowl_sr": 20.0, "bowl_avg": 28.0
    })

    new_count = 0
    for _, row in df.iterrows():
        name = row['Player_Name']
        role = str(row['Role']).lower()
        
        # Add to batsmen if they bat
        if name not in db['batsmen'] and ('batter' in role or 'all-rounder' in role or 'wk' in role):
            db['batsmen'][name] = {
                "runs": 0, "boundaries": 0, "balls_faced": 0, "dismissals": 0,
                "strike_rate": baselines['bat_sr'],
                "average": baselines['bat_avg'],
                "boundary_pct": baselines['bat_bound_pct'],
                "team_2026": row['Team_Code']
            }
            new_count += 1
            
        # Add to bowlers if they bowl
        if name not in db['bowlers'] and ('bowler' in role or 'all-rounder' in role):
            db['bowlers'][name] = {
                "runs_conceded": 0, "balls_bowled": 0, "wickets": 0,
                "economy": baselines['bowl_econ'],
                "strike_rate": baselines['bowl_sr'],
                "average": baselines['bowl_avg'],
                "team_2026": row['Team_Code']
            }
            new_count += 1

    with open(db_path, 'w') as f:
        json.dump(db, f, indent=4)
    logger.info(f"✅ Player database updated with {new_count} new entries.")

def _process_match_data(csv_files: List[Path], db_path: str):
    """Extracts actual 2026 performance stats from ball-by-ball files."""
    logger.info(f"📊 Processing {len(csv_files)} 2026 match files...")
    
    with open(db_path, 'r') as f:
        db = json.load(f)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Mapping: Over, Inns, Batting_Team, Bowling_Team, Bowler, Batsman, Runs_Scored, Total_Runs, Total_Wickets, Wicket_Type
        
        # Update Batsmen
        for batter, group in df.groupby('Batsman'):
            runs = group['Runs_Scored'].sum()
            balls = len(group) # Approx
            boundaries = len(group[group['Runs_Scored'] >= 4])
            wickets = len(group[group['Dismissed_Batsman'] == batter])
            
            if batter in db['batsmen']:
                stats = db['batsmen'][batter]
                stats['runs'] += int(runs)
                stats['balls_faced'] += int(balls)
                stats['boundaries'] += int(boundaries)
                stats['dismissals'] += int(wickets)
                # Recalculate SR/Avg
                if stats['balls_faced'] > 0:
                    stats['strike_rate'] = (stats['runs'] / stats['balls_faced']) * 100
                if stats['dismissals'] > 0:
                    stats['average'] = stats['runs'] / stats['dismissals']
                else:
                    stats['average'] = stats['runs']
        
        # Update Bowlers
        for bowler, group in df.groupby('Bowler'):
            runs_conced = group['Runs_Scored'].sum() # Basic
            balls_bowl = len(group)
            wickets = len(group[group['Wicket_Type'].notna() & (group['Wicket_Type'] != 'Run Out')])
            
            if bowler in db['bowlers']:
                stats = db['bowlers'][bowler]
                stats['runs_conceded'] += int(runs_conced)
                stats['balls_bowled'] += int(balls_bowl)
                stats['wickets'] += int(wickets)
                # Recalculate Econ/Avg
                if stats['balls_bowled'] > 0:
                    stats['economy'] = (stats['runs_conceded'] / (stats['balls_bowled'] / 6))
                if stats['wickets'] > 0:
                    stats['strike_rate'] = stats['balls_bowled'] / stats['wickets']
                    stats['average'] = stats['runs_conceded'] / stats['wickets']

    with open(db_path, 'w') as f:
        json.dump(db, f, indent=4)
    logger.info("✅ Player stats updated with 2026 performance data.")

if __name__ == "__main__":
    DATA_DIR = r"e:\IDEAS\ipl prediction engine\completed match"
    DB_PATH = r"e:\IDEAS\ipl prediction engine\backend\ml_engine\player_stats_db.json"
    ingest_2026_data(DATA_DIR, DB_PATH)
