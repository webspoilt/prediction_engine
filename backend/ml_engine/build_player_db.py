import pandas as pd
import json
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_player_stats(csv_path: str, output_path: str):
    logger.info(f"Loading ball by ball data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        return

    logger.info("Calculating Batsman Statistics...")
    
    # --- Batsman Stats ---
    # Convert string booleans to actual booleans if they aren't already
    if df['is_wide_ball'].dtype == object:
        df['is_wide_ball'] = df['is_wide_ball'].astype(str).str.lower() == 'true'
    if df['is_no_ball'].dtype == object:
        df['is_no_ball'] = df['is_no_ball'].astype(str).str.lower() == 'true'
    if df['is_wicket'].dtype == object:
        df['is_wicket'] = df['is_wicket'].astype(str).str.lower() == 'true'

    # Filter out wides for balls faced
    balls_faced_df = df[~df['is_wide_ball']]
    
    # Calculate runs, balls, boundaries
    batter_stats = df.groupby('batter').agg(
        runs=('batter_runs', 'sum'),
        boundaries=('batter_runs', lambda x: ((x == 4) | (x == 6)).sum())
    )
    
    # Boundaries count
    batter_balls = balls_faced_df.groupby('batter').size().rename('balls_faced')
    
    # Calculate dismissals (where player_out == batter)
    dismissals = df[df['is_wicket'] & (df['player_out'] == df['batter'])].groupby('batter').size().rename('dismissals')
    
    # Merge batsman stats
    bat_df = pd.concat([batter_stats, batter_balls, dismissals], axis=1).fillna(0)
    
    # Derived Batsman Defaults
    bat_df['strike_rate'] = np.where(bat_df['balls_faced'] > 0, (bat_df['runs'] / bat_df['balls_faced']) * 100, 0)
    bat_df['average'] = np.where(bat_df['dismissals'] > 0, bat_df['runs'] / bat_df['dismissals'], bat_df['runs'])
    bat_df['boundary_pct'] = np.where(bat_df['balls_faced'] > 0, (bat_df['boundaries'] / bat_df['balls_faced']) * 100, 0)

    logger.info("Calculating Bowler Statistics...")
    
    # --- Bowler Stats ---
    # Runs conceded: total - byes - leg byes
    df['bowler_runs'] = df['total_runs'] - df['bye_runs'].fillna(0) - df['leg_bye_runs'].fillna(0) - df['penalty_runs'].fillna(0)
    
    # Balls bowled: exclude wides and no balls
    legal_deliveries = df[~(df['is_wide_ball'] | df['is_no_ball'])]
    
    # Wickets: exclude run outs, retired hurts, etc. (we want pure bowler wickets)
    bowler_wickets_df = df[df['is_wicket'] & (~df['wicket_kind'].isin(['run out', 'retired hurt', 'obstructing the field', 'hit ball twice', 'handled ball', 'timed out']))]

    bowl_stats = df.groupby('bowler').agg(
        runs_conceded=('bowler_runs', 'sum')
    )
    bowl_balls = legal_deliveries.groupby('bowler').size().rename('balls_bowled')
    bowl_wickets = bowler_wickets_df.groupby('bowler').size().rename('wickets')
    
    # Merge bowler stats
    bl_df = pd.concat([bowl_stats, bowl_balls, bowl_wickets], axis=1).fillna(0)
    
    # Derived Bowler Stats
    bl_df['economy'] = np.where(bl_df['balls_bowled'] > 0, (bl_df['runs_conceded'] / (bl_df['balls_bowled'] / 6)), 0)
    bl_df['strike_rate'] = np.where(bl_df['wickets'] > 0, bl_df['balls_bowled'] / bl_df['wickets'], 0)
    bl_df['average'] = np.where(bl_df['wickets'] > 0, bl_df['runs_conceded'] / bl_df['wickets'], 0)
    
    # Format into JSON lookup dictionaries
    player_db = {
        "batsmen": bat_df.to_dict(orient='index'),
        "bowlers": bl_df.to_dict(orient='index')
    }
    
    # Add medians / global defaults for unknown players
    logger.info("Calculating global baselines for unknown players...")
    player_db['global_baselines'] = {
        "bat_sr": float(bat_df[bat_df['balls_faced'] >= 10]['strike_rate'].median() or 120.0),
        "bat_avg": float(bat_df[bat_df['dismissals'] >= 1]['average'].median() or 20.0),
        "bat_bound_pct": float(bat_df[bat_df['balls_faced'] >= 10]['boundary_pct'].median() or 10.0),
        "bowl_econ": float(bl_df[bl_df['balls_bowled'] >= 12]['economy'].median() or 8.0),
        "bowl_sr": float(bl_df[bl_df['wickets'] >= 1]['strike_rate'].median() or 24.0),
        "bowl_avg": float(bl_df[bl_df['wickets'] >= 1]['average'].median() or 30.0)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(player_db, f, indent=4)
        
    logger.info(f"Successfully built Player Stats DB with {len(bat_df)} batsmen and {len(bl_df)} bowlers!")
    logger.info(f"Saved to {output_path}")

def update_with_new_players(json_files, db_path: str):
    """
    Scans new Cricsheet JSON match files for players not currently in the DB.
    Initializes new players with the global baselines to prevent inference errors.
    """
    if not os.path.exists(db_path):
        logger.error(f"Player DB not found at {db_path}. Run build full first.")
        return

    with open(db_path, 'r') as f:
        player_db = json.load(f)

    baselines = player_db.get('global_baselines', {
        "bat_sr": 120.0, "bat_avg": 20.0, "bat_bound_pct": 10.0,
        "bowl_econ": 8.0, "bowl_sr": 24.0, "bowl_avg": 30.0
    })

    batsmen_db = player_db.get('batsmen', {})
    bowler_db = player_db.get('bowlers', {})

    new_batsmen_count = 0
    new_bowlers_count = 0

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                match_data = json.load(f)
                
            for inning in match_data.get('innings', []):
                for over in inning.get('overs', []):
                    for delivery in over.get('deliveries', []):
                        batter = delivery.get('batter')
                        bowler = delivery.get('bowler')
                        
                        if batter and batter not in batsmen_db:
                            batsmen_db[batter] = {
                                'runs': 0, 'boundaries': 0, 'balls_faced': 0, 'dismissals': 0,
                                'strike_rate': baselines['bat_sr'], 
                                'average': baselines['bat_avg'], 
                                'boundary_pct': baselines['bat_bound_pct']
                            }
                            new_batsmen_count += 1
                            logger.info(f"➕ Added new batsman from {os.path.basename(file_path)}: {batter}")
                            
                        if bowler and bowler not in bowler_db:
                            bowler_db[bowler] = {
                                'runs_conceded': 0, 'balls_bowled': 0, 'wickets': 0,
                                'economy': baselines['bowl_econ'],
                                'strike_rate': baselines['bowl_sr'],
                                'average': baselines['bowl_avg']
                            }
                            new_bowlers_count += 1
                            logger.info(f"➕ Added new bowler from {os.path.basename(file_path)}: {bowler}")
        except Exception as e:
            logger.warning(f"Error processing {file_path} for new players: {e}")

    if new_batsmen_count > 0 or new_bowlers_count > 0:
        player_db['batsmen'] = batsmen_db
        player_db['bowlers'] = bowler_db
        with open(db_path, 'w') as f:
            json.dump(player_db, f, indent=4)
        logger.info(f"✅ Updated Player DB: added {new_batsmen_count} batsmen, {new_bowlers_count} bowlers.")
    else:
        logger.info("ℹ️ No new players found in the recent matches.")

if __name__ == "__main__":
    csv_in = r"e:\IDEAS\ipl prediction engine\dataset\archive (2)\all_ball_by_ball_data.csv"
    json_out = r"e:\IDEAS\ipl prediction engine\backend\ml_engine\player_stats_db.json"
    build_player_stats(csv_in, json_out)
