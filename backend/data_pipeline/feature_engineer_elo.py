import pandas as pd
from pathlib import Path

import os

BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.parent))
MATCHES_CSV = BASE_DIR / "dataset/archive/matches.csv"
ENHANCED_DIR = BASE_DIR / "dataset/enhanced"
ELO_CSV = ENHANCED_DIR / "team_elo.csv"

# Hyperparameters for ELO
INITIAL_ELO = 1500
K_FACTOR = 20

def normalize_team_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    # Handle known franchise rebrands
    rebrands = {
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiants": "Rising Pune Supergiant" # Fix singular/plural
    }
    return rebrands.get(name, name)

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def main():
    print(f"Loading matches from {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    
    # Sort chronologically
    # Date might be object, parse it to datetime for sorting
    df['parsed_date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.sort_values('parsed_date').reset_index(drop=True)
    
    # Initialize ELO dictionary
    # Tracks the CURRENT ELO of each team
    team_elos = {}
    
    # We will build a list of records representing the PRE-MATCH ELO
    elo_records = []
    
    for idx, row in df.iterrows():
        match_id = row['id']
        team1 = normalize_team_name(row['team1'])
        team2 = normalize_team_name(row['team2'])
        winner = normalize_team_name(row['winner'])
        result = row['result'] # Can be 'normal', 'tie', 'no result'
        
        if pd.isna(team1) or pd.isna(team2):
            continue
            
        # Get current ELO, default to 1500
        elo1 = team_elos.get(team1, INITIAL_ELO)
        elo2 = team_elos.get(team2, INITIAL_ELO)
        
        # Record PRE-MATCH states
        elo_records.append({
            'match_id': match_id,
            'team1': row['team1'], # Keep original names in the output for easy merging
            'team2': row['team2'],
            'team1_elo_pre': round(elo1, 2),
            'team2_elo_pre': round(elo2, 2)
        })
        
        # Update ELO based on match result
        # Determine actual scores (1 for win, 0 for loss, 0.5 for tie/no result)
        if result == 'no result' or pd.isna(winner):
            # No result/ abandoned
            actual1, actual2 = 0.5, 0.5
        elif result == 'tie':
            # Super over or tie
            # We could look at who won the super over (winner column usually has super over winner)
            # but standard tie without winner is 0.5
            if not pd.isna(winner):
                actual1 = 1.0 if winner == team1 else 0.0
                actual2 = 1.0 if winner == team2 else 0.0
            else:
                actual1, actual2 = 0.5, 0.5
        else:
            # Normal result
            actual1 = 1.0 if winner == team1 else 0.0
            actual2 = 1.0 if winner == team2 else 0.0
            
        # Expectations
        exp1 = calculate_expected_score(elo1, elo2)
        exp2 = calculate_expected_score(elo2, elo1)
        
        # New ELO
        new_elo1 = elo1 + K_FACTOR * (actual1 - exp1)
        new_elo2 = elo2 + K_FACTOR * (actual2 - exp2)
        
        # Save to dict
        team_elos[team1] = new_elo1
        team_elos[team2] = new_elo2
        
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(elo_records)
    out_df.to_csv(ELO_CSV, index=False)
    
    print(f"Calculated ELO for {len(out_df)} matches.")
    print(f"Top Teams Current ELO:")
    sorted_elos = sorted(team_elos.items(), key=lambda x: x[1], reverse=True)
    for team, elo in sorted_elos[:5]:
        print(f"  {team}: {round(elo, 2)}")
        
    print(f"Saved ELO dataset to {ELO_CSV}")

if __name__ == "__main__":
    main()
