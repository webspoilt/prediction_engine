import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def consolidate_ipl_data(json_folder="data", output_csv="data/processed"):
    all_matches = []
    all_balls = []
    
    # Ensure processed directory exists
    Path(output_csv).mkdir(parents=True, exist_ok=True)
    
    json_files = list(Path(json_folder).glob("*.json"))
    print(f"🔄 Consolidating {len(json_files)} matches...")
    
    for json_file in tqdm(json_files):
        with open(json_file) as f:
            try:
                match = json.load(f)
            except:
                continue
        
        # Extract match metadata
        # Some older files might have different ID locations, using filename as fallback
        match_id = json_file.stem
        info = match.get("info", {})
        
        match_info = {
            "match_id": match_id,
            "season": info.get("season"),
            "city": info.get("city"),
            "team1": info.get("teams", ["N/A", "N/A"])[0],
            "team2": info.get("teams", ["N/A", "N/A"])[1],
            "winner": info.get("outcome", {}).get("winner"),
            "win_by_runs": info.get("outcome", {}).get("by", {}).get("runs"),
            "win_by_wickets": info.get("outcome", {}).get("by", {}).get("wickets"),
            "date": info.get("dates", [""])[0],
            "toss_winner": info.get("toss", {}).get("winner"),
            "toss_decision": info.get("toss", {}).get("decision")
        }
        all_matches.append(match_info)
        
        # Extract ball-by-ball data
        for inning_idx, inning in enumerate(match.get("innings", []), 1):
            batting_team = inning.get("team")
            for over in inning.get("overs", []):
                over_num = over.get("over")
                for ball in over.get("deliveries", []):
                    ball_data = {
                        "match_id": match_id,
                        "inning": inning_idx,
                        "batting_team": batting_team,
                        "bowling_team": match_info["team2"] if batting_team == match_info["team1"] else match_info["team1"],
                        "over": over_num,
                        "batsman": ball.get("batter"),
                        "bowler": ball.get("bowler"),
                        "non_striker": ball.get("non_striker"),
                        "runs_batter": ball.get("runs", {}).get("batter", 0),
                        "runs_total": ball.get("runs", {}).get("total", 0),
                        "extras": ball.get("runs", {}).get("extras", 0),
                        "wicket": 1 if "wickets" in ball else 0,
                        "player_dismissed": ball.get("wickets", [{}])[0].get("player_out") if "wickets" in ball else None,
                        "dismissal_kind": ball.get("wickets", [{}])[0].get("kind") if "wickets" in ball else None
                    }
                    all_balls.append(ball_data)
    
    print(f"💾 Saving to {output_csv}...")
    pd.DataFrame(all_matches).to_csv(f"{output_csv}/ipl_matches.csv", index=False)
    pd.DataFrame(all_balls).to_csv(f"{output_csv}/ipl_ball.csv", index=False)
    print(f"✅ Consolidated {len(all_matches)} matches and {len(all_balls)} ball entries.")

if __name__ == "__main__":
    consolidate_ipl_data()
