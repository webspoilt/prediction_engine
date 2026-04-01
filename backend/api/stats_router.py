from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import os
from typing import Optional

router = APIRouter(prefix="/ipl", tags=["ipl"])

# Load CSVs from the processed data directory
BASE_DIR = "data/processed"
MATCHES_PATH = os.path.join(BASE_DIR, "ipl_matches.csv")
BALLS_PATH = os.path.join(BASE_DIR, "ipl_ball.csv")

try:
    matches_df = pd.read_csv(MATCHES_PATH)
    balls_df = pd.read_csv(BALLS_PATH)
except Exception as e:
    print(f"⚠️ Warning: Could not load IPL datasets: {e}")
    matches_df = pd.DataFrame()
    balls_df = pd.DataFrame()

@router.get("/teams")
async def get_teams():
    if matches_df.empty:
        return {"teams": []}
    teams = sorted(list(set(matches_df["team1"].unique().tolist() + matches_df["team2"].unique().tolist())))
    return {"teams": teams}

@router.get("/players")
async def get_players(search: Optional[str] = Query(None)):
    if balls_df.empty:
        return {"players": [], "total": 0}
        
    batsmen = balls_df["batsman"].unique().tolist()
    bowlers = balls_df["bowler"].unique().tolist()
    all_players = list(set(batsmen + bowlers))
    
    if search:
        search_lower = search.lower()
        all_players = [p for p in all_players if search_lower in p.lower()]
        
    return {"players": sorted(all_players), "total": len(all_players)}

@router.get("/player/{name}")
async def get_player_stats(name: str):
    if balls_df.empty:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
        
    # Batting Stats
    batting = balls_df[balls_df["batsman"] == name]
    if batting.empty and balls_df[balls_df["bowler"] == name].empty:
        raise HTTPException(status_code=404, detail="Player not found")
        
    batting_stats = {
        "matches": batting["match_id"].nunique(),
        "runs": int(batting["runs_batter"].sum()),
        "fours": len(batting[batting["runs_batter"] == 4]),
        "sixes": len(batting[batting["runs_batter"] == 6]),
        "strike_rate": round((batting["runs_batter"].sum() / len(batting)) * 100, 2) if len(batting) > 0 else 0
    }
    
    # Bowling Stats
    bowling = balls_df[balls_df["bowler"] == name]
    bowling_stats = {
        "matches": bowling["match_id"].nunique(),
        "wickets": int(bowling[bowling["wicket"] == 1].shape[0]),
        "economy": round((bowling["runs_total"].sum() / (len(bowling) / 6)), 2) if len(bowling) > 0 else 0
    }
    
    return {
        "player": name,
        "batting": batting_stats,
        "bowling": bowling_stats
    }

@router.get("/archive")
async def get_historical_matches(season: Optional[int] = None):
    if matches_df.empty:
        return {"matches": []}
        
    df = matches_df.copy()
    if season:
        df = df[df["season"] == season]
        
    return {"matches": df.to_dict("records")}
