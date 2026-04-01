import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

class CricbuzzAPI:
    """
    Zero-scraping overhead API client for Cricbuzz.
    Uses hidden JSON endpoints for Live Scores and Schedules.
    """
    BASE_URL = "https://www.cricbuzz.com/match-api"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.cricbuzz.com",
        "Referer": "https://www.cricbuzz.com/cricket-match/live-scores"
    }

    @classmethod
    async def get_live_matches(cls) -> List[Dict]:
        """Fetch all currently live cricket matches natively via JSON."""
        url = f"{cls.BASE_URL}/livematches.json"
        
        async with aiohttp.ClientSession(headers=cls.HEADERS) as session:
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        matches = []
                        
                        # Cricbuzz livematches.json structure roughly has a 'matches' object
                        for match_id, match_data in data.get('matches', {}).items():
                            series_name = match_data.get('series', {}).get('name', '').lower()
                            
                            # Filter for IPL matches
                            if 'indian premier league' in series_name or 'ipl' in series_name:
                                status = match_data.get('header', {}).get('state', '').lower()
                                
                                if status == 'inprogress' or 'live' in status:
                                    team1 = match_data.get('team1', {}).get('name', 'Team A')
                                    team2 = match_data.get('team2', {}).get('name', 'Team B')
                                    
                                    matches.append({
                                        'match_id': str(match_id),
                                        'teams': f"{team1} vs {team2}",
                                        'status': 'live',
                                        # Synthesize a fallback URL
                                        'url': f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/ipl-match"
                                    })
                        return matches
                    else:
                        logger.warning(f"Cricbuzz live matches API failed with status {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Error fetching live matches from Cricbuzz API: {e}")
                return []

    @classmethod
    async def get_match_score(cls, match_id: str) -> Optional[Dict]:
        """Fetch latest score for a specific match via JSON API."""
        url = f"{cls.BASE_URL}/livematches.json"
        
        async with aiohttp.ClientSession(headers=cls.HEADERS) as session:
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        match_data = data.get('matches', {}).get(str(match_id))
                        if not match_data:
                            return None
                            
                        header = match_data.get('header', {})
                        team1 = match_data.get('team1', {}).get('name', 'Team A')
                        team2 = match_data.get('team2', {}).get('name', 'Team B')
                        
                        # Cricbuzz score structure is often nested in 'innings' or 'score'
                        # For brevity, we'll try to extract common fields
                        score_data = match_data.get('score', {})
                        runs = score_data.get('runs', 0)
                        wickets = score_data.get('wickets', 0)
                        overs = score_data.get('overs', 0.0)
                        
                        # Determine batting team
                        batting_team = team1 if score_data.get('batting_team_id') == match_data.get('team1', {}).get('id') else team2

                        return {
                            'match_id': match_id,
                            'batting_team': batting_team,
                            'bowling_team': team2 if batting_team == team1 else team1,
                            'total_runs': int(runs) if runs else 0,
                            'total_wickets': int(wickets) if wickets else 0,
                            'over': float(overs) if overs else 0.0,
                            'inning': int(score_data.get('inning', 1)) 
                        }
            except Exception as e:
                logger.error(f"Error fetching match score for {match_id}: {e}")
        return None

    @classmethod
    async def get_match_schedule(cls) -> List[Dict]:
        """Fetch the upcoming match schedule to determine when to wake up the engine."""
        # For simplicity, we can fetch the series schedule (if known series ID)
        # Alternatively, hit the upcoming matches JSON
        # Since the live API also lists upcoming today:
        url = f"{cls.BASE_URL}/livematches.json"
        
        async with aiohttp.ClientSession(headers=cls.HEADERS) as session:
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        upcoming_ipl = []
                        
                        for match_id, match_data in data.get('matches', {}).items():
                            series_name = match_data.get('series', {}).get('name', '').lower()
                            if 'indian premier league' in series_name or 'ipl' in series_name:
                                header = match_data.get('header', {})
                                state = header.get('state', '').lower()
                                
                                if state == 'preview' or state == 'upcoming':
                                    # Cricbuzz usually provides start_time in epoch
                                    start_time = header.get('start_time')
                                    if start_time:
                                        team1 = match_data.get('team1', {}).get('name', '')
                                        team2 = match_data.get('team2', {}).get('name', '')
                                        upcoming_ipl.append({
                                            'match_id': str(match_id),
                                            'teams': f"{team1} vs {team2}",
                                            'start_time': int(start_time)
                                        })
                                        
                        # Sort by soonest upcoming
                        upcoming_ipl.sort(key=lambda x: x['start_time'])
                        return upcoming_ipl
                    return []
            except Exception as e:
                logger.error(f"Error fetching schedule from Cricbuzz API: {e}")
                return []
