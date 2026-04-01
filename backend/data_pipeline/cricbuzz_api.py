import asyncio
import logging
import json
from curl_cffi.requests import AsyncSession
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cricbuzz.com/",
        "Origin": "https://www.cricbuzz.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    @classmethod
    async def get_live_matches_espn(cls) -> List[Dict]:
        """Discovery using ESPN-Cricinfo Consumer API with Chrome Impersonation (bypass 403)."""
        url = "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/current?lang=en&clubId=null"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    matches = []
                    for m_info in data.get('matches', []):
                        series = m_info.get('series', {})
                        if "Indian Premier League" in series.get('name', '') or "IPL" in series.get('name', ''):
                            if m_info.get('status') == 'Live':
                                matches.append({
                                    'match_id': str(m_info.get('id')),
                                    'teams': f"{m_info.get('teams', [{}, {}])[0].get('team', {}).get('name', 'T1')} vs {m_info.get('teams', [{}, {}])[1].get('team', {}).get('name', 'T2')}",
                                    'url': f"https://www.espncricinfo.com/series/ipl-2026-1410320/match-slug-{m_info.get('id')}/live-score"
                                })
                    return matches
        except Exception as e:
            print(f"DEBUG: ESPN Consumer API (curl_cffi) failed: {e}")
        return []

    @classmethod
    async def get_live_matches(cls) -> List[Dict]:
        """Primary discovery method: Try ESPN then fallback to Cricbuzz."""
        espn_matches = await cls.get_live_matches_espn()
        if espn_matches:
            return espn_matches
            
        url = f"{cls.BASE_URL}/livematches.json"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    matches = []
                    for match_id, match_data in data.get('matches', {}).items():
                        series_name = match_data.get('series', {}).get('name', '').lower()
                        if 'indian premier league' in series_name or 'ipl' in series_name:
                            status = match_data.get('header', {}).get('state', '').lower()
                            if status == 'inprogress' or 'live' in status:
                                team1 = match_data.get('team1', {}).get('name', 'Team A')
                                team2 = match_data.get('team2', {}).get('name', 'Team B')
                                matches.append({
                                    'match_id': str(match_id),
                                    'teams': f"{team1} vs {team2}",
                                    'status': 'live',
                                    'url': f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/ipl-match"
                                })
                    return matches
                else:
                    logger.warning(f"Cricbuzz live matches API failed with status {resp.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching live matches from Cricbuzz API (curl_cffi): {e}")
            return []

    @classmethod
    async def get_match_score(cls, match_id: str) -> Optional[Dict]:
        """Fetch latest score for a specific match via JSON API."""
        url = f"{cls.BASE_URL}/livematches.json"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    match_data = data.get('matches', {}).get(str(match_id))
                    if not match_data:
                        return None
                    score_data = match_data.get('score', {})
                    return {
                        'match_id': match_id,
                        'batting_team': match_data.get('team1', {}).get('name', 'T1'), # fallback
                        'bowling_team': match_data.get('team2', {}).get('name', 'T2'),
                        'total_runs': int(score_data.get('runs', 0)),
                        'total_wickets': int(score_data.get('wickets', 0)),
                        'over': float(score_data.get('overs', 0.0)),
                        'inning': int(score_data.get('inning', 1))
                    }
        except Exception as e:
            logger.error(f"Error fetching match score for {match_id} (curl_cffi): {e}")
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
