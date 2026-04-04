"""
Cricbuzz & ESPN API Client — curl_cffi Edition
================================================
All HTTP calls use curl_cffi with Chrome impersonation (bypass 403/Cloudflare).
Fixed: Removed broken aiohttp references from get_match_schedule().
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

from curl_cffi.requests import AsyncSession

logger = logging.getLogger(__name__)


class CricbuzzAPI:
    """
    Lightweight JSON API client for Cricbuzz & ESPN.
    Uses curl_cffi Chrome impersonation for all requests.
    """

    BASE_URL = "https://www.cricbuzz.com/match-api"
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cricbuzz.com/",
        "Origin": "https://www.cricbuzz.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    # ── In-Memory Response Cache ─────────────────────────────────────────────
    _cache: Dict[str, Dict] = {}  # key -> {"data": ..., "expires": float}
    CACHE_TTL = 30  # seconds
    MAX_CACHE_SIZE = 200

    @classmethod
    def _get_cached(cls, key: str) -> Optional[any]:
        entry = cls._cache.get(key)
        if entry:
            if time.time() < entry["expires"]:
                return entry["data"]
            else:
                del cls._cache[key]
        return None

    @classmethod
    def _set_cached(cls, key: str, data: any, ttl: int = None):
        if len(cls._cache) >= cls.MAX_CACHE_SIZE:
            now = time.time()
            expired = [k for k, v in cls._cache.items() if now > v["expires"]]
            for k in expired:
                del cls._cache[k]
            if len(cls._cache) >= cls.MAX_CACHE_SIZE:
                # Remove oldest entry
                oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k]["expires"])
                del cls._cache[oldest_key]
                
        cls._cache[key] = {"data": data, "expires": time.time() + (ttl or cls.CACHE_TTL)}

    # ── ESPN Consumer API ────────────────────────────────────────────────────

    @classmethod
    async def get_live_matches_espn(cls) -> List[Dict]:
        """Discovery using ESPN-Cricinfo Consumer API with Chrome Impersonation (bypass 403)."""
        cached = cls._get_cached("espn_live")
        if cached is not None:
            return cached

        url = "https://hs-consumer-api.espncricinfo.com/v1/pages/matches/current?lang=en&clubId=null"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    matches = []
                    for m_info in data.get("matches", []):
                        series = m_info.get("series", {})
                        series_name = series.get("name", "")
                        if "Indian Premier League" not in series_name and "IPL" not in series_name:
                            continue
                        if m_info.get("status") == "Live" or m_info.get("state") == "LIVE":
                            teams_list = m_info.get("teams", [{}, {}])
                            t1 = teams_list[0].get("team", {}).get("name", "T1") if len(teams_list) > 0 else "T1"
                            t2 = teams_list[1].get("team", {}).get("name", "T2") if len(teams_list) > 1 else "T2"
                            matches.append({
                                "match_id": str(m_info.get("id")),
                                "teams": f"{t1} vs {t2}",
                                "url": f"https://www.espncricinfo.com/series/ipl-2026-1410320/match-slug-{m_info.get('id')}/live-score",
                            })
                    cls._set_cached("espn_live", matches)
                    return matches
        except Exception as e:
            logger.warning(f"ESPN Consumer API (curl_cffi) failed: {e}")
        return []

    # ── Cricbuzz Live Matches ────────────────────────────────────────────────

    @classmethod
    async def get_live_matches(cls) -> List[Dict]:
        """Primary discovery method: Try ESPN then fallback to Cricbuzz."""
        espn_matches = await cls.get_live_matches_espn()
        if espn_matches:
            return espn_matches

        cached = cls._get_cached("cb_live")
        if cached is not None:
            return cached

        url = f"{cls.BASE_URL}/livematches.json"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    matches = []
                    for match_id, match_data in data.get("matches", {}).items():
                        series_name = match_data.get("series", {}).get("name", "").lower()
                        if "indian premier league" not in series_name and "ipl" not in series_name:
                            continue
                        status = match_data.get("header", {}).get("state", "").lower()
                        if status == "inprogress" or "live" in status:
                            team1 = match_data.get("team1", {}).get("name", "Team A")
                            team2 = match_data.get("team2", {}).get("name", "Team B")
                            matches.append({
                                "match_id": str(match_id),
                                "teams": f"{team1} vs {team2}",
                                "status": "live",
                                "url": f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/ipl-match",
                            })
                    cls._set_cached("cb_live", matches)
                    return matches
                else:
                    logger.warning(f"Cricbuzz live API returned {resp.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Cricbuzz live matches error (curl_cffi): {e}")
            return []

    # ── Match Score (per match) ──────────────────────────────────────────────

    @classmethod
    async def get_match_score(cls, match_id: str) -> Optional[Dict]:
        """Fetch latest score for a specific match via JSON API."""
        cached = cls._get_cached(f"score_{match_id}")
        if cached is not None:
            return cached

        url = f"{cls.BASE_URL}/livematches.json"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    match_data = data.get("matches", {}).get(str(match_id))
                    if not match_data:
                        return None
                    score_data = match_data.get("score", {})
                    result = {
                        "match_id": match_id,
                        "batting_team": match_data.get("team1", {}).get("name", "T1"),
                        "bowling_team": match_data.get("team2", {}).get("name", "T2"),
                        "total_runs": int(score_data.get("runs", 0)),
                        "total_wickets": int(score_data.get("wickets", 0)),
                        "over": float(score_data.get("overs", 0.0)),
                        "inning": int(score_data.get("inning", 1)),
                    }
                    cls._set_cached(f"score_{match_id}", result, ttl=15)
                    return result
        except Exception as e:
            logger.error(f"Score fetch error for {match_id} (curl_cffi): {e}")
        return None

    # ── Match Schedule (FIXED — was using aiohttp, now curl_cffi) ────────────

    @classmethod
    async def get_match_schedule(cls) -> List[Dict]:
        """Fetch the upcoming match schedule. Uses curl_cffi (fixed from aiohttp)."""
        cached = cls._get_cached("schedule")
        if cached is not None:
            return cached

        url = f"{cls.BASE_URL}/livematches.json"
        try:
            async with AsyncSession(impersonate="chrome110") as s:
                resp = await s.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    upcoming_ipl = []
                    for match_id, match_data in data.get("matches", {}).items():
                        series_name = match_data.get("series", {}).get("name", "").lower()
                        if "indian premier league" not in series_name and "ipl" not in series_name:
                            continue

                        header = match_data.get("header", {})
                        state = header.get("state", "").lower()

                        if state in ("preview", "upcoming", "upcoming_match"):
                            start_time = header.get("start_time")
                            if start_time:
                                team1 = match_data.get("team1", {}).get("name", "")
                                team2 = match_data.get("team2", {}).get("name", "")
                                upcoming_ipl.append({
                                    "match_id": str(match_id),
                                    "teams": f"{team1} vs {team2}",
                                    "start_time": int(start_time),
                                })

                    upcoming_ipl.sort(key=lambda x: x["start_time"])
                    cls._set_cached("schedule", upcoming_ipl, ttl=300)
                    return upcoming_ipl
                return []
        except Exception as e:
            logger.error(f"Schedule fetch error (curl_cffi): {e}")
            return []
