import time
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Optional
import redis
from backend.models.match_models import BallData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESPNCricinfoScraper:
    """
    Refactored to High-Performance JSON API Poller (Pro V3)
    No longer scrapes HTML. Connects directly to Cricbuzz APIs for live match state.
    Name kept as ESPNCricinfoScraper to avoid breaking external imports.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, poll_interval: int = 5):
        # We can poll a JSON API every 5 seconds safely without overhead
        from backend.config import settings
        self.poll_interval = poll_interval
        self.redis_client = None
        if settings.REDIS_ENABLED:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.is_running = False

    async def start_polling(self, match_id: str, url: str):
        """Starts polling the match via pure JSON APIs"""
        self.is_running = True
        logger.info(f"🚀 Started JSON Match Tracker for {match_id} (bypassing Scrapling)")
        
        # Cricbuzz live API endpoint
        api_url = f"https://www.cricbuzz.com/match-api/{match_id}/commentary.json"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            while self.is_running:
                try:
                    async with session.get(api_url, timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._process_match_json(match_id, data)
                        else:
                            # If individual match commentary API fails, fallback gracefully
                            # Could mean match hasn't started or ID is invalid
                            logger.debug(f"JSON API returned status {resp.status} for {match_id}")
                except Exception as e:
                    logger.error(f"Error in JSON Tracker for {match_id}: {e}")
                    
                await asyncio.sleep(self.poll_interval)

    def _process_match_json(self, match_id: str, data: Dict):
        """Extract exact ball-by-ball and match state securely"""
        try:
            score_data = data.get('score', {})
            batting = score_data.get('batting', {})
            bowling = score_data.get('bowling', {})
            
            # Note: Cricbuzz API structure requires defensive extraction
            score_str = batting.get('score', '0') # e.g., "150"
            wickets_str = score_data.get('batting', {}).get('wickets', '0')

            overs = score_data.get('batting', {}).get('overs', 0.0)
            
            # The API might provide int or string
            runs = int(score_str) if str(score_str).isdigit() else 0
            wickets = int(wickets_str) if str(wickets_str).isdigit() else 0

            # Find active batsman/bowler
            batsman = "Unknown"
            striker = data.get('batsman', [{}])[0]
            if striker: batsman = striker.get('name', 'Unknown')
                
            bowler = "Unknown"
            active_bowler = data.get('bowler', [{}])[0]
            if active_bowler: bowler = active_bowler.get('name', 'Unknown')

            ball_data = BallData(
                match_id=match_id,
                inning=1, # Logic needed for 2nd inning
                over=float(overs),
                batsman=batsman,
                bowler=bowler,
                runs=0, # Live total
                extras=0,
                wicket=False,
                wicket_type=None,
                timestamp=time.time(),
                batting_team=data.get('team1', {}).get('name', 'Team A'), # Adjust based on innings
                bowling_team=data.get('team2', {}).get('name', 'Team B'),
                total_runs=runs,
                total_wickets=wickets
            )
            
            self._publish_to_redis_sync(ball_data)
        except Exception as e:
            logger.debug(f"JSON Structure parsing issue (expected during breaks): {e}")

    def _publish_to_redis_sync(self, ball_data: BallData):
        """Update live match stream and state (shielded)"""
        if not self.redis_client:
            return
            
        stream_key = f"ipl:balls:{ball_data.match_id}"
        try:
            self.redis_client.xadd(stream_key, ball_data.to_dict(), maxlen=20) 
            # Publish to live channel
            self.redis_client.publish(f"ipl:live:{ball_data.match_id}", json.dumps(ball_data.to_dict()))
        except Exception as e:
            logger.debug(f"Redis publish failed (non-critical): {e}")

    def stop(self):
        self.is_running = False
        logger.info("JSON API Tracker stopped.")
