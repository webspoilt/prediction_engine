import time
import json
import logging
import asyncio
from typing import Dict, Optional
from scrapling import Fetcher
import redis
from backend.data_pipeline.ws_sniffer import BallData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESPNCricinfoScraper:
    """
    High-performance scraper for ESPNcricinfo live match states.
    Uses BS4/Scrapling for low-latency extraction from the live scoreboard.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, poll_interval: int = 5):
        self.poll_interval = poll_interval
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.is_running = False

    async def start_polling(self, match_id: str, url: str):
        """Starts polling an ESPNcricinfo live match URL"""
        self.is_running = True
        logger.info(f"🚀 Started ESPN Scraper for {match_id} at {url}")
        
        # Use curl_cffi to avoid playwright dependency in production
        fetcher = Fetcher(adapter='curl_cffi', auto_match=True)
        
        while self.is_running:
            try:
                # Get the live score page
                page = await asyncio.to_thread(fetcher.get, url)
                
                # ESPN Cricinfo Live Score Selectors
                # Score is typically in .ds-text-compact-m or similar ds-* classes
                score_container = page.css('.ds-flex.ds-flex-col.ds-mt-2.ds-mb-2')
                
                if not score_container:
                    # Fallback to older/alternative selectors
                    score_text = page.css('.ds-text-compact-m.ds-text-typo.ds-text-right.ds-whitespace-nowrap::text').get()
                else:
                    score_text = score_container.css('.ds-text-compact-m::text').get()

                # Overs often in a summary block
                overs_text = page.css('.ds-text-compact-s.ds-text-typo-mid3::text').get()
                
                # Team names
                teams = page.css('.ds-text-tight-l.ds-font-bold::text').getall()
                batting_team = teams[0] if len(teams) > 0 else "Unknown"
                bowling_team = teams[1] if len(teams) > 1 else "Unknown"

                if score_text and '/' in score_text:
                    parts = score_text.split('/')
                    runs = int(parts[0])
                    wickets = int(parts[1]) if parts[1].isdigit() else 0
                    
                    # Clean overs text like "(20.0 ov)"
                    overs = 0.0
                    if overs_text:
                        import re
                        match = re.search(r"(\d+\.\d+)", overs_text)
                        if match:
                            overs = float(match.group(1))

                    ball_data = BallData(
                        match_id=match_id,
                        inning=1, # simplified, detection logic needed
                        over=overs,
                        batsman="Live Player", # Placeholder, requires deep parsing
                        bowler="Live Bowler",
                        runs=0, # This scraper is for total state; delta logic would track ball-by-ball
                        extras=0,
                        wicket=False,
                        wicket_type=None,
                        timestamp=time.time(),
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        total_runs=runs,
                        total_wickets=wickets
                    )
                    
                    await self._publish_to_redis(ball_data)
                    logger.debug(f"ESPN Scraped: {runs}/{wickets} in {overs} ov")

            except Exception as e:
                logger.error(f"Error in ESPN Scraper for {match_id}: {e}")
                
            await asyncio.sleep(self.poll_interval)

    async def _publish_to_redis(self, ball_data: BallData):
        """Update live match stream and state"""
        stream_key = f"ipl:balls:{ball_data.match_id}"
        self.redis_client.xadd(stream_key, ball_data.to_dict(), maxlen=20) # Only keep last few for live view
        # Publish to live channel
        self.redis_client.publish(f"ipl:live:{ball_data.match_id}", json.dumps(ball_data.to_dict()))

    def stop(self):
        self.is_running = False
        logger.info("ESPN Scraper stopped.")

if __name__ == "__main__":
    # Test execution
    scraper = ESPNCricinfoScraper(poll_interval=2)
    test_url = "https://www.espncricinfo.com/series/indian-premier-league-2024-1410320/mumbai-indians-vs-delhi-capitals-20th-match-1422138/live-cricket-score"
    asyncio.run(scraper.start_polling("test_espn", test_url))
