import asyncio
import logging
import json
import os
import redis
from typing import List, Dict
from scrapling import Fetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target URL for live scores
LIVE_SCORES_URL = "https://www.cricbuzz.com/cricket-match/live-scores"
IPL_SERIES_NAME = "Indian Premier League"

class MatchDiscoveryService:
    """
    Background worker that scans for live IPL matches.
    If a match is found, it registers it in Redis and triggers scraping.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.poll_interval = 300  # 5 minutes
        self.active_scrapers = {} # match_id -> Task

    async def run(self):
        """Main loop for match discovery"""
        logger.info("🚀 Starting Automated Match Discovery Service...")
        while True:
            try:
                live_matches = self._find_live_ipl_matches()
                logger.info(f"🔎 Discovery found {len(live_matches)} active IPL matches.")
                
                for match in live_matches:
                    match_id = match['match_id']
                    if not self.redis_client.exists(f"active:match:{match_id}"):
                        logger.info(f"🟢 New Match Detected! {match['teams']} - {match_id}")
                        self._register_match(match)
                
                # Cleanup finished matches (simplified)
                # In a real app, we would check if match status is 'result' or 'complete'
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                
            await asyncio.sleep(self.poll_interval)

    def _find_live_ipl_matches(self) -> List[Dict]:
        """Scrape Cricbuzz to find live IPL matches"""
        found_matches = []
        try:
            # Using Scrapling Fetcher (impersonates browser internally)
            fetcher = Fetcher(auto_match=True)
            page = fetcher.get(LIVE_SCORES_URL)
            
            # Find the IPL section
            # Cricbuzz structure: Series headings are usually in h2 or specific classes
            # We look for containers containing "Indian Premier League"
            
            # This is a heuristic parser for Cricbuzz Live Score page
            series_containers = page.css('.cb-mtch-lst') # Match list containers
            
            for container in series_containers:
                series_link = container.css('.cb-lv-grps-hdr a::text').get()
                if series_link and IPL_SERIES_NAME.lower() in series_link.lower():
                    # This cluster belongs to IPL
                    matches = container.css('.cb-col-100.cb-col.cb-schdl-itm')
                    for match in matches:
                        match_link = match.css('a::attr(href)').get()
                        if match_link:
                            # Extract Match ID from URL like '/live-cricket-scores/12345/match-name'
                            parts = match_link.split('/')
                            m_id = parts[2] if len(parts) > 2 else "unknown"
                            
                            match_url = f"https://www.cricbuzz.com{match_link}"
                            teams = match.css('.cb-lv-scr-mtch-hdr a::text').get()
                            
                            found_matches.append({
                                'match_id': m_id,
                                'url': match_url,
                                'teams': teams,
                                'status': 'live'
                            })
                            
            return found_matches
            
        except Exception as e:
            logger.error(f"Scraping error in _find_live_ipl_matches: {e}")
            return []

    def _register_match(self, match: Dict):
        """Store match in Redis and notify the system"""
        m_id = match['match_id']
        key = f"active:match:{m_id}"
        
        # Set with expiration (matches rarely last > 8 hours)
        self.redis_client.hset(key, mapping=match)
        self.redis_client.expire(key, 28800) # 8 hours
        
        # Also add to a global list of active matches
        self.redis_client.sadd("active:matches:set", m_id)
        
        # Publish notification for any listening backend workers
        self.redis_client.publish("match:discovered", json.dumps(match))

async def main():
    service = MatchDiscoveryService()
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())
