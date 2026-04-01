import asyncio
import logging
import json
import redis
import time
from typing import List, Dict

# High-speed JSON API client
from backend.data_pipeline.cricbuzz_api import CricbuzzAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchDiscoveryService:
    """
    Background worker that schedules itself based on the Indian Premier League timetable.
    Eliminates constant 5-second scraping polling.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.active_scrapers = {} # match_id -> Task
        self.default_poll = 300  # Fallback 5 mins if no schedule found
        self.sync_interval = 60  # API Fallback sync every 60s

    async def run(self):
        """Main loop: Find live matches or sleep until the next scheduled match."""
        logger.info("🚀 Starting Pro V3 Intelligent Match Discovery Service...")
        
        # Start the API Fallback Sync Worker
        asyncio.create_task(self._api_sync_worker())
        
        while True:
            try:
                # 0. HARD FALLBACK: Register next 5 matches from static schedule in Redis
                # This ensures the UI is NEVER empty.
                upcoming_static = self._get_local_schedule()
                for match in upcoming_static[:5]:
                    m_id = match['match_id']
                    if not self.redis_client.exists(f"active:match:{m_id}"):
                        # Register as 'scheduled' status
                        match['status'] = 'scheduled'
                        self._register_match(match)
                        logger.info(f"💾 Registered static match: {match['teams']}")

                # 1. Instantly check via lightweight JSON if a match is ALREADY live
                live_matches = await CricbuzzAPI.get_live_matches()
                
                if live_matches:
                    logger.info(f"🔎 JSON API found {len(live_matches)} active IPL matches.")
                    for match in live_matches:
                        match_id = match['match_id']
                        if not self.redis_client.exists(f"active:match:{match_id}"):
                            logger.info(f"🟢 New Match Detected! {match['teams']} - {match_id}")
                            self._register_match(match)
                            
                    # While matches are live, poll every 5 mins to check for new ones or closures
                    await asyncio.sleep(self.default_poll)
                    continue

                # 2. No live matches? Fetch schedule and calculate exact sleep time!
                upcoming_matches = await CricbuzzAPI.get_match_schedule()
                
                # FALLBACK strictly to our local, verified 2026 Timetable if API is empty
                if not upcoming_matches:
                    upcoming_matches = self._get_local_schedule()
                    
                if upcoming_matches:
                    next_match = upcoming_matches[0]
                    # start_time is usually epoch seconds or milliseconds. 
                    # If it's a 13-digit number, it's milliseconds.
                    start_epoch = next_match['start_time']
                    if start_epoch > 10000000000:
                        start_epoch = start_epoch / 1000.0
                        
                    time_until_match = start_epoch - time.time()
                    
                    if time_until_match > 600:
                        # Sleep until 10 minutes before the match!
                        sleep_seconds = time_until_match - 600
                        logger.info(f"⏳ No matches live. Next match ({next_match['teams']}) starts in {int(time_until_match/60)} mins.")
                        logger.info(f"😴 Engine pausing discovery loop for {int(sleep_seconds)} seconds. Zzz...")
                        await asyncio.sleep(sleep_seconds)
                        continue
                        
                # 3. Fallback: If no schedule and no live matches, poll every 5 minutes
                await asyncio.sleep(self.default_poll)

            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(self.default_poll)

    async def _api_sync_worker(self):
        """
        Background task that ensures Redis has live scores via API if scrapers fail.
        This provides the 'Free API Fallback' the user requested.
        """
        logger.info("🔄 API Fallback Sync Worker started.")
        while True:
            try:
                active_match_ids = self.redis_client.smembers("active:matches:set")
                for m_id in active_match_ids:
                    # Check if scraper is effectively working (has detailed balls in stream)
                    ball_stream = f"ipl:balls:{m_id}"
                    last_ball = self.redis_client.xrevrange(ball_stream, count=1)
                    
                    # If no balls in last 30s or no balls at all, fetch via API
                    needs_sync = False
                    if not last_ball:
                        needs_sync = True
                    else:
                        ts = int(last_ball[0][0].split('-')[0]) / 1000.0
                        if time.time() - ts > 60: # No update for 60s
                            needs_sync = True
                    
                    if needs_sync:
                        logger.info(f"📡 Scraper stale for {m_id}. Fetching API Fallback score...")
                        score = await CricbuzzAPI.get_match_score(m_id)
                        if score:
                            # Push basic state to the ball stream
                            self.redis_client.xadd(ball_stream, score)
                            logger.info(f"✅ Synced API score for {m_id}: {score['total_runs']}/{score['total_wickets']}")
                
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"API Sync Worker error: {e}")
                await asyncio.sleep(self.sync_interval)

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

    def _get_local_schedule(self) -> List[Dict]:
        """Parses the official IPL 2026 CSV timetable into exact Epoch timestamps."""
        upcoming = []
        try:
            import csv
            import os
            from datetime import datetime, timezone, timedelta
            
            # IST is UTC + 5:30
            ist_offset = timedelta(hours=5, minutes=30)
            ist_tz = timezone(ist_offset)
            
            csv_path = os.path.join(os.path.dirname(__file__), 'ipl_2026_schedule.csv')
            
            if not os.path.exists(csv_path):
                return []
                
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = f"{row['Date']} {row['Time (IST)']}"
                    try:
                        dt = datetime.strptime(date_str, "%b %d, %Y %I:%M %p")
                        dt_ist = dt.replace(tzinfo=ist_tz)
                        start_epoch = dt_ist.timestamp()
                        
                        if start_epoch > time.time():
                            upcoming.append({
                                'match_id': f"official_{row['Match']}",
                                'teams': row['Match details'],
                                'start_time': start_epoch
                            })
                    except Exception as parse_err:
                        continue
                        
            upcoming.sort(key=lambda x: x['start_time'])
        except Exception as e:
            logger.error(f"Local schedule fallback error: {e}")
        return upcoming

async def main():
    service = MatchDiscoveryService()
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())
