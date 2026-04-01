"""
HTTP Fallback Scraper using Scrapling
IPL Win Probability Prediction Engine - Data Ingestion Module

This module provides a robust HTTP polling scraper using the Scrapling framework
to impersonate real browsers and bypass cloudflare/anti-bot measures.
It serves as a secondary fallback if the WebSocket sniffer is blocked.
"""

import time
import json
import logging
import asyncio
from typing import Dict, Optional

# Using Scrapling for stealthy fetching and precise parsing
from scrapling.fetchers import FetcherSession

import redis
from redis.asyncio import Redis as AsyncRedis

from backend.models.match_models import BallData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScraplingHttpScraper:
    """
    HTTP polling scraper using Scrapling.
    Polls live match URLs to extract scoreboard information using TLS fingerprinting.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, poll_interval: int = 5):
        self.poll_interval = poll_interval
        self.async_redis = AsyncRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.is_running = False

    async def start_polling(self, match_id: str, url: str):
        """
        Starts polling using Scrapling's FetcherSession directly taking advantage of HTTP/3
        and browser impersonation.
        """
        self.is_running = True
        logger.info(f"Started Scrapling HTTP polling for match {match_id} at {url}")
        
        # Open an async context-aware Scrapling session impersonating Chrome
        async with FetcherSession(http3=True, impersonate='chrome') as session:
            while self.is_running:
                try:
                    # session.get provides a stealthy response that behaves exactly like BeautifulSoup
                    page = await session.get(url, stealthy_headers=True)
                    ball_data = self._parse_page(match_id, page)
                    
                    if ball_data:
                        await self._publish_to_redis(ball_data)
                        logger.debug(f"Scraped via Scrapling: {ball_data.runs} runs, Wickets: {ball_data.total_wickets}")
                        
                except Exception as e:
                    logger.error(f"Error during Scrapling fetch for {url}: {e}")
                    
                await asyncio.sleep(self.poll_interval)
            
    def _parse_page(self, match_id: str, page) -> Optional[BallData]:
        """
        Parses the score using Scrapling's built-in CSS/XPath selector syntax.
        """
        try:
            # Example CSS selectors for aggregate scores
            # page.css() works just like BeautifulSoup/Scrapy
            score_text = page.css('.cb-font-20.text-bold::text').get() 
            overs_text = page.css('.cb-text-complete::text').get()
            
            if not score_text:
                return None
                
            runs_str, wickets_str = score_text.split('/')
            total_runs = int(runs_str)
            total_wickets = int(wickets_str) if wickets_str.isdigit() else 10
            
            overs = float(overs_text.split(' ')[0]) if overs_text else 0.0
            
            return BallData(
                match_id=match_id,
                inning=1,
                over=overs,
                batsman='Unknown',
                bowler='Unknown',
                runs=0,
                extras=0,
                wicket=False,
                wicket_type=None,
                timestamp=time.time(),
                batting_team='TBD',
                bowling_team='TBD',
                total_runs=total_runs,
                total_wickets=total_wickets
            )
        except Exception as parse_error:
            logger.debug(f"Could not parse score: {parse_error}")
            return None

    async def _publish_to_redis(self, ball_data: BallData):
        """Publish ball data to Redis Stream, matching the WebSocket format"""
        stream_key = f"ipl:balls:{ball_data.match_id}"
        
        # Publish to stream
        await self.async_redis.xadd(
            stream_key,
            ball_data.to_dict(),
            maxlen=1000
        )
        
        # Publish to live channel
        await self.async_redis.publish(
            f"ipl:live:{ball_data.match_id}",
            json.dumps(ball_data.to_dict())
        )

    async def stop(self):
        """Stop polling"""
        self.is_running = False
        await self.async_redis.close()
        logger.info("Scrapling Poller stopped")


async def main():
    scraper = ScraplingHttpScraper(poll_interval=5)
    # Start polling for a mock match
    try:
        await scraper.start_polling("match_123", "https://quotes.toscrape.com/") # dummy link
    except KeyboardInterrupt:
        await scraper.stop()

if __name__ == "__main__":
    asyncio.run(main())
