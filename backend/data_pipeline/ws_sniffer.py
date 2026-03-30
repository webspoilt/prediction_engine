"""
TASK 1: ZERO-COST WEBSOCKET DATA PIPELINE
IPL Win Probability Prediction Engine - Data Ingestion Module

This module provides redundant data ingestion using Playwright to sniff WebSocket
traffic from live score sites, with automatic failover to Poco X3 backup.

Hardware: Asus TUF (i5 10th Gen, 16GB RAM, GTX 1650 Ti)
Failover: Poco X3 (Termux/Linux)
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

# Playwright for WebSocket interception
from playwright.async_api import async_playwright, Page, WebSocket

# Redis for state management
import redis
from redis.asyncio import Redis as AsyncRedis

# For failover HTTP communication
import aiohttp
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BallData:
    """Standardized ball-by-ball data structure"""
    match_id: str
    inning: int
    over: float
    batsman: str
    bowler: str
    runs: int
    extras: int
    wicket: bool
    wicket_type: Optional[str]
    timestamp: float
    batting_team: str
    bowling_team: str
    total_runs: int
    total_wickets: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BallData':
        return cls(**data)


@dataclass
class MatchState:
    """Current match state for prediction"""
    match_id: str
    inning: int
    over: float
    total_runs: int
    total_wickets: int
    crr: float  # Current Run Rate
    rrr: float  # Required Run Rate
    target: Optional[int]
    powerplay: bool
    batting_team: str
    bowling_team: str
    last_18_balls: List[BallData]
    timestamp: float


class WebSocketSniffer:
    """
    Automated Pure WebSocket Client for live cricket scores.
    Phase 1: Uses Playwright temporarily to discover the WSS URL and auth tokens.
    Phase 2: Closes browser and connects directly via pure Python `websockets` 
             for zero-overhead, high-speed ingestion.
    """
    
    # Known WebSocket URL patterns for cricket sites
    WS_PATTERNS = {
        'cricbuzz': ['wss://*.cricbuzz.com/*', 'wss://*.cricbuzz.io/*'],
        'betfair': ['wss://*.betfair.com/*', 'wss://*.betfair.es/*'],
        'espn': ['wss://*.espncricinfo.com/*'],
        'hotstar': ['wss://*.hotstar.com/*'],
    }
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.async_redis = None
        self.ws_handlers: Dict[str, Callable] = {}
        self.message_buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        self.is_running = False
        
        # Connection specifics discovered during Phase 1
        self.discovered_ws_url = None
        self.discovered_headers = None
        
        # Register parsers
        self._register_parsers()
        
    def _register_parsers(self):
        """Register site-specific parsers"""
        self.ws_handlers['cricbuzz'] = self._parse_cricbuzz
        self.ws_handlers['betfair'] = self._parse_betfair
        self.ws_handlers['generic'] = self._parse_generic
        
    async def initialize(self):
        """Initialize async Redis connection"""
        self.async_redis = AsyncRedis(host='localhost', port=6379, decode_responses=True)
        
    async def _discover_ws_connection(self, match_url: str):
        """
        Phase 1: Spin up headless browser JUST to intercept the WSS handshake URL
        and required headers.
        """
        import websockets # Ensure we have the library
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        discovery_complete = asyncio.Event()

        def handle_websocket(ws: WebSocket):
            logger.info(f"Discovered Target WebSocket: {ws.url}")
            self.discovered_ws_url = ws.url
            # Some sites pass auth tokens in the URL itself, or cookies. 
            self.discovered_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Origin": "/".join(match_url.split('/')[:3]) # e.g. https://www.cricbuzz.com
            }
            discovery_complete.set()
            
        page.on("websocket", handle_websocket)
        logger.info(f"Phase 1: Navigating to {match_url} to sniff WSS URL...")
        
        await page.goto(match_url)
        
        try:
            # Wait up to 15 seconds for a WS connection to trigger
            await asyncio.wait_for(discovery_complete.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("No WebSocket connection discovered within 15 seconds.")
            
        # Immediately shut down browser to free RAM!
        await page.close()
        await browser.close()
        await playwright.stop()
        logger.info("Phase 1 Complete: Browser shut down. RAM freed.")


    async def _pure_websocket_stream(self, source: str):
        """
        Phase 2: Connect directly using pure websockets library and stream indefinitely.
        """
        import websockets
        
        retry_delay = 1
        while self.is_running:
            try:
                logger.info(f"Phase 2: Connecting pure Python WebSocket to {self.discovered_ws_url}")
                async with websockets.connect(
                    self.discovered_ws_url, 
                    extra_headers=self.discovered_headers,
                    ping_interval=20, 
                    ping_timeout=20
                ) as ws:
                    logger.info("Pure WebSocket connected successfully! Enjoy zero-overhead speeds.")
                    retry_delay = 1 # reset delay on successful connect
                    
                    async for message in ws:
                        if not self.is_running:
                            break
                        await self._process_ws_message(source, message)
                        
            except websockets.ConnectionClosed as e:
                logger.warning(f"Pure WebSocket closed: {e}. Reconnecting in {retry_delay}s...")
            except Exception as e:
                logger.error(f"Pure WebSocket error: {e}. Reconnecting in {retry_delay}s...")

            if not self.is_running:
                break
                
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60) # exponential backoff

    async def start_sniffing(self, match_url: str, source: str = 'cricbuzz'):
        """
        Orchestrates Phase 1 (Discovery) and Phase 2 (Pure Stream)
        """
        self.is_running = True
        
        # Phase 1: Discover
        await self._discover_ws_connection(match_url)
        
        if not self.discovered_ws_url:
            logger.error("Failed to automatically discover WSS URL. Cannot start pure WebSocket stream.")
            self.is_running = False
            return
            
        # Detect true source if necessary
        actual_source = self._detect_source(self.discovered_ws_url)
        if actual_source != 'generic':
            source = actual_source

        # Phase 2: Stream using pure websockets
        await self._pure_websocket_stream(source)
        
    def _detect_source(self, url: str) -> str:
        url_lower = url.lower()
        for source, patterns in self.WS_PATTERNS.items():
            for pattern in patterns:
                if source in url_lower:
                    return source
        return 'generic'
    
    async def _process_ws_message(self, source: str, data: str):
        try:
            message = self._decode_message(data)
            parser = self.ws_handlers.get(source, self._parse_generic)
            ball_data = parser(message)
            
            if ball_data:
                with self.buffer_lock:
                    self.message_buffer.append(ball_data.to_dict())
                await self._publish_to_redis(ball_data)
                
        except Exception as e:
            logger.error(f"Error processing pure WS message: {e}")
            
    def _decode_message(self, data: str) -> Dict:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {'raw': data}
    
    def _parse_cricbuzz(self, message: Dict) -> Optional[BallData]:
        try:
            if 'score' not in message: return None
            score = message.get('score', {})
            return BallData(
                match_id=message.get('match_id', 'unknown'),
                inning=score.get('inning', 1),
                over=float(score.get('over', 0)),
                batsman=score.get('batsman', ''),
                bowler=score.get('bowler', ''),
                runs=score.get('runs', 0),
                extras=score.get('extras', 0),
                wicket=score.get('wicket', False),
                wicket_type=score.get('wicket_type'),
                timestamp=time.time(),
                batting_team=score.get('batting_team', ''),
                bowling_team=score.get('bowling_team', ''),
                total_runs=score.get('total_runs', 0),
                total_wickets=score.get('total_wickets', 0)
            )
        except Exception as e:
            return None
    
    def _parse_betfair(self, message: Dict) -> Optional[BallData]:
        return None
    
    def _parse_generic(self, message: Dict) -> Optional[BallData]:
        return None
    
    async def _publish_to_redis(self, ball_data: BallData):
        stream_key = f"ipl:balls:{ball_data.match_id}"
        await self.async_redis.xadd(stream_key, ball_data.to_dict(), maxlen=1000)
        await self.async_redis.publish(f"ipl:live:{ball_data.match_id}", json.dumps(ball_data.to_dict()))
        
    async def stop(self):
        self.is_running = False
        if self.async_redis:
            await self.async_redis.close()
        logger.info("Pure WebSocket listener stopped.")


class FailoverManager:
    """
    Manages failover to Poco X3 (Termux) when primary scraper is blocked.
    Implements heartbeat monitoring and automatic switching.
    """
    
    def __init__(self, 
                 poco_ip: str = '192.168.1.100',
                 poco_port: int = 8080,
                 heartbeat_interval: int = 10):
        self.poco_ip = poco_ip
        self.poco_port = poco_port
        self.poco_url = f"http://{poco_ip}:{poco_port}"
        self.heartbeat_interval = heartbeat_interval
        
        self.redis_client = redis.Redis(decode_responses=True)
        self.is_primary_active = True
        self.last_heartbeat = time.time()
        self.failover_triggered = False
        
        # Circuit breaker state
        self.failure_count = 0
        self.failure_threshold = 3
        self.circuit_open = False
        
    async def start_heartbeat_monitor(self):
        """Start continuous heartbeat monitoring"""
        while True:
            await self._check_health()
            await asyncio.sleep(self.heartbeat_interval)
            
    async def _check_health(self):
        """Check health of primary and backup systems"""
        # Check if primary is producing data
        primary_health = await self._check_primary_health()
        
        # Check Poco X3 availability
        backup_health = await self._check_backup_health()
        
        # Update Redis with health status
        health_status = {
            'primary_active': primary_health,
            'backup_active': backup_health,
            'timestamp': time.time(),
            'failover_active': self.failover_triggered
        }
        
        self.redis_client.hset('system:health', mapping=health_status)
        
        # Failover logic
        if not primary_health and backup_health and not self.failover_triggered:
            await self._trigger_failover()
        elif primary_health and self.failover_triggered:
            await self._revert_to_primary()
            
    async def _check_primary_health(self) -> bool:
        """Check if primary scraper is healthy"""
        try:
            # Check last ball timestamp in Redis
            last_ball = self.redis_client.xrevrange('ipl:balls:*', count=1)
            
            if not last_ball:
                self.failure_count += 1
            else:
                # Check if data is recent (within 2 minutes)
                last_timestamp = float(last_ball[0][1].get('timestamp', 0))
                if time.time() - last_timestamp > 120:
                    self.failure_count += 1
                else:
                    self.failure_count = max(0, self.failure_count - 1)
            
            # Circuit breaker logic
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                return False
                
            return not self.circuit_open
            
        except Exception as e:
            logger.error(f"Primary health check error: {e}")
            self.failure_count += 1
            return False
            
    async def _check_backup_health(self) -> bool:
        """Check if Poco X3 backup is available"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.poco_url}/health") as resp:
                    return resp.status == 200
        except Exception as e:
            logger.warning(f"Backup health check failed: {e}")
            return False
            
    async def _trigger_failover(self):
        """Trigger failover to Poco X3"""
        logger.warning("!!! FAILOVER TRIGGERED - Switching to Poco X3 !!!")
        self.failover_triggered = True
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.poco_url}/activate", json={
                    'mode': 'primary',
                    'target_match': self._get_active_matches()
                })
                
            # Update Redis
            self.redis_client.set('system:mode', 'failover')
            
        except Exception as e:
            logger.error(f"Failover activation failed: {e}")
            
    async def _revert_to_primary(self):
        """Revert back to primary system"""
        logger.info("Reverting to primary system")
        self.failover_triggered = False
        self.circuit_open = False
        self.failure_count = 0
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.poco_url}/standby")
                
            self.redis_client.set('system:mode', 'primary')
            
        except Exception as e:
            logger.error(f"Revert to primary failed: {e}")
            
    def _get_active_matches(self) -> List[str]:
        """Get list of active matches from Redis"""
        keys = self.redis_client.keys('ipl:balls:*')
        return [k.split(':')[-1] for k in keys]


class DataPipeline:
    """
    Main data pipeline orchestrator.
    Combines WebSocket sniffing, failover management, and data normalization.
    """
    
    def __init__(self):
        self.sniffer = WebSocketSniffer()
        self.failover = FailoverManager()
        self.redis_client = redis.Redis(decode_responses=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize all components"""
        await self.sniffer.initialize()
        logger.info("Data pipeline initialized")
        
    async def run(self, match_urls: List[str]):
        """Run the complete data pipeline"""
        await self.initialize()
        
        # Start failover monitoring
        failover_task = asyncio.create_task(
            self.failover.start_heartbeat_monitor()
        )
        
        # Start WebSocket sniffing for each match
        sniff_tasks = [
            self.sniffer.start_sniffing(url) 
            for url in match_urls
        ]
        
        # Start feature extraction worker
        feature_task = asyncio.create_task(
            self._feature_extraction_worker()
        )
        
        # Run all tasks
        await asyncio.gather(
            failover_task,
            *sniff_tasks,
            feature_task,
            return_exceptions=True
        )
        
    async def _feature_extraction_worker(self):
        """Background worker to extract features from raw ball data"""
        while True:
            try:
                # Read from Redis Stream
                streams = self.redis_client.keys('ipl:balls:*')
                
                for stream in streams:
                    # Get new entries
                    entries = self.redis_client.xrange(stream)
                    
                    for entry_id, data in entries:
                        # Convert to BallData
                        ball = BallData.from_dict(data)
                        
                        # Extract features
                        features = self._extract_features(ball)
                        
                        # Store features
                        feature_key = f"ipl:features:{ball.match_id}"
                        self.redis_client.xadd(feature_key, features, maxlen=500)
                        
                await asyncio.sleep(2)  # 2-5 minute intervals as specified
                
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                await asyncio.sleep(5)
                
    def _extract_features(self, ball: BallData) -> Dict:
        """Extract ML features from ball data"""
        # Get recent balls for context
        recent_balls = self._get_recent_balls(ball.match_id, 18)
        
        # Calculate derived features
        features = {
            'match_id': ball.match_id,
            'inning': ball.inning,
            'over': ball.over,
            'total_runs': ball.total_runs,
            'total_wickets': ball.total_wickets,
            'crr': self._calculate_crr(recent_balls),
            'rrr': self._calculate_rrr(ball),
            'runs_last_6': sum(b.runs for b in recent_balls[-6:]),
            'wickets_last_6': sum(1 for b in recent_balls[-6:] if b.wicket),
            'boundary_rate': self._calculate_boundary_rate(recent_balls),
            'dot_ball_pressure': self._calculate_dot_pressure(recent_balls),
            'timestamp': ball.timestamp
        }
        
        return features
    
    def _get_recent_balls(self, match_id: str, n: int) -> List[BallData]:
        """Get last n balls for a match"""
        stream = f"ipl:balls:{match_id}"
        entries = self.redis_client.xrevrange(stream, count=n)
        return [BallData.from_dict(e[1]) for e in reversed(entries)]
    
    def _calculate_crr(self, balls: List[BallData]) -> float:
        """Calculate current run rate"""
        if not balls:
            return 0.0
        total_runs = sum(b.runs for b in balls)
        overs = len(balls) / 6
        return total_runs / max(overs, 0.1)
    
    def _calculate_rrr(self, ball: BallData) -> float:
        """Calculate required run rate"""
        # Implementation depends on match context
        # Placeholder for second innings calculation
        if ball.inning == 2:
            # Calculate based on target
            pass
        return 0.0
    
    def _calculate_boundary_rate(self, balls: List[BallData]) -> float:
        """Calculate boundary hitting rate"""
        if not balls:
            return 0.0
        boundaries = sum(1 for b in balls if b.runs >= 4)
        return boundaries / len(balls)
    
    def _calculate_dot_pressure(self, balls: List[BallData]) -> float:
        """Calculate dot ball pressure"""
        if not balls:
            return 0.0
        dot_balls = sum(1 for b in balls if b.runs == 0 and not b.wicket)
        return dot_balls / len(balls)


# ==================== USAGE EXAMPLE ====================

async def main():
    """Example usage of the data pipeline"""
    pipeline = DataPipeline()
    
    # List of live match URLs to monitor
    match_urls = [
        "https://www.cricbuzz.com/live-cricket-scores/12345/match-name",
        # Add more matches as needed
    ]
    
    try:
        await pipeline.run(match_urls)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await pipeline.sniffer.stop()


if __name__ == "__main__":
    asyncio.run(main())
