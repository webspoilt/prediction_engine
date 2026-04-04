"""
Match Discovery Service — Zero-Dependency Edition
====================================================
Uses the MultiSourceFetcher waterfall pipeline instead of direct API calls.
Redis is OPTIONAL — works entirely with in-memory state when Redis is unavailable.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MatchDiscoveryService:
    """
    Background worker that discovers IPL matches using the 5-source waterfall.
    Schedules itself intelligently based on the IPL timetable.
    Redis is optional — degrades gracefully to in-memory-only operation.
    """

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.active_scrapers: Dict[str, asyncio.Task] = {}
        self.default_poll = 300  # 5 min fallback
        self.sync_interval = 60  # API sync every 60s
        self._fetcher = None

    def _get_fetcher(self):
        """Lazy-init the MultiSourceFetcher."""
        if self._fetcher is None:
            from backend.data_pipeline.multi_source_fetcher import get_fetcher
            self._fetcher = get_fetcher()
        return self._fetcher

    async def run(self):
        """Main discovery loop."""
        logger.info("🚀 Match Discovery Service starting (MultiSource Waterfall)...")

        # Start API sync worker
        asyncio.create_task(self._api_sync_worker())

        while True:
            try:
                fetcher = self._get_fetcher()

                # Discover matches through the 5-source waterfall
                all_matches = await fetcher.discover_matches()
                live_matches = [m for m in all_matches if m.get("status") == "live"]

                # If Redis is available, sync match data to it
                if self.redis_client:
                    self._sync_to_redis(all_matches)

                if live_matches:
                    logger.info(f"🔎 Discovery: {len(live_matches)} live, {len(all_matches)} total (source: {fetcher.last_source_used})")

                    # While live, poll more frequently
                    await asyncio.sleep(120)
                    continue

                # No live matches — calculate sleep until next match
                upcoming = [m for m in all_matches if m.get("status") == "scheduled"]
                upcoming.sort(key=lambda m: m.get("start_epoch", 0) or float("inf"))

                if upcoming:
                    next_match = upcoming[0]
                    start_epoch = next_match.get("start_epoch", 0)
                    if start_epoch > 0:
                        time_until = start_epoch - time.time()
                        if time_until > 600:  # More than 10 min away
                            sleep_for = min(time_until - 600, 3600)  # Wake 10 min before, max 1 hour sleep
                            teams = next_match.get("teams", ["?", "?"])
                            logger.info(
                                f"⏳ Next match: {teams[0]} vs {teams[1]} in {int(time_until / 60)} min. "
                                f"Sleeping {int(sleep_for)}s..."
                            )
                            await asyncio.sleep(sleep_for)
                            continue

                await asyncio.sleep(self.default_poll)

            except asyncio.CancelledError:
                logger.info("Discovery loop cancelled")
                return
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(self.default_poll)

    async def _api_sync_worker(self):
        """
        Background task: keeps Redis in sync with fresh data from the waterfall.
        If Redis is unavailable, does nothing (in-memory cache is the primary).
        """
        logger.info("🔄 API Sync Worker started")
        while True:
            try:
                if self.redis_client:
                    try:
                        self.redis_client.ping()
                    except Exception:
                        self.redis_client = None
                        logger.warning("⚠️ Redis lost. Sync worker running in memory-only mode.")

                if self.redis_client:
                    fetcher = self._get_fetcher()
                    live_matches = await fetcher.get_live_only()
                    for match in live_matches:
                        m_id = match.get("match_id", "")
                        if m_id:
                            m_key = f"active:match:{m_id}"
                            flat = {
                                "match_id": m_id,
                                "teams": " vs ".join(match.get("teams", [])),
                                "status": match.get("status", "live"),
                                "score": match.get("score", "—"),
                                "over": str(match.get("over", 0.0)),
                                "venue": match.get("venue", "TBD"),
                                "source": match.get("source", "unknown"),
                            }
                            self.redis_client.hset(m_key, mapping=flat)
                            self.redis_client.expire(m_key, 28800)

                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"API Sync Worker error: {e}")
                await asyncio.sleep(self.sync_interval)

    def _sync_to_redis(self, matches: List[Dict]):
        """Push discovered matches to Redis (if available)."""
        if not self.redis_client:
            return
        try:
            for match in matches[:20]:  # Limit to prevent Redis overload
                m_id = match.get("match_id", "")
                if not m_id:
                    continue
                m_key = f"active:match:{m_id}"
                flat = {
                    "match_id": m_id,
                    "teams": " vs ".join(match.get("teams", [])),
                    "status": match.get("status", "scheduled"),
                    "score": match.get("score", "—"),
                    "over": str(match.get("over", 0.0)),
                    "venue": match.get("venue", "TBD"),
                    "source": match.get("source", "static"),
                }
                self.redis_client.hset(m_key, mapping=flat)
                ttl = 86400 if match.get("status") == "scheduled" else 28800
                self.redis_client.expire(m_key, ttl)
                self.redis_client.sadd("active:matches:set", m_id)
        except Exception as e:
            logger.warning(f"Redis sync failed (non-critical): {e}")

    def _get_local_schedule(self) -> List[Dict]:
        """Legacy compat — returns static schedule from MultiSourceFetcher."""
        fetcher = self._get_fetcher()
        matches = fetcher.get_static_schedule()
        # Convert to the legacy format expected by api_server.py
        legacy = []
        for m in matches:
            epoch = m.get("start_epoch", 0)
            if epoch > time.time():
                legacy.append({
                    "match_id": m["match_id"],
                    "teams": " vs ".join(m.get("teams", [])),
                    "start_time": epoch,
                })
        return legacy


async def main():
    service = MatchDiscoveryService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
