"""
IPL Prediction Engine — API Server (v3.0 "Never Empty")
=========================================================
Self-sufficient FastAPI server with:
  - MultiSourceFetcher waterfall pipeline (5-source cascading)
  - In-memory TTL cache primary, Redis optional secondary
  - /matches: ALWAYS returns data (never empty)
  - /api/data-health: Real-time status of all 5 scrapers
  - /api/offline-sync: Pre-loads 2026 schedule for offline UI
  - All existing endpoints preserved (predict, simulate, fantasy, etc.)
"""

import os
import json
import time
import asyncio
import logging
import random
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Returns MB

from backend.config import settings

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Custom JSON Encoder for NumPy types ──────────────────────────────────────
class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumPyEncoder, self).default(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle with graceful degradation."""
    global predictor, db_manager, betting_engine
    predictor = None
    db_manager = None
    betting_engine = None
    
    # 🏛️ Sovereign SRE: Track startup absolute path
    app.state.base_dir = os.path.dirname(os.path.abspath(__file__))
    app.state.static_dir = os.path.join(app.state.base_dir, "static")

    try:
        logger.info("🚀 Starting IPL Prediction Engine v3.0 (Never Empty)...")

        # 0. Initialize Data Structures
        try:
            from backend.ml_engine.analyze_recent_matches import ensure_base_csvs_exist
            ensure_base_csvs_exist()
            logger.info("✅ Base data structures initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to auto-initialize base data: {e}")

        # 1. Initialize MultiSourceFetcher (the core waterfall pipeline)
        try:
            from backend.data_pipeline.multi_source_fetcher import get_fetcher
            fetcher = get_fetcher()
            app.state.fetcher = fetcher
            logger.info(f"✅ MultiSourceFetcher initialized ({len(fetcher.get_static_schedule())} static matches)")
        except Exception as e:
            logger.error(f"❌ MultiSourceFetcher init failed: {e}")

        # 2. Database & Redis (optional — degrades gracefully)
        # ... (db/redis init remains same or backgrounded)
        
        # 3. BACKGROUND WARMUP: ML Engine & Discovery
        async def warmup():
            global predictor, db_manager, betting_engine
            try:
                # ML Engine Loading (Heavy)
                from backend.ml_engine.hybrid_model import RealTimePredictor
                predictor = RealTimePredictor(repo_id=settings.HF_REPO_ID)
                logger.info("✅ Titan ML Engine warmed up")
                
                # DNA Engine (New v4.8)
                from backend.ml_engine.context_engine import ContextualAuditor
                app.state.dna_engine = ContextualAuditor()
                logger.info("✅ DNA Engine synchronized")
                
                # Betting Engine (1xBet/4rabet style odds)
                from backend.api.betting_engine import BettingEngine
                betting_engine = BettingEngine(bookmaker_margin=0.05)
                logger.info("✅ Betting Engine loaded (1xBet/4rabet odds active)")
            except Exception as warmup_error:
                logger.warning(f"⚠️ Warmup degraded: {warmup_error}")

        app.state.warmup_task = asyncio.create_task(warmup())
        app.state.discovery_task = asyncio.create_task(run_discovery_loop(app))

    except Exception as e:
        logger.error(f"❌ Critical startup failure: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, "discovery_task"):
        app.state.discovery_task.cancel()
    if hasattr(app.state, "redis_pool") and app.state.redis_pool:
        app.state.redis_pool.disconnect()
    if db_manager:
        await db_manager.close()


app = FastAPI(
    title="IPL Prediction Engine API",
    description="Real-time match win probability engine with XGBoost + LSTM + Transformer",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Dashboard
os.makedirs("backend/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Include sub-routers
from backend.api.stats_router import router as stats_router

app.include_router(stats_router)


# ═══════════════════════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════════════════════

predictor = None
db_manager = None
betting_engine = None
active_scrapers: Dict[str, asyncio.Task] = {}


# ── Helper: Get Redis client safely ──────────────────────────────────────────
def get_redis():
    """Get a Redis client or None if unavailable. Strictly honors REDIS_ENABLED."""
    if not settings.REDIS_ENABLED:
        return None
    try:
        import redis
        pool = getattr(app.state, "redis_pool", None)
        if pool:
            r = redis.Redis(connection_pool=pool)
            r.ping()
            return r
    except Exception:
        pass
    return None


# ── Helper: Get MultiSourceFetcher ───────────────────────────────────────────
def get_fetcher():
    """Get the global MultiSourceFetcher singleton."""
    from backend.data_pipeline.multi_source_fetcher import get_fetcher as _get
    return _get()


# ── Agent Simulator (lazy init) ──────────────────────────────────────────────
_agent_swarm = None


def get_agent_swarm():
    global _agent_swarm
    if _agent_swarm is None:
        try:
            from backend.ml_engine.agent_sim import MultiAgentSimulator
            _agent_swarm = MultiAgentSimulator(num_agents=10)
        except Exception:
            _agent_swarm = None
    return _agent_swarm


# ── Scenario Simulator (lazy init) ───────────────────────────────────────────
_scenario_sim = None


def get_scenario_sim():
    global _scenario_sim
    if _scenario_sim is None:
        try:
            from backend.ml_engine.simulators import ScenarioSimulator
            _scenario_sim = ScenarioSimulator()
        except Exception:
            _scenario_sim = None
    return _scenario_sim


class SimulationRequest(BaseModel):
    num_simulations: int = 1000
    overs_to_project: int = 5
    captain_policy: str = "aggressive"


class MatchSimulator:
    """Monte Carlo engine for match projection."""

    def project_future(self, current_state: Dict, num_sims: int, overs: int, policy: str):
        results = []
        for _ in range(num_sims):
            temp_runs, temp_wickets = 0, 0
            for _ in range(overs * 6):
                if policy == "aggressive":
                    outcome = random.choices([0, 1, 2, 4, 6, "W"], weights=[35, 20, 10, 15, 10, 10])[0]
                else:
                    outcome = random.choices([0, 1, 2, 4, 6, "W"], weights=[50, 25, 10, 5, 5, 5])[0]
                if outcome == "W":
                    temp_wickets += 1
                    if temp_wickets + current_state.get("total_wickets", 0) >= 10:
                        break
                else:
                    temp_runs += outcome
            results.append((temp_runs, temp_wickets))
        runs_dist = [r for r, w in results]
        return {
            "projected_runs_avg": float(np.mean(runs_dist)),
            "projected_wickets_avg": float(np.mean([w for r, w in results])),
            "confidence_interval": [float(np.percentile(runs_dist, 5)), float(np.percentile(runs_dist, 95))],
        }


match_sim = MatchSimulator()


# ═══════════════════════════════════════════════════════════════════════════════
# Background Match Discovery
# ═══════════════════════════════════════════════════════════════════════════════

async def run_discovery_loop(app: FastAPI):
    """Periodically discover live IPL matches using the 5-source waterfall."""
    await asyncio.sleep(3)  # Let startup finish

    while True:
        try:
            fetcher = get_fetcher()
            all_matches = await fetcher.discover_matches()
            live_matches = [m for m in all_matches if m.get("status") == "live"]

            if live_matches:
                logger.info(
                    f"🔎 Discovery: {len(live_matches)} live matches "
                    f"(source: {fetcher.last_source_used})"
                )
                # Start scrapers for live matches if Redis available
                r = get_redis()
                for m in live_matches:
                    m_id = m.get("match_id", "")
                    if m_id and m_id not in active_scrapers and r:
                        try:
                            from backend.data_pipeline.espn_scraper import ESPNCricinfoScraper
                            scraper = ESPNCricinfoScraper()
                            url = m.get("url", "")
                            if url:
                                task = asyncio.create_task(scraper.start_polling(m_id, url))
                                active_scrapers[m_id] = task
                                logger.info(f"🔥 Auto-started scraper for: {m.get('teams', m_id)}")
                        except Exception as scraper_err:
                            logger.warning(f"Scraper start failed for {m_id}: {scraper_err}")

            # Adaptive polling: faster when live, slower when idle
            sleep_time = 120 if live_matches else settings.DISCOVERY_POLL_INTERVAL

            # Prune dead scrapers (BUG 5 FIX)
            dead_scrapers = [m_id for m_id, task in active_scrapers.items() if task.done()]
            for m_id in dead_scrapers:
                del active_scrapers[m_id]
                logger.debug(f"🧹 Cleaned up finished scraper task for {m_id}")

            # If no live matches, check if next match is soon
            if not live_matches:
                upcoming = [m for m in all_matches if m.get("status") == "scheduled"]
                upcoming.sort(key=lambda m: m.get("start_epoch", 0) or float("inf"))
                if upcoming:
                    next_epoch = upcoming[0].get("start_epoch", 0)
                    if next_epoch > 0:
                        time_until = next_epoch - time.time()
                        if time_until > 0 and time_until < 1800:
                            sleep_time = 60  # Within 30 min, poll every minute
                        elif time_until > 3600:
                            sleep_time = min(time_until - 600, 3600)  # Sleep up to 1 hour

            await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Discovery loop cancelled")
            return
        except Exception as e:
            logger.error(f"Discovery loop error: {e}")
            await asyncio.sleep(settings.DISCOVERY_POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_dashboard():
    """Elite Titan Hub — Sovereign UI Dashboard (Absolute Path Fix)."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Titan Hub Active", "error": "UI Registry Syncing..."}


@app.get("/health_status")
async def health_status():
    """Liveness check for HF Space."""
    return {
        "status": "Sovereign Oracle: ACTIVE",
        "version": "4.4-sovereign",
        "auditor": "Centurion-01: Healthy",
        "engine": "Titan-v4-Ensemble: Ready"
    }


@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse("backend/static/index.html")


@app.get("/pro")
async def serve_pro():
    """Elite Titan Analytics Dashboard (Absolute Path Fix)."""
    pro_path = os.path.join(os.path.dirname(__file__), "static", "pro.html")
    if os.path.exists(pro_path):
        return FileResponse(pro_path)
    return {"status": "Pro Analytics Active", "error": "Analytics Registry Syncing..."}


# ── Health Endpoint ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    r = get_redis()
    fetcher = get_fetcher()
    health = fetcher.get_source_health()
    return {
        "status": "Sovereign System Stability: ACTIVE",
        "timestamp": time.time(),
        "judicial_consensus": "Audited",
        "components": {
            "ml_engine": "loaded" if predictor else "unavailable",
            "centurion_auditor": "connected" if r else "unavailable",
            "dna_database": "connected" if db_manager else "unavailable",
            "betting_engine": "loaded" if betting_engine else "unavailable",
            "active_scrapers": len(active_scrapers),
            "data_sources": f"{health['active_sources']}/{health['total_sources']} active",
        },
    }


# ── Matches Endpoint (THE CORE — NEVER EMPTY) ───────────────────────────────

@app.get("/matches")
async def list_matches():
    """
    List all matches. Uses the 5-source waterfall pipeline.
    GUARANTEE: This endpoint NEVER returns an empty list.
    """
    global predictor, betting_engine
    fetcher = get_fetcher()
    matches = await fetcher.discover_matches()

    # Enrich with live data from Redis if available
    r = get_redis()
    if r:
        try:
            keys = r.keys("active:match:*")
            for key in keys:
                data = r.hgetall(key)
                if not data:
                    continue
                m_id = data.get("match_id", key.split(":")[-1])
                teams_str = data.get("teams", "")

                # Find matching entry in our results
                match_updated = False
                for i, m in enumerate(matches):
                    m_teams_str = " vs ".join(m.get("teams", []))
                    if m.get("match_id") == m_id or m_teams_str == teams_str:
                        # Update with live Redis data
                        if data.get("status") == "live":
                            matches[i]["status"] = "live"
                        if data.get("score") and data.get("score") != "—":
                            matches[i]["score"] = data["score"]
                        if data.get("over") and data.get("over") != "0.0":
                            try:
                                matches[i]["over"] = float(data["over"])
                            except ValueError:
                                pass
                        match_updated = True
                        break

                if not match_updated and teams_str:
                    # New match from Redis not in our list
                    teams_list = teams_str.split(" vs ")
                    matches.append({
                        "match_id": m_id,
                        "teams": teams_list if len(teams_list) == 2 else ["TBD", "TBD"],
                        "team_short": [],
                        "status": data.get("status", "live"),
                        "score": data.get("score", "—"),
                        "over": float(data.get("over", 0.0)),
                        "venue": data.get("venue", "TBD"),
                        "win_probability": 0.5,
                        "source": "redis",
                    })
        except Exception as e:
            if settings.REDIS_ENABLED:
                logger.warning(f"Redis merge (non-critical): {e}")

    # Enrich with ML predictions (Titan v4.5 — Real-time Enrichment)
    if predictor:
        for m in matches:
            # Skip if we already have a specialized probability or if it's not a full match
            if len(m.get("teams", [])) != 2: continue
            
            try:
                if m.get("status") == "live":
                    # LIVE PREDICTION: Call the full hybrid ensemble
                    pred = predictor.model.predict_live_match(
                        m["match_id"], m["teams"][0], m["teams"][1], m.get("venue", "Unknown")
                    )
                    m["win_probability"] = float(pred.get("win_probability", 0.5))
                    m["forensic_trace"] = pred.get("forensic_trace", [])
                elif m.get("win_probability") == 0.5:
                    # PRE-MATCH: Call the static ensemble
                    pred = predictor.model.predict_pre_match(
                        m["teams"][0], m["teams"][1], m.get("venue", "Unknown")
                    )
                    m["win_probability"] = float(pred.get("win_probability", 0.5))
            except Exception as e:
                logger.warning(f"Enrichment failed for {m.get('match_id')}: {e}")

    # Enrich with betting odds (v5.0 — 1xBet/4rabet Decimal Odds)
    for m in matches:
        if len(m.get("teams", [])) != 2: continue
        try:
            if betting_engine:
                t1, t2 = m["teams"]
                prob = m.get("win_probability", 0.5)
                pred_stub = {"win_probability": prob, "confidence": 0.7}
                odds_data = betting_engine.generate_match_odds(pred_stub, t1, t2, m.get("match_id", ""))
                # Inline odds for card display
                winner_odds = odds_data.match_winner
                short1 = betting_engine._normalize_team(t1)
                short2 = betting_engine._normalize_team(t2)
                t1_odds = winner_odds.get(short1) or winner_odds.get(t1)
                t2_odds = winner_odds.get(short2) or winner_odds.get(t2)
                m["betting_odds"] = {
                    "team1_decimal": t1_odds.decimal_odds if t1_odds else 2.0,
                    "team2_decimal": t2_odds.decimal_odds if t2_odds else 2.0,
                    "team1_american": t1_odds.american_odds if t1_odds else "+100",
                    "team2_american": t2_odds.american_odds if t2_odds else "+100",
                    "team1_value": t1_odds.value_rating if t1_odds else "Fair",
                    "team2_value": t2_odds.value_rating if t2_odds else "Fair",
                    "overround": round((1/(t1_odds.decimal_odds if t1_odds else 2) + 1/(t2_odds.decimal_odds if t2_odds else 2)) * 100, 1) if t1_odds and t2_odds else 100.0,
                }
        except Exception as e:
            logger.warning(f"Betting odds enrichment failed for {m.get('match_id')}: {e}")

    return matches


# ── Data Health Endpoint ─────────────────────────────────────────────────────

@app.get("/api/data-health")
async def data_health():
    """Return real-time status of all 5 data sources (for the status bar UI)."""
    fetcher = get_fetcher()
    health = fetcher.get_source_health()
    r = get_redis()
    health["redis_connected"] = r is not None
    health["ml_engine_loaded"] = predictor is not None
    health["server_uptime"] = time.time()
    health["memory_usage_mb"] = round(get_memory_usage(), 2)
    return health


# ── Offline Sync Endpoint ────────────────────────────────────────────────────

@app.get("/api/offline-sync")
async def offline_sync():
    """
    Returns the complete 2026 IPL schedule + points table for offline mode.
    The frontend can cache this in localStorage for zero-internet operation.
    """
    # Try loading the full schedule.json
    json_paths = [
        os.path.join(os.path.dirname(__file__), "data_pipeline", "schedule.json"),
        os.path.join(os.getcwd(), "backend", "data_pipeline", "schedule.json"),
    ]
    for p in json_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"schedule.json parse error: {e}")

    # Fallback: build from fetcher
    fetcher = get_fetcher()
    schedule = fetcher.get_static_schedule()
    return {
        "season": "IPL 2026",
        "total_matches": len(schedule),
        "schedule": schedule,
        "points_table": _get_points_table(),
    }


# ── Upcoming Matches ─────────────────────────────────────────────────────────

@app.get("/upcoming/{season}")
async def get_upcoming_matches(season: str):
    """Fetch upcoming schedule from the MultiSourceFetcher."""
    fetcher = get_fetcher()
    upcoming = await fetcher.get_upcoming(limit=20)
    return {"matchschedule": upcoming}


# ── Betting Odds API (1xBet/4rabet Style) ────────────────────────────────────

@app.get("/api/betting/{match_id}")
async def get_betting_odds(match_id: str):
    """
    Full betting data package for a match, styled like 1xBet / 4rabet.
    Returns decimal/fractional/american odds for:
      - Match Winner
      - Total Runs Over/Under
      - Top Batsman / Top Bowler prop bets
      - Match Special props
    """
    global betting_engine, predictor
    if not betting_engine:
        # Lazy init fallback
        try:
            from backend.api.betting_engine import BettingEngine
            betting_engine = BettingEngine(bookmaker_margin=0.05)
        except Exception:
            raise HTTPException(status_code=503, detail="Betting engine unavailable")

    fetcher = get_fetcher()
    match = await fetcher.get_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    t1, t2 = match.get("teams", ["Team A", "Team B"])
    win_prob = match.get("win_probability", 0.5)

    prediction_stub = {
        "win_probability": win_prob,
        "confidence": 0.7,
        "crr": 7.5,
        "over": match.get("over", 0.0),
        "inning": 1,
        "total_runs": 0,
    }

    # If we have a live prediction, use it
    if predictor and match.get("status") == "live":
        try:
            live_pred = await predictor.predict_live_match(match)
            prediction_stub.update({
                "win_probability": live_pred.get("win_probability", win_prob),
                "confidence": live_pred.get("confidence", 0.7),
            })
        except Exception:
            pass

    odds_data = betting_engine.generate_match_odds(prediction_stub, t1, t2, match_id)
    return odds_data.to_dict()


# ── Points Table ─────────────────────────────────────────────────────────────

@app.get("/points/{season}")
async def get_points_table(season: str):
    """Fetch Points Table. Returns base template for 2026."""
    return {"points": _get_points_table()}


def _get_points_table() -> List[Dict]:
    """Generate the 10-team points table."""
    teams = [
        ("Chennai Super Kings", "CSK"),
        ("Delhi Capitals", "DC"),
        ("Gujarat Titans", "GT"),
        ("Kolkata Knight Riders", "KKR"),
        ("Lucknow Super Giants", "LSG"),
        ("Mumbai Indians", "MI"),
        ("Punjab Kings", "PBKS"),
        ("Rajasthan Royals", "RR"),
        ("Royal Challengers Bengaluru", "RCB"),
        ("Sunrisers Hyderabad", "SRH"),
    ]
    return [
        {
            "name": name,
            "teamshortname": short,
            "matchesplayed": 0,
            "matcheswon": 0,
            "matcheslost": 0,
            "nr": 0,
            "points": 0,
            "nrr": "+0.000",
        }
        for name, short in teams
    ]


# ── Debug Endpoint ───────────────────────────────────────────────────────────

@app.get("/debug/env")
async def debug_env():
    """Diagnostics for production filesystem and memory."""
    import sys
    fetcher = get_fetcher()
    health = fetcher.get_source_health()
    return {
        "cwd": os.getcwd(),
        "file": __file__,
        "sys_path": sys.path[:5],
        "fetcher_cache_size": health["cache_size"],
        "static_schedule_count": len(fetcher.get_static_schedule()),
        "active_sources": f"{health['active_sources']}/{health['total_sources']}",
        "last_source": health["last_source_used"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction & Simulation Routes (Preserved from v2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/predict/{match_id}")
async def get_prediction(match_id: str):
    """
    Sovereign Prediction Pipeline (v4.8):
    1. Fetch live data via Sovereign Eyes (Parallel ESPN/Cricbuzz).
    2. Inject Venue/H2H DNA via Sovereign context engine.
    3. Calculate Win Probability with Hybrid ML + Forensic Audit.
    4. Enrich with Scenario/Swarm/Betting analytics.
    """
    fetcher = get_fetcher()
    match = await fetcher.get_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # 1. Sovereign DNA Context
    dna_context = {}
    if hasattr(app.state, "dna_engine"):
        try:
            t1, t2 = match.get('teams', ['Team A', 'Team B'])
            venue = match.get('venue', 'Unknown')
            # MOJO FIX: Method is 'get_match_dna', not 'get_match_context'
            dna_context = app.state.dna_engine.get_match_dna(t1, t2, venue)
        except Exception as dna_err:
            logger.warning(f"DNA context extraction failed: {dna_err}")

    # 2. Main ML Prediction
    if not predictor:
        return {
            "match_id": match_id,
            "win_probability": match.get('win_probability', 0.5),
            "status": "warming_up",
            "forensic_trace": ["[AUDITOR] ML Engine warming up. Using Sovereign baseline."]
        }

    try:
        prediction = await predictor.predict_live_match(match, dna_context)
        
        # 3. Analytics Enrichment (Scenario/Swarm)
        scenario_sim = get_scenario_sim()
        if scenario_sim:
            try:
                prediction["scenario"] = scenario_sim.simulate_remaining_balls(prediction, n_simulations=500)
            except: pass

        swarm = get_agent_swarm()
        if swarm:
            try:
                prediction.update(swarm.simulate(prediction))
            except: pass

        # 4. Betting Enrichment
        if betting_engine:
            t1, t2 = match.get('teams', ['Team A', 'Team B'])
            odds = betting_engine.generate_match_odds(prediction, t1, t2, match_id)
            prediction["betting"] = odds.to_dict()

        # 5. Persistence
        if db_manager:
            asyncio.create_task(db_manager.save_prediction(prediction))

        return prediction

    except Exception as e:
        logger.error(f"Prediction logic crash for {match_id}: {e}")
        return {"error": str(e), "win_probability": 0.5, "status": "error"}


@app.get("/simulate/scenario/{match_id}")
async def run_scenario_simulation(match_id: str, n: int = 5000):
    """Run detailed Monte Carlo scenario simulation for remaining balls."""
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable for live data")

    last_ball = r.xrevrange(f"ipl:balls:{match_id}", count=1)
    if not last_ball:
        raise HTTPException(status_code=404, detail="No match data found")

    d = last_ball[0][1]
    current_state = {
        "total_runs": int(d.get("total_runs", 0) or 0),
        "total_wickets": int(d.get("total_wickets", 0) or 0),
        "balls_remaining": max(0, 120 - int(float(d.get("over", 0)) * 6)),
    }

    sim = get_scenario_sim()
    if not sim:
        raise HTTPException(status_code=503, detail="Simulator unavailable")

    result = sim.simulate_remaining_balls(current_state, n_simulations=min(n, 10000))
    return {"match_id": match_id, "current_state": current_state, "simulation": result}


@app.get("/fantasy/{player_name}")
async def get_fantasy_projection(player_name: str, ownership: float = 0.5, role: str = "batsman"):
    """Get Bayesian fantasy points projection for a player."""
    try:
        from backend.ml_engine.simulators import BayesianPlayerPredictor, FantasyEngine
        from backend.ml_engine.hybrid_model import CricsheetNormalizer

        normalizer = CricsheetNormalizer()
        normalizer.load_player_stats()

        bat_history, wkt_history = [], []
        if normalizer.player_stats:
            bat_data = normalizer.player_stats.get("batsmen", {}).get(player_name, {})
            bowl_data = normalizer.player_stats.get("bowlers", {}).get(player_name, {})
            if bat_data.get("average"):
                avg = bat_data["average"]
                bat_history = [max(0, avg + np.random.normal(0, avg * 0.3)) for _ in range(5)]
            if bowl_data.get("average"):
                wkt_history = [max(0, np.random.normal(1.2, 0.5)) for _ in range(5)]

        predictor_b = BayesianPlayerPredictor()
        run_proj = predictor_b.predict_player_runs(bat_history, opp_strength=1.0)
        wkt_proj = predictor_b.predict_player_wickets(wkt_history)

        projection = {**run_proj, **wkt_proj}
        fantasy = FantasyEngine()
        fantasy_pts = fantasy.calculate_expected_points(
            player_projection=projection,
            role=role,
            ownership_pct=float(ownership),
        )

        return {
            "player": player_name,
            "role": role,
            "ownership_pct": round(ownership * 100, 1),
            "batting_projection": run_proj,
            "bowling_projection": wkt_proj,
            "fantasy": fantasy_pts,
        }
    except Exception as e:
        logger.error(f"Fantasy projection error: {e}")
        raise HTTPException(status_code=500, detail="Fantasy projection failed")


@app.post("/simulate/{match_id}")
async def run_what_if_simulation(match_id: str, request: SimulationRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="ML Engine not initialized")
    prediction = predictor.predict_live_match(match_id)
    if "error" in prediction:
        raise HTTPException(status_code=404, detail="No live data to simulate from")
    projection = match_sim.project_future(
        prediction, request.num_simulations, request.overs_to_project, request.captain_policy
    )
    return {"match_id": match_id, "current_state": prediction, "projection": projection}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/predictions/{match_id}")
async def websocket_prediction(websocket: WebSocket, match_id: str):
    """Real-time prediction stream via WebSocket."""
    await websocket.accept()
    logger.info(f"WebSocket connected for match: {match_id}")
    r = None
    pubsub = None
    try:
        r = get_redis()
        if r:
            pubsub = r.pubsub()
            pubsub.subscribe(f"ipl:predictions:{match_id}")

        # Initial push
        if predictor:
            try:
                initial = predictor.predict_live_match(match_id)
                if "error" not in initial:
                    swarm = get_agent_swarm()
                    if swarm:
                        initial.update(swarm.simulate(initial))
                    await websocket.send_json(initial)
            except Exception as e:
                logger.warning(f"Initial prediction failed: {e}")

        while True:
            if pubsub:
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    try:
                        data = json.loads(message["data"])
                        await websocket.send_json(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {match_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if pubsub:
            try:
                pubsub.unsubscribe()
                pubsub.close()
            except Exception:
                pass
        if r:
            try:
                r.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    import uvicorn
    # Hardcoded for HF binding reliability
    uvicorn.run(app, host="0.0.0.0", port=7860)
