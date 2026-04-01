import os
import json
import time
import asyncio
import logging
import redis
import random
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from backend.config import settings

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON Encoder for NumPy types (Pro-fix for ML serialization 500s)
class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumPyEncoder, self).default(obj)

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle with graceful degradation."""
    global predictor, db_manager, betting_engine
    predictor = None
    db_manager = None
    betting_engine = None

    try:
        logger.info("🚀 Starting IPL Prediction Engine...")

        # 1. Database (optional — degrades gracefully)
        try:
            from backend.infrastructure.db_manager import DatabaseManager
            db_manager = DatabaseManager()
            await db_manager.connect()
            await db_manager.initialize_schema()
            app.state.db = db_manager
            logger.info("✅ Database connected")
        except Exception as e:
            logger.warning(f"⚠️ Database unavailable (running without persistence): {e}")
            db_manager = None

        # 2. Redis (optional — degrades gracefully)
        if settings.REDIS_ENABLED:
            try:
                app.state.redis_pool = redis.ConnectionPool(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True
                )
                r = redis.Redis(connection_pool=app.state.redis_pool)
                r.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable (running in degraded mode): {e}")
                app.state.redis_pool = None
        else:
            app.state.redis_pool = None
            logger.info("ℹ️ Redis disabled by config")

        # 3. ML Engine
        try:
            from backend.ml_engine.hybrid_model import RealTimePredictor
            predictor = RealTimePredictor(repo_id=settings.HF_REPO_ID)
            logger.info(f"✅ ML Engine loaded from {settings.HF_REPO_ID}")
        except Exception as e:
            logger.warning(f"⚠️ ML Engine failed to load (predictions unavailable): {e}")
            predictor = None

        # 4. Betting Engine
        try:
            from backend.api.betting_engine import BettingEngine
            betting_engine = BettingEngine(bookmaker_margin=settings.BOOKMAKER_MARGIN)
            logger.info("✅ Betting Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Betting Engine failed: {e}")
            betting_engine = None

        # 5. Background Match Discovery (auto-detect live IPL matches)
        app.state.discovery_task = asyncio.create_task(
            run_discovery_loop(app), name="match_discovery"
        )

    except Exception as e:
        logger.error(f"❌ Critical startup failure: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, 'discovery_task'):
        app.state.discovery_task.cancel()
    if hasattr(app.state, 'redis_pool') and app.state.redis_pool:
        app.state.redis_pool.disconnect()
    if db_manager:
        await db_manager.close()


app = FastAPI(
    title="IPL Prediction Engine API",
    description="Real-time match win probability engine with XGBoost + LSTM + Transformer",
    version="2.0.0",
    lifespan=lifespan
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

# ─── Global State ─────────────────────────────────────────────────────────────
predictor = None
db_manager = None
betting_engine = None
active_scrapers: Dict[str, asyncio.Task] = {}

# ─── Agent Simulator (lazy init) ─────────────────────────────────────────────
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
                    outcome = random.choices([0,1,2,4,6,'W'], weights=[35,20,10,15,10,10])[0]
                else:
                    outcome = random.choices([0,1,2,4,6,'W'], weights=[50,25,10,5,5,5])[0]
                if outcome == 'W':
                    temp_wickets += 1
                    if temp_wickets + current_state.get('total_wickets', 0) >= 10:
                        break
                else:
                    temp_runs += outcome
            results.append((temp_runs, temp_wickets))
        runs_dist = [r for r, w in results]
        return {
            "projected_runs_avg": float(np.mean(runs_dist)),
            "projected_wickets_avg": float(np.mean([w for r, w in results])),
            "confidence_interval": [float(np.percentile(runs_dist, 5)), float(np.percentile(runs_dist, 95))]
        }

match_sim = MatchSimulator()


# ─── Helper: Get Redis client safely ─────────────────────────────────────────
def get_redis() -> Optional[redis.Redis]:
    """Get a Redis client or None if unavailable."""
    try:
        pool = getattr(app.state, 'redis_pool', None)
        if pool:
            r = redis.Redis(connection_pool=pool)
            r.ping()
            return r
    except Exception:
        pass
    return None


# ─── Background Match Discovery ──────────────────────────────────────────────
async def run_discovery_loop(app: FastAPI):
    """Periodically scan for live IPL matches and auto-start scrapers."""
    await asyncio.sleep(5)  # Let startup finish
    while True:
        try:
            from backend.data_pipeline.match_discovery import MatchDiscoveryService, IPTLiveFeed
            discovery = MatchDiscoveryService()
            matches = discovery._find_live_ipl_matches()
            logger.info(f"🔎 Discovery scan: {len(matches)} live IPL matches found")

            for m in matches:
                m_id = m['match_id']
                if m_id not in active_scrapers:
                    logger.info(f"🔥 Auto-starting scraper for: {m.get('teams', m_id)}")
                    try:
                        from backend.data_pipeline.espn_scraper import ESPNCricinfoScraper
                        scraper = ESPNCricinfoScraper()
                        task = asyncio.create_task(scraper.start_polling(m_id, m['url']))
                        active_scrapers[m_id] = task
                    except Exception as scraper_err:
                        logger.error(f"Failed to start scraper for {m_id}: {scraper_err}")

                    if db_manager:
                        try:
                            await db_manager.save_match({
                                'match_id': m_id,
                                'teams': m.get('teams', '').split(' vs ') if isinstance(m.get('teams'), str) else [],
                                'status': 'live',
                                'metadata': {'url': m['url']}
                            })
                        except Exception as db_err:
                            logger.error(f"DB save error: {db_err}")

        except Exception as e:
            logger.error(f"Discovery loop error: {e}")

        await asyncio.sleep(settings.DISCOVERY_POLL_INTERVAL)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse("backend/static/index.html")

@app.get("/pro")
async def serve_pro():
    return FileResponse("backend/static/index.html")  # Unified dashboard


@app.get("/health")
async def health_check():
    r = get_redis()
    return {
        "status": "healthy" if predictor else "degraded",
        "timestamp": time.time(),
        "components": {
            "ml_engine": "loaded" if predictor else "unavailable",
            "redis": "connected" if r else "unavailable",
            "database": "connected" if db_manager else "unavailable",
            "betting_engine": "loaded" if betting_engine else "unavailable",
            "active_scrapers": len(active_scrapers),
        }
    }


@app.get("/upcoming/{season}")
async def upcoming_matches(season: int):
    from backend.data_pipeline.match_discovery import IPTLiveFeed
    data = IPTLiveFeed.get_upcoming_matches(season)
    if not data:
        raise HTTPException(status_code=503, detail="Live feed unavailable")
    return data


@app.get("/points/{season}")
async def points_table(season: int):
    from backend.data_pipeline.match_discovery import IPTLiveFeed
    data = IPTLiveFeed.get_points_table(season)
    if not data:
        raise HTTPException(status_code=503, detail="Points feed unavailable")
    return data


@app.get("/matches")
async def list_matches():
    """List all active matches from Redis."""
    r = get_redis()
    if not r:
        return []
    try:
        keys = r.keys("ipl:balls:*")
        match_ids = [k.split(":")[-1] for k in keys]
        matches = []
        for mid in match_ids:
            try:
                last_ball = r.xrevrange(f"ipl:balls:{mid}", count=1)
                if last_ball:
                    data = last_ball[0][1]
                    # Explicit casting prevents 500 serialization errors from NumPy types
                    matches.append({
                        "match_id": str(mid),
                        "teams": [str(data.get('batting_team', 'N/A')), str(data.get('bowling_team', 'N/A'))],
                        "inning": int(data.get('inning', 1)) if data.get('inning') else 1,
                        "score": f"{data.get('runs', 0)}/{data.get('wicket', 0)}",
                        "over": float(data.get('over', 0.0)) if data.get('over') else 0.0
                    })
            except Exception as e:
                logger.warning(f"Error parsing match {mid} from Redis: {e}")
                continue
        return matches
    except Exception as e:
        logger.error(f"List matches error: {e}")
        return []


@app.get("/predict/{match_id}")
async def get_prediction(match_id: str):
    """Fetch latest win probability with betting odds enrichment."""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML Engine not initialized")
    try:
        prediction = predictor.predict_live_match(match_id)
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])

        # Enrich with agent swarm
        swarm = get_agent_swarm()
        if swarm:
            agent_features = swarm.simulate(prediction)
            prediction.update(agent_features)

        # Enrich with betting odds
        if betting_engine:
            r = get_redis()
            teams = ["Team A", "Team B"]
            if r:
                last_ball = r.xrevrange(f"ipl:balls:{match_id}", count=1)
                if last_ball:
                    d = last_ball[0][1]
                    teams = [d.get('batting_team', 'Team A'), d.get('bowling_team', 'Team B')]

            odds_data = betting_engine.generate_match_odds(prediction, teams[0], teams[1], match_id)
            prediction['betting'] = odds_data.to_dict()
            prediction['model_accuracy'] = odds_data.model_accuracy

        # Persist
        if db_manager:
            asyncio.create_task(db_manager.save_prediction(prediction))

        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {match_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")


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
                if 'error' not in initial:
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
                        data = json.loads(message['data'])
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
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)
