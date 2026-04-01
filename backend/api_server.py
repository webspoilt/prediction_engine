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

        # 0. Initialize Data Structures
        try:
            from backend.ml_engine.analyze_recent_matches import ensure_base_csvs_exist
            ensure_base_csvs_exist()
            logger.info("✅ Base data structures initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to auto-initialize base data: {e}")

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

# ─── Scenario Simulator (lazy init) ───────────────────────────────────────────
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
    """Periodically scan for live IPL matches and inject static schedule if empty."""
    await asyncio.sleep(5)  # Let startup finish
    while True:
        try:
            r = get_redis()
            if not r:
                await asyncio.sleep(60)
                continue

            from backend.data_pipeline.cricbuzz_api import CricbuzzAPI
            matches = await CricbuzzAPI.get_live_matches()
            
            # Injection: Always ensure next 5 matches from static schedule are in Redis
            try:
                from backend.data_pipeline.match_discovery import MatchDiscoveryService
                service = MatchDiscoveryService()
                upcoming = service._get_local_schedule()
                for match in upcoming[:5]:
                    m_key = f"active:match:{match['match_id']}"
                    if not r.exists(m_key):
                        match['status'] = 'scheduled'
                        r.hset(m_key, mapping=match)
                        r.expire(m_key, 86400) # 24 hours for scheduled
            except Exception as e:
                logger.warning(f"Static schedule injection failed: {e}")

            if matches:
                logger.info(f"🔎 Discovery scan: {len(matches)} live IPL matches found")

            for m in matches:
                m_id = m['match_id']
                m_key = f"active:match:{m_id}"
                
                # Check for live scraper
                if m_id not in active_scrapers:
                    logger.info(f"🔥 Auto-starting scraper for: {m.get('teams', m_id)}")
                    try:
                        from backend.data_pipeline.espn_scraper import ESPNCricinfoScraper
                        scraper = ESPNCricinfoScraper()
                        task = asyncio.create_task(scraper.start_polling(m_id, m['url']))
                        active_scrapers[m_id] = task
                        
                        # Mark as live in Redis
                        m['status'] = 'live'
                        r.hset(m_key, mapping=m)
                    except Exception as scraper_err:
                        logger.error(f"Failed to start scraper for {m_id}: {scraper_err}")

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




@app.get("/matches")
async def list_matches():
    """List all discovered matches (Live and Scheduled)."""
    r = get_redis()
    if not r:
        return []
    
    try:
        keys = r.keys("active:match:*")
        matches = []
        for key in keys:
            m_id = key.split(":")[-1]
            data = r.hgetall(key)
            status = data.get('status', 'scheduled')
            
            match_entry = {
                "match_id": str(m_id),
                "teams": data.get('teams', '').split(' vs ') if isinstance(data.get('teams'), str) else ["TBD", "TBD"],
                "status": status,
                "score": "0/0",
                "over": 0.0,
                "win_probability": 0.5
            }

            if status == 'live':
                try:
                    last_ball = r.xrevrange(f"ipl:balls:{m_id}", count=1)
                    if last_ball:
                        b_data = last_ball[0][1]
                        match_entry["score"] = f"{b_data.get('runs', 0)}/{b_data.get('wicket', 0)}"
                        match_entry["over"] = float(b_data.get('over', 0.0))
                except Exception:
                    pass
            
            # Add pre-match prediction if available for scheduled games
            if predictor:
                try:
                    teams = match_entry["teams"]
                    pred = predictor.model.predict_pre_match(teams[0], teams[1], data.get('venue', 'Unknown'))
                    match_entry["win_probability"] = float(pred.get('win_probability', 0.5))
                except Exception:
                    pass
                    
            matches.append(match_entry)
        
        return matches
    except Exception as e:
        logger.error(f"Error listing matches: {e}")
        return []

@app.get("/upcoming/{season}")
async def get_upcoming_matches(season: str):
    """Fetch upcoming schedule dynamically from the injected CSV."""
    import csv
    import os
    from datetime import datetime, timedelta, timezone

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    schedule_path = os.path.join(os.path.dirname(__file__), 'data_pipeline', 'ipl_2026_schedule.csv')
    upcoming = []

    try:
        if os.path.exists(schedule_path):
            with open(schedule_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = f"{row['Date']} {row['Time (IST)']}"
                    try:
                        dt = datetime.strptime(date_str, "%b %d, %Y %I:%M %p").replace(tzinfo=ist_tz)
                        if dt.timestamp() > time.time():
                            teams = row['Match details'].split(' vs ')
                            teama = teams[0].strip() if len(teams) > 0 else "TBD"
                            teamb = teams[1].strip() if len(teams) > 1 else "TBD"
                            
                            upcoming.append({
                                "matchdate": row['Date'],
                                "teama": teama,
                                "teamb": teamb,
                                "venue": row['Venue'],
                                "win_probability": 0.5 # Default
                            })

                            if predictor:
                                try:
                                    # Fix: Use normalized names for prediction if needed, 
                                    # but predict_pre_match handles normalization.
                                    pred = predictor.model.predict_pre_match(teama, teamb, row['Venue'])
                                    upcoming[-1]['win_probability'] = float(pred.get('win_probability', 0.5))
                                except Exception as e:
                                    logger.warning(f"Failed pre-match prediction for {teama} vs {teamb}: {e}")
                    except Exception:
                        continue
                        
        return {"matchschedule": upcoming}
    except Exception as e:
        logger.error(f"Error fetching upcoming schedule: {e}")
        return {"matchschedule": []}

@app.get("/points/{season}")
async def get_points_table(season: str):
    """Fetch Points Table. For now, returns a base template for 2026."""
    teams = [
        {"name": "Chennai Super Kings", "teamshortname": "CSK", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Delhi Capitals", "teamshortname": "DC", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Gujarat Titans", "teamshortname": "GT", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Kolkata Knight Riders", "teamshortname": "KKR", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Lucknow Super Giants", "teamshortname": "LSG", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Mumbai Indians", "teamshortname": "MI", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Punjab Kings", "teamshortname": "PBKS", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Rajasthan Royals", "teamshortname": "RR", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Royal Challengers Bengaluru", "teamshortname": "RCB", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
        {"name": "Sunrisers Hyderabad", "teamshortname": "SRH", "matchesplayed": 0, "matcheswon": 0, "matcheslost": 0, "points": 0, "nrr": "+0.00"},
    ]
    return {"points": teams}



@app.get("/predict/{match_id}")
async def get_prediction(match_id: str):
    """Fetch latest win probability with uncertainty intervals, SHAP factors, and betting odds."""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML Engine not initialized")
    try:
        prediction = predictor.predict_live_match(match_id)
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])

        # ── Enrich with Scenario Simulator ──────────────────────────────────
        scenario_sim = get_scenario_sim()
        if scenario_sim:
            try:
                scenario = scenario_sim.simulate_remaining_balls(
                    current_state=prediction, n_simulations=2000
                )
                prediction['scenario'] = scenario
                # Surface key milestones at top level for the frontend
                prediction['prob_180_plus'] = scenario.get('prob_180_plus', 0.0)
                prediction['projected_score_p90'] = scenario.get('p90_score', 0.0)
            except Exception as sim_err:
                logger.warning(f"Scenario simulator skipped: {sim_err}")

        # ── Enrich with Agent Swarm ──────────────────────────────────────────
        swarm = get_agent_swarm()
        if swarm:
            try:
                agent_features = swarm.simulate(prediction)
                prediction.update(agent_features)
            except Exception as swarm_err:
                logger.warning(f"Agent swarm skipped: {swarm_err}")

        # ── Enrich with Betting Odds ─────────────────────────────────────────
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

        # ── Persist asynchronously ───────────────────────────────────────────
        if db_manager:
            asyncio.create_task(db_manager.save_prediction(prediction))

        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {match_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")


@app.get("/simulate/scenario/{match_id}")
async def run_scenario_simulation(match_id: str, n: int = 5000):
    """Run detailed Monte Carlo scenario simulation for remaining balls."""
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    last_ball = r.xrevrange(f"ipl:balls:{match_id}", count=1)
    if not last_ball:
        raise HTTPException(status_code=404, detail="No match data found")

    d = last_ball[0][1]
    current_state = {
        'total_runs': int(d.get('total_runs', 0) or 0),
        'total_wickets': int(d.get('total_wickets', 0) or 0),
        'balls_remaining': max(0, 120 - int(float(d.get('over', 0)) * 6)),
    }

    sim = get_scenario_sim()
    if not sim:
        raise HTTPException(status_code=503, detail="Simulator unavailable")

    result = sim.simulate_remaining_balls(current_state, n_simulations=min(n, 10000))
    return {"match_id": match_id, "current_state": current_state, "simulation": result}


@app.get("/fantasy/{player_name}")
async def get_fantasy_projection(player_name: str, ownership: float = 0.5, role: str = 'batsman'):
    """Get Bayesian fantasy points projection for a player."""
    try:
        from backend.ml_engine.simulators import BayesianPlayerPredictor, FantasyEngine
        from backend.ml_engine.hybrid_model import CricsheetNormalizer

        normalizer = CricsheetNormalizer()
        normalizer.load_player_stats()

        # Fetch historical stats from player_stats_db if available
        bat_history, wkt_history = [], []
        if normalizer.player_stats:
            bat_data = normalizer.player_stats.get('batsmen', {}).get(player_name, {})
            bowl_data = normalizer.player_stats.get('bowlers', {}).get(player_name, {})
            # Simulate last-5 form from season average
            if bat_data.get('average'):
                avg = bat_data['average']
                bat_history = [max(0, avg + np.random.normal(0, avg * 0.3)) for _ in range(5)]
            if bowl_data.get('average'):
                wkt_history = [max(0, np.random.normal(1.2, 0.5)) for _ in range(5)]

        predictor_b = BayesianPlayerPredictor()
        run_proj = predictor_b.predict_player_runs(bat_history, opp_strength=1.0)
        wkt_proj = predictor_b.predict_player_wickets(wkt_history)

        projection = {**run_proj, **wkt_proj}
        fantasy = FantasyEngine()
        fantasy_pts = fantasy.calculate_expected_points(
            player_projection=projection,
            role=role,
            ownership_pct=float(ownership)
        )

        return {
            "player": player_name,
            "role": role,
            "ownership_pct": round(ownership * 100, 1),
            "batting_projection": run_proj,
            "bowling_projection": wkt_proj,
            "fantasy": fantasy_pts
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
