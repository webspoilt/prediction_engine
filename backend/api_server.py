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
from typing import List, Dict, Optional
from pydantic import BaseModel
from backend.ml_engine.hybrid_model import RealTimePredictor
from backend.ml_engine.agent_sim import MultiAgentSimulator

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        logger.info("🚀 Starting API Server and loading ML Engine...")
        app.state.redis_pool = redis.ConnectionPool(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=REDIS_DB, 
            password=REDIS_PASSWORD, 
            decode_responses=True
        )
        predictor = RealTimePredictor()
        logger.info("✅ ML Engine successfully loaded into memory.")
    except Exception as e:
        logger.error(f"❌ Failed to load Predictor: {e}")
    yield
    logger.info("Shutting down...")
    if hasattr(app.state, 'redis_pool'):
        app.state.redis_pool.disconnect()

app = FastAPI(
    title="IPL Prediction Engine API",
    description="Real-time match win probability engine with XGBoost + LSTM",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enrichment Layers
agent_swarm = MultiAgentSimulator(num_agents=10)

class SimulationRequest(BaseModel):
    num_simulations: int = 1000
    overs_to_project: int = 5
    captain_policy: str = "aggressive" # aggressive, balanced, defensive

class MatchSimulator:
    """Monte Carlo engine for match projection"""
    def project_future(self, current_state: Dict, num_sims: int, overs: int, policy: str):
        results = []
        for _ in range(num_sims):
            temp_runs = 0
            temp_wickets = 0
            for _ in range(overs * 6): # ball by ball
                # Simple probabilistic transition (can be improved with Transformer)
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

# Environment Config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Global Predictor Instance
predictor: Optional[RealTimePredictor] = None

# Startup logic now handled by lifespan context manager

@app.get("/health")
async def health_check():
    """System health monitor"""
    res = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "ml_engine": "loaded" if predictor else "initializing",
            "redis": "checking..."
        }
    }
    
    try:
        r = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=REDIS_DB, 
            password=REDIS_PASSWORD, 
            decode_responses=True
        )
        r.ping()
        res["components"]["redis"] = "connected"
    except Exception as e:
        res["status"] = "degraded"
        res["components"]["redis"] = f"error: {str(e)}"
        
    return res

@app.get("/matches")
async def list_matches():
    """List all active and recent matches in the system"""
    try:
        r = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=REDIS_DB, 
            password=REDIS_PASSWORD, 
            decode_responses=True
        )
        # Search for ball streams: ipl:balls:{match_id}
        keys = r.keys("ipl:balls:*")
        match_ids = [k.split(":")[-1] for k in keys]
        
        matches = []
        for mid in match_ids:
            # Fetch last ball for basic status
            last_ball = r.xrevrange(f"ipl:balls:{mid}", count=1)
            if last_ball:
                data = last_ball[0][1]
                matches.append({
                    "match_id": mid,
                    "teams": [data.get('batting_team', 'N/A'), data.get('bowling_team', 'N/A')],
                    "inning": data.get('inning'),
                    "score": f"{data.get('runs', 0)}/{data.get('wicket', 0)}",
                    "over": data.get('over')
                })
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{match_id}")
async def get_prediction(match_id: str):
    """Fetch the latest win probability for a match"""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML Engine not initialized")
        
    try:
        prediction = predictor.predict_live_match(match_id)
    
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])

        # Enrich with Multi-Agent Simulation features (MiroFish-style)
        # We use current match data as seed for the swarm
        agent_features = agent_swarm.simulate(prediction)
        prediction.update(agent_features)
        
        return prediction
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Prediction Error for {match_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

@app.post("/simulate/{match_id}")
async def run_what_if_simulation(match_id: str, request: SimulationRequest):
    """
    Project future match outcomes using Monte Carlo simulations.
    Allows testing 'What If' tactical scenarios.
    """
    prediction = predictor.predict_live_match(match_id)
    if "error" in prediction:
        raise HTTPException(status_code=404, detail="No live data to simulate from")
    
    projection = match_sim.project_future(
        prediction, 
        request.num_simulations, 
        request.overs_to_project, 
        request.captain_policy
    )
    
    return {
        "match_id": match_id,
        "current_state": prediction,
        "projection": projection
    }

@app.websocket("/ws/predictions/{match_id}")
async def websocket_prediction(websocket: WebSocket, match_id: str):
    """Real-time prediction stream via WebSocket"""
    await websocket.accept()
    logger.info(f"WebSocket connected for match: {match_id}")
    
    r = None
    pubsub = None
    
    try:
        r = redis.Redis(connection_pool=app.state.redis_pool)
        pubsub = r.pubsub()
        pubsub.subscribe(f"ipl:predictions:{match_id}")
        
        # Initial push
        if predictor:
            initial_predict = predictor.predict_live_match(match_id)
            if 'error' not in initial_predict:
                await websocket.send_json(initial_predict)
        
        while True:
            # Poll for new messages from PubSub
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                data = json.loads(message['data'])
                await websocket.send_json(data)
            
            # Keep-alive and throttle
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {match_id}")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        if pubsub:
            pubsub.unsubscribe()
            pubsub.close()
        if r:
            r.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
