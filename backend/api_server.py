import os
import json
import time
import asyncio
import logging
import redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from backend.ml_engine.hybrid_model import RealTimePredictor

# Environment Config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IPL Prediction Engine API",
    description="Real-time match win probability engine with XGBoost + LSTM",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor: Optional[RealTimePredictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        logger.info("🚀 Starting API Server and loading ML Engine...")
        predictor = RealTimePredictor()
        logger.info("✅ ML Engine successfully loaded into memory.")
    except Exception as e:
        logger.error(f"❌ Failed to load Predictor: {e}")

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
        result = predictor.predict_live_match(match_id)
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        return result
    except Exception as e:
        logger.error(f"Prediction Error for {match_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

@app.websocket("/ws/predictions/{match_id}")
async def websocket_prediction(websocket: WebSocket, match_id: str):
    """Real-time prediction stream via WebSocket"""
    await websocket.accept()
    logger.info(f"WebSocket connected for match: {match_id}")
    
    r = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=REDIS_DB, 
        password=REDIS_PASSWORD, 
        decode_responses=True
    )
    pubsub = r.pubsub()
    pubsub.subscribe(f"ipl:predictions:{match_id}")
    
    try:
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
        pubsub.unsubscribe()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
