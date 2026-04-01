"""
IPL Prediction Engine — FastAPI Application
=============================================
Main entry-point. Mounts:
  • REST endpoints  (/predict/{match_id}, /health)
  • WebSocket       (/ws/odds/{match_id})
  • Static frontend  (/  → serves index.html)

Integration Guide
-----------------
To wire `calculate_betting_metrics` into your EXISTING `/predict/{match_id}`
endpoint, locate the "⚠️ INTEGRATION POINT" comments below. Copy the
highlighted block into your own route — nothing else needs to change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from models import (
    BallData,
    BettingMetrics,
    EliteInsights,
    WsOddsTick,
    WsPropBetUpdate,
    HistoricalOddsSnapshot,
)
from betting_metrics import calculate_betting_metrics

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ipl-engine")

# ─── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks — initialise shared state here."""
    # In-memory odds-history ring buffer per match (max 200 ticks)
    app.state.odds_history: dict[str, list[HistoricalOddsSnapshot]] = {}
    # WebSocket connection manager
    app.state.ws_clients: dict[str, list[WebSocket]] = {}
    logger.info("IPL Prediction Engine started ✓")
    yield
    logger.info("IPL Prediction Engine shutting down …")


# ─── App Factory ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="IPL Premium Prediction Engine",
    version="2.0.0",
    description="Hybrid XGBoost + LSTM model with Premium Betting Insights",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Simulated Model Inference (replace with your actual model) ──────────
def _run_xgboost_inference(match_id: str, ball: BallData) -> float:
    """
    ⚠️ REPLACE this stub with your actual XGBoost model inference.
    Expected return: win probability ∈ (0, 1).
    """
    # Simulated: slight random walk around 0.5 + innings context
    base = 0.45 + 0.05 * ball.total_runs / 160.0 - 0.03 * ball.wickets_fallen
    return float(np.clip(base + 0.02 * np.random.randn(), 0.05, 0.95))


def _run_lstm_inference(match_id: str, ball: BallData) -> float:
    """
    ⚠️ REPLACE this stub with your actual LSTM model inference.
    Expected return: win probability ∈ (0, 1).
    """
    # Simulated LSTM with slightly different prior
    base = 0.48 + 0.04 * ball.total_runs / 160.0 - 0.025 * ball.wickets_fallen
    return float(np.clip(base + 0.015 * np.random.randn(), 0.05, 0.95))


# ═══════════════════════════════════════════════════════════════════════════
#  REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "ipl-premium-v2"}


@app.get("/predict/{match_id}")
async def predict_match(
    match_id: str,
    over: float = Query(10.0, ge=0, le=20),
    ball: int = Query(1, ge=1, le=6),
    total_runs: int = Query(85, ge=0),
    wickets: int = Query(2, ge=0, le=10),
    striker_runs: int = Query(35, ge=0),
    striker_balls: int = Query(22, ge=0),
    target: Optional[int] = Query(None, ge=1),
):
    """
    ⚠️ INTEGRATION POINT
    ────────────────────
    This endpoint is a DROP-IN REPLACEMENT for your existing /predict/{match_id}.
    The `elite_insights` key is appended — your existing keys stay untouched.

    Copy the block marked with ╔══ ELITE INSIGHTS INJECTION ══╗ into your
    existing handler. Everything else is your original code.
    """

    # ── Build ball context ────────────────────────────────────────────
    ball_data = BallData(
        over=over,
        ball=ball,
        total_runs=total_runs,
        wickets_fallen=wickets,
        striker_runs=striker_runs,
        striker_balls=striker_balls,
        non_striker_runs=int(striker_runs * 0.6),
        non_striker_balls=int(striker_balls * 0.8),
        chasing_target=target,
    )

    # ── Your existing model inference ─────────────────────────────────
    xgb_prob = _run_xgboost_inference(match_id, ball_data)
    lstm_prob = _run_lstm_inference(match_id, ball_data)

    # Hybrid blend (XGBoost 60 % + LSTM 40 %)
    blend_weights = {"xgboost": 0.6, "lstm": 0.4}
    win_probability = (
        blend_weights["xgboost"] * xgb_prob
        + blend_weights["lstm"] * lstm_prob
    )

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║                ⚡ ELITE INSIGHTS INJECTION ⚡                   ║
    # ║  Copy this block into your existing /predict endpoint.          ║
    # ║  It requires: ball_data (BallData), win_probability (float).    ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # Fetch odds history for volatility (optional but recommended)
    history = app.state.odds_history.get(match_id, [])

    # Core betting-metrics computation — O(1), no GPU needed
    betting_metrics: BettingMetrics = calculate_betting_metrics(
        probability=win_probability,
        ball_data=ball_data,
        odds_history=history,
    )

    # Compute sharp edge: |model_prob - implied_prob| / implied_prob
    implied = betting_metrics.decimal_odds.implied_home
    sharp_edge = abs(win_probability - implied) / max(implied, 0.01)

    # Package into EliteInsights envelope
    elite_insights = EliteInsights(
        betting_metrics=betting_metrics,
        model_blend_weights=blend_weights,
        sharp_edge_pct=round(sharp_edge * 100, 2),
        last_updated_ms=int(time.time() * 1000),
    )

    # Update odds history ring buffer
    snapshot = HistoricalOddsSnapshot(
        timestamp_ms=int(time.time() * 1000),
        decimal_odds=betting_metrics.decimal_odds.home,
    )
    if match_id not in app.state.odds_history:
        app.state.odds_history[match_id] = []
    app.state.odds_history[match_id].append(snapshot)
    if len(app.state.odds_history[match_id]) > 200:
        app.state.odds_history[match_id] = app.state.odds_history[match_id][-200:]

    # ═════════════════════════════════════════════════════════════════

    # ── Your existing response structure (extended with elite_insights) ─
    return {
        "match_id": match_id,
        "prediction": {
            "win_probability": round(win_probability, 4),
            "xgboost_prob": round(xgb_prob, 4),
            "lstm_prob": round(lstm_prob, 4),
            "predicted_total": round(betting_metrics.over_under_lines[0].projected_total, 1),
            "innings_phase": betting_metrics.innings_phase.value,
        },
        # ⚡ NEW FIELD — the premium insights payload
        "elite_insights": elite_insights.model_dump(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET — Real-Time Odds Stream
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket clients per match."""

    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, match_id: str, ws: WebSocket):
        await ws.accept()
        if match_id not in self.active:
            self.active[match_id] = []
        self.active[match_id].append(ws)
        logger.info(f"WS client connected for match {match_id} "
                     f"(total: {len(self.active[match_id])})")

    def disconnect(self, match_id: str, ws: WebSocket):
        if match_id in self.active:
            self.active[match_id].remove(ws)
            if not self.active[match_id]:
                del self.active[match_id]

    async def broadcast(self, match_id: str, message: dict):
        if match_id not in self.active:
            return
        dead = []
        for ws in self.active[match_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(match_id, ws)


manager = ConnectionManager()


@app.websocket("/ws/odds/{match_id}")
async def ws_odds_stream(websocket: WebSocket, match_id: str):
    """
    Real-time WebSocket endpoint for live odds ticks.
    Clients receive two types of messages:
      1. WsOddsTick   — every 2 seconds (odds + volatility)
      2. WsPropBetUpdate — every 5 seconds (prop-bet recalc)
    """
    await manager.connect(match_id, websocket)

    # Send initial snapshot immediately
    ball_data = BallData(over=10, ball=1, total_runs=85, wickets_fallen=2,
                         striker_runs=35, striker_balls=22)
    history = app.state.odds_history.get(match_id, [])
    metrics = calculate_betting_metrics(0.55, ball_data, history)
    tick = WsOddsTick(
        match_id=match_id,
        timestamp_ms=int(time.time() * 1000),
        decimal_odds=metrics.decimal_odds,
        market_volatility=metrics.market_volatility,
        delta_label="→",
    )
    await websocket.send_json(tick.model_dump())

    try:
        odds_counter = 0
        prop_counter = 0

        while True:
            # Simulate live ball updates (replace with your real feed)
            await asyncio.sleep(0.5)  # 2 Hz tick rate

            # Simulate progressing match state
            ball_data.total_runs += int(np.random.choice([0, 1, 1, 2, 4, 6],
                                                          p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.03]))
            ball_data.ball += 1
            if ball_data.ball > 6:
                ball_data.ball = 1
                ball_data.over += 1
                ball_data.striker_runs = int(np.random.randint(0, 50))
                ball_data.striker_balls = int(np.random.randint(5, 40))
            if ball_data.over >= 20:
                ball_data.over = 10  # Reset for demo

            history = app.state.odds_history.get(match_id, [])
            win_p = float(np.clip(0.55 + 0.01 * np.random.randn(), 0.10, 0.90))
            metrics = calculate_betting_metrics(win_p, ball_data, history)

            # Update ring buffer
            snapshot = HistoricalOddsSnapshot(
                timestamp_ms=int(time.time() * 1000),
                decimal_odds=metrics.decimal_odds.home,
            )
            if match_id not in app.state.odds_history:
                app.state.odds_history[match_id] = []
            app.state.odds_history[match_id].append(snapshot)
            if len(app.state.odds_history[match_id]) > 200:
                app.state.odds_history[match_id] = app.state.odds_history[match_id][-200:]

            # Determine delta direction
            prev_snapshot = history[-1] if history else None
            delta = "→"
            if prev_snapshot:
                diff = metrics.decimal_odds.home - prev_snapshot.decimal_odds
                delta = "↑" if diff > 0.005 else "↓" if diff < -0.005 else "→"

            odds_counter += 1
            prop_counter += 1

            # Broadcast odds tick every 4 ticks (~2 s)
            if odds_counter >= 4:
                odds_counter = 0
                tick = WsOddsTick(
                    match_id=match_id,
                    timestamp_ms=int(time.time() * 1000),
                    decimal_odds=metrics.decimal_odds,
                    market_volatility=metrics.market_volatility,
                    delta_label=delta,
                )
                await manager.broadcast(match_id, tick.model_dump())

            # Broadcast prop-bet update every 10 ticks (~5 s)
            if prop_counter >= 10:
                prop_counter = 0
                prop_update = WsPropBetUpdate(
                    match_id=match_id,
                    timestamp_ms=int(time.time() * 1000),
                    prop_bets=metrics.prop_bets,
                )
                await manager.broadcast(match_id, prop_update.model_dump())

    except WebSocketDisconnect:
        manager.disconnect(match_id, websocket)
        logger.info(f"WS client disconnected from match {match_id}")


# ═══════════════════════════════════════════════════════════════════════════
#  STATIC FRONTEND
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the Pro-Analytics dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend/</h1>")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
