"""
IPL Premium Betting Insights — Pydantic Data Models
=====================================================
Defines every schema flowing through the prediction + betting pipeline.
All models are JSON-serialisable and typed for IDE / OpenAPI docs.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────
class WicketMethod(str, Enum):
    CAUGHT = "Caught"
    BOWLED = "Bowled"
    LBW = "LBW"
    RUN_OUT = "Run Out"
    STUMPED = "Stumped"
    HIT_WICKET = "Hit Wicket"
    RETIRED_OUT = "Retired Out"


class InningsPhase(str, Enum):
    POWERPLAY = "powerplay"       # Overs 1-6
    MIDDLE = "middle"             # Overs 7-15
    DEATH = "death"               # Overs 16-20


# ─── Request / Context Models ────────────────────────────────────────────
class BallData(BaseModel):
    """Raw ball-level feed consumed by the metrics engine."""
    over: float = Field(..., ge=0, le=20, description="Current over")
    ball: int = Field(..., ge=1, le=6, description="Ball within the over")
    runs_scored: int = Field(0, ge=0, description="Runs scored off this ball")
    wickets_fallen: int = Field(0, ge=0, le=10, description="Wickets down so far")
    total_runs: int = Field(0, ge=0, description="Team total at this point")
    striker_runs: int = Field(0, ge=0, description="Batsman on strike — runs")
    striker_balls: int = Field(0, ge=0, description="Batsman on strike — balls faced")
    non_striker_runs: int = Field(0, ge=0)
    non_striker_balls: int = Field(0, ge=0)
    is_wicket_ball: bool = False
    wicket_method: Optional[WicketMethod] = None
    extra_type: Optional[str] = None          # wide, noball, bye, legbye
    bowler_wickets: int = Field(0, ge=0)
    bowler_runs: int = Field(0, ge=0)
    bowler_overs: float = Field(0, ge=0, le=4)
    venue: str = "neutral"
    chasing_target: Optional[int] = None


class HistoricalOddsSnapshot(BaseModel):
    """A single point in the odds-history timeline for volatility calc."""
    timestamp_ms: int
    decimal_odds: float


# ─── Betting-Metrics Response Models ─────────────────────────────────────
class DecimalOdds(BaseModel):
    home: float = Field(..., gt=0, description="Home team decimal odds")
    away: float = Field(..., gt=0, description="Away team decimal odds")
    implied_home: float = Field(..., ge=0, le=1)
    implied_away: float = Field(..., ge=0, le=1)
    overround: float = Field(..., description="Bookmaker margin (0 = fair)")

    class Config:
        json_schema_extra = {
            "example": {
                "home": 1.85, "away": 2.05,
                "implied_home": 0.5405, "implied_away": 0.4878,
                "overround": 0.0283
            }
        }


class OverUnderLine(BaseModel):
    market: str = Field(..., description="e.g. 'Total Runs', 'Team A Runs'")
    line: float
    over_odds: float
    under_odds: float
    projected_total: float = Field(..., description="Model-projected value")
    confidence: float = Field(..., ge=0, le=1)


class NextWicketMethodProb(BaseModel):
    method: WicketMethod
    probability: float = Field(..., ge=0, le=1)
    pressure_multiplier: float = Field(1.0, description="Adjusted by match context")


class MarketVolatility(BaseModel):
    score: float = Field(..., ge=0, description="0 = calm, 1 = extreme")
    label: str = Field(..., description="Low / Medium / High / Extreme")
    tick_velocity: float = Field(..., description="Odds-shifts per minute")
    max_shift_pct: float = Field(..., description="Largest single-tick % shift")


class PropBetPlayer(BaseModel):
    player_name: str
    milestone: str = Field(..., description="e.g. '50+', '100+', '3+ Wickets'")
    probability: float = Field(..., ge=0, le=1)
    current_value: int = Field(..., description="Runs / wickets right now")
    balls_remaining: int = Field(..., ge=0)
    strike_rate: Optional[float] = None


class BettingMetrics(BaseModel):
    """Top-level response returned by `calculate_betting_metrics`."""
    decimal_odds: DecimalOdds
    over_under_lines: list[OverUnderLine]
    next_wicket_method: list[NextWicketMethodProb]
    market_volatility: MarketVolatility
    prop_bets: list[PropBetPlayer]
    innings_phase: InningsPhase
    projection_confidence: float = Field(..., ge=0, le=1)


# ─── Elite Insights Envelope (attached to /predict) ─────────────────────
class EliteInsights(BaseModel):
    betting_metrics: BettingMetrics
    model_blend_weights: dict[str, float] = Field(
        default_factory=lambda: {"xgboost": 0.6, "lstm": 0.4}
    )
    sharp_edge_pct: float = Field(
        ..., description="% edge model has over closing line"
    )
    last_updated_ms: int


# ─── WebSocket Messages ─────────────────────────────────────────────────
class WsOddsTick(BaseModel):
    """Pushed every time the odds engine recalculates."""
    match_id: str
    timestamp_ms: int
    decimal_odds: DecimalOdds
    market_volatility: MarketVolatility
    delta_label: str = Field(..., description="↑ / ↓ / →")


class WsPropBetUpdate(BaseModel):
    match_id: str
    timestamp_ms: int
    prop_bets: list[PropBetPlayer]
