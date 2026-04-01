"""
Centralized Configuration for IPL Prediction Engine.
Single source of truth — replaces scattered os.getenv() calls.
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # ─── API Server ───
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 7860
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:7860"
    DEBUG: bool = False

    # ─── Redis ───
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_ENABLED: bool = True  # Set False for dev without Redis

    # ─── Database (Neon/Supabase PostgreSQL) ───
    DATABASE_URL: Optional[str] = None

    # ─── Hugging Face Hub ───
    HF_REPO_ID: str = "zeroday01/ipl_prediction_engine"
    HF_TOKEN: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # ─── OpenAI (Optional — for MultiAgent LLM mode) ───
    OPENAI_API_KEY: Optional[str] = None

    # ─── Twitter/Sentiment (Optional) ───
    TWITTER_CONSUMER_KEY: Optional[str] = None
    TWITTER_CONSUMER_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = None

    # ─── ML Engine ───
    MODEL_PATH: str = "models/hybrid_ensemble"
    DATA_DIR: str = "data"
    TRAINING_BATCH_SIZE: int = 64
    TRAINING_CHUNK_SIZE: int = 50

    # ─── Data Pipeline ───
    DISCOVERY_POLL_INTERVAL: int = 300  # seconds
    SCRAPER_POLL_INTERVAL: int = 5  # seconds
    RETRAIN_CHECK_INTERVAL: int = 3600  # 1 hour

    # ─── Betting Engine ───
    BOOKMAKER_MARGIN: float = 0.05  # 5% overround
    MIN_CONFIDENCE_THRESHOLD: float = 0.95  # 95% minimum

    @property
    def origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance
settings = Settings()
