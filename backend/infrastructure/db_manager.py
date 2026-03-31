import os
import json
import logging
import asyncio
import asyncpg
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages persistence to external PostgreSQL (Supabase/Neon).
    Used for storing long-term match data, predictions, and model metadata.
    """
    
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize connection pool"""
        if not self.dsn:
            logger.error("DATABASE_URL not found in environment.")
            return False
            
        try:
            self.pool = await asyncpg.create_pool(dsn=self.dsn)
            logger.info("✅ Connected to PostgreSQL successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

    async def initialize_schema(self):
        """Creates required tables if they don't exist"""
        if not self.pool: return
        
        queries = [
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                teams TEXT[],
                venue TEXT,
                date DATE,
                status TEXT DEFAULT 'scheduled',
                winner TEXT,
                metadata JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                match_id TEXT REFERENCES matches(match_id),
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                win_probability FLOAT,
                confidence FLOAT,
                inning INTEGER,
                over FLOAT,
                score TEXT,
                raw_data JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                version TEXT PRIMARY KEY,
                repo_id TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                metrics JSONB
            );
            """
        ]
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for query in queries:
                    await conn.execute(query)
        logger.info("Database schema initialized.")

    async def save_match(self, match_data: Dict):
        """Upsert match info"""
        if not self.pool: return
        
        query = """
            INSERT INTO matches (match_id, teams, venue, date, status, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (match_id) DO UPDATE 
            SET status = EXCLUDED.status, metadata = matches.metadata || EXCLUDED.metadata;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query, 
                match_data['match_id'],
                match_data.get('teams'),
                match_data.get('venue'),
                match_data.get('date'),
                match_data.get('status', 'scheduled'),
                json.dumps(match_data.get('metadata', {}))
            )

    async def save_prediction(self, prediction: Dict):
        """Log a prediction point"""
        if not self.pool: return
        
        query = """
            INSERT INTO predictions (match_id, win_probability, confidence, inning, over, score, raw_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7);
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                prediction['match_id'],
                prediction['win_probability'],
                prediction.get('confidence'),
                prediction.get('inning'),
                prediction.get('over'),
                prediction.get('score'),
                json.dumps(prediction)
            )

    async def get_latest_predictions(self, match_id: str, limit: int = 10) -> List[Dict]:
        """Fetch historical predictions for chart display"""
        if not self.pool: return []
        
        query = "SELECT * FROM predictions WHERE match_id = $1 ORDER BY timestamp DESC LIMIT $2"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, match_id, limit)
            return [dict(r) for r in rows]

async def test_db():
    # To be used in CLI tests
    db = DatabaseManager()
    if await db.connect():
        await db.initialize_schema()
        await db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_db())
