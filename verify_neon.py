import asyncio
import os
import asyncpg
import sys

async def verify_neon():
    dsn = "postgresql://neondb_owner:npg_Ibc2WMd9TOfl@ep-fancy-hat-amuddfn9-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    print(f"Connecting to: {dsn[:30]}...")
    try:
        conn = await asyncpg.connect(dsn)
        print("✅ Connection established!")
        tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        print("✅ Tables:")
        for t in tables:
            print(f" - {t['table_name']}")
        await conn.close()
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(verify_neon())
