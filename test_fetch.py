import asyncio
from backend.data_pipeline.multi_source_fetcher import MultiSourceFetcher

async def main():
    f = MultiSourceFetcher()
    res = await f.get_live_matches_waterfall()
    print("LIVE MATCHES:", res)

asyncio.run(main())
