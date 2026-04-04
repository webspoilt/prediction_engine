import sys
import os
import asyncio
import logging

# Setup mock logging
logging.basicConfig(level=logging.ERROR)

async def test_v4_8():
    try:
        from backend.data_pipeline.multi_source_fetcher import get_fetcher
        fetcher = get_fetcher()
        print("✅ MultiSourceFetcher import successful.")
        
        matches = await fetcher.discover_matches()
        print(f"✅ Discovered {len(matches)} matches.")
        
        # Check Priority
        if matches:
            top = matches[0]
            print(f"✅ Top Match: {top['teams'][0]} vs {top['teams'][1]} ({top['match_id']})")
            if 'ipl2026_9' in top['match_id']:
                print("⭐ Priority 1: Match 9 (GT vs RR) correctly pinned.")
            elif top['status'] == 'live':
                print("⭐ Priority 1: Live match pinned.")
        
        from backend.ml_engine.hybrid_model import RealTimePredictor
        print("✅ Hybrid Model import successful.")
        
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_v4_8())
