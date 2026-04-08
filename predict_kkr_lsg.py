import numpy as np
import os
import sys
import json

# Ensure Python can find the local modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from ml_engine.hybrid_model import RealTimePredictor, ModelConfig

def predict_match():
    print("🔮 Sovereign Oracle: High-Frequency Audit — KKR vs LSG (Match 15, IPL 2026)")
    
    predictor = RealTimePredictor()
    
    # 1. Match Parameters
    match_id = "ipl2026_15"
    t1, t2 = "KKR", "LSG"
    venue = "Eden Gardens, Kolkata"
    
    # 2. Run Pre-Match Prediction
    # This uses the same logic as the API server
    result = predictor.model.predict_pre_match(t1, t2, venue)
    
    # 3. Generate Intelligence (The 'Walkthrough' style details)
    # We create a dummy match_data to feed into the intelligence generator
    match_data = {
        "match_id": match_id,
        "teams": [t1, t2],
        "venue": venue,
        "status": "scheduled",
        "win_probability": result['win_probability']
    }
    
    intel = predictor._generate_live_intelligence(match_data, result['win_probability'])
    
    # 4. Output the Verdict
    print("\n" + "═"*60)
    print(f"      SOVEREIGN VERDICT: {t1} vs {t2}")
    print("═"*60)
    print(f"🏟️  Venue:   {venue}")
    print(f"📈 Odds Implied: KKR @ 1.85 | LSG @ 1.95 (Sovereign Spread)")
    print("-" * 60)
    
    wp = result['win_probability']
    print(f"🏆 Win Probability ({t1}):  {wp*100:.1f}%")
    print(f"🏆 Win Probability ({t2}):  {(1-wp)*100:.1f}%")
    print("-" * 60)
    
    print("🧠 Engine Intelligence (Match Preview):")
    for p in intel.get('predictions', []):
        print(f"  ● {p['factor']}: {p['prediction']} (Conf: {p['confidence']})")
    
    print("-" * 60)
    print(f"🎰 Betting Strategy: {intel.get('betting', {}).get('recommendation', 'N/A')}")
    print(f"⚠️  Volatility: {intel.get('betting', {}).get('volatility', 'N/A')}")
    print("═"*60)

if __name__ == "__main__":
    predict_match()
