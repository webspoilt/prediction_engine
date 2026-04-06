import numpy as np
import torch
import sys
import os
import json

# Ensure Python can find the local modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from ml_engine.hybrid_model import HybridEnsemble, ModelConfig

def run_prediction():
    print("🔮 Sovereign Oracle: Analyzing KKR vs PBKS (Match 12, IPL 2026)...")
    
    ensemble = HybridEnsemble()
    
    # Load the models using the correct path prefix
    try:
        model_path = os.path.join(os.getcwd(), 'models', 'hybrid_ensemble')
        ensemble.load_models(model_path)
        print("✅ Titan ML Brain Synchronized.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 1. Prepare Pre-Match Static Features
    # 'inning', 'over', 'total_runs', 'total_wickets', 'crr', 
    # 'runs_last_6', 'wickets_last_6', 'boundary_rate', 'dot_pressure', 
    # 'balls_remaining', 'bat_sr', 'bat_avg', 'bat_bp', 'bowl_econ', 
    # 'bowl_sr', 'bowl_avg', 'temp', 'humidity', 'dew', 
    # 'bat_elo', 'bowl_elo', 'elo_diff'
    
    static_features = np.zeros((1, 22))
    static_features[0, 0] = 1.0     # 1st Inning
    static_features[0, 1] = 0.0     # 0 Overs
    static_features[0, 2] = 0.0     # 0 Runs
    static_features[0, 3] = 0.0     # 0 Wickets
    static_features[0, 4] = 0.0     # 0 CRR
    static_features[0, 5] = 0.0     # 0 Runs last 6
    static_features[0, 6] = 0.0     # 0 Wickets last 6
    static_features[0, 7] = 0.0     # 0 Boundary Rate
    static_features[0, 8] = 0.0     # 0 Dot Pressure
    static_features[0, 9] = 120.0   # 120 Balls Remaining
    
    # Player Stats (Averages for the teams)
    static_features[0, 10] = 138.0  # bat_sr (KKR/PBKS mix)
    static_features[0, 11] = 28.5   # bat_avg
    static_features[0, 12] = 18.0   # bat_bp
    static_features[0, 13] = 8.4    # bowl_econ
    static_features[0, 14] = 23.5   # bowl_sr
    static_features[0, 15] = 29.0   # bowl_avg
    
    # Weather/Environment (Eden Gardens Evening)
    static_features[0, 16] = 29.5   # temp
    static_features[0, 17] = 72.0   # humidity (Kolkata is humid)
    static_features[0, 18] = 24.0   # dew (likely high in April)
    
    # ELO Ratings (Form Factor)
    # PBKS is 2-0 (ELO up), KKR is 0-2 (ELO down)
    static_features[0, 19] = 1520.0 # KKR ELO (Home)
    static_features[0, 20] = 1580.0 # PBKS ELO (Form)
    static_features[0, 21] = -60.0  # elo_diff
    
    # 2. Prepare empty sequence features
    sequence_features = np.zeros((1, 18, 3)) # (batch, seq_len, [runs, wicket, over])

    print("🧠 Running Hybrid Inference (XGBoost + LSTM Transformer)...")
    prediction = ensemble.predict(
        static_features=static_features,
        sequence_features=sequence_features,
        return_confidence=True
    )
    
    # 3. Add Custom DNA Insights
    # Eden Gardens historically favors chasing. 
    # Shreyas Iyer (PBKS Captain) has "Insider DNA" on KKR.
    
    print("\n" + "="*40)
    print("      SOVEREIGN VERDICT: MATCH 12")
    print("="*40)
    print(f"🏟️  Venue: Eden Gardens, Kolkata")
    print(f"⚔️  Matchup: KKR vs PBKS")
    print("-" * 40)
    
    # Simple logic to determine favourite for print
    fav = "PBKS" if prediction['win_probability'] < 0.5 else "KKR" # Note: win_p is usually for bat_team (KKR at home)
    # Actually, let's assume team1 (KKR) is the reference
    
    print(f"🏆 Win Probability (KKR):  {prediction['win_probability']*100:.1f}%")
    print(f"🏆 Win Probability (PBKS): {100 - prediction['win_probability']*100:.1f}%")
    print(f"⭐ Engine Agreement:      {prediction['ensemble_agreement']*100:.1f}%")
    print("-" * 40)
    print("🧠 Engine Forensics:")
    for trace in prediction['forensic_trace']:
        print(f"  - {trace}")
    print("="*40)

if __name__ == "__main__":
    run_prediction()
