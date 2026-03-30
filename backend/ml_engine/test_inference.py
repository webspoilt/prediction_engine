import numpy as np
import pandas as pd
import json
import torch
import sys
import os

# Ensure Python can find the local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.hybrid_model import HybridEnsemble, ModelConfig

def test_mock_inference():
    print("🤖 Initializing AI Brain (Loading from 'models/hybrid_ensemble')...")
    
    ensemble = HybridEnsemble()
    
    try:
        ensemble.load_models('models/hybrid_ensemble')
        print("✅ Models and scalers successfully loaded into memory!\n")
    except Exception as e:
        print(f"❌ Error loading models. Are you running this from the right folder? Error: {e}")
        return

    static_shape = 22
    
    print(f"📊 The XGBoost model expects exactly {static_shape} static variables.")
    print("📈 Creating a fake 'Nail-Biter' match scenario: 175 runs, 4 wickets down in the 18th over...\n")
    
    # 1. Mock Static Match State 
    raw_static = np.zeros((1, static_shape))
    # Standard Features
    raw_static[0, 0] = 1.0     # Inning
    raw_static[0, 1] = 18.0    # Overs completed
    raw_static[0, 2] = 175.0   # Total Runs
    raw_static[0, 3] = 4.0     # Total Wickets
    raw_static[0, 4] = 9.72    # CRR (175/18)
    raw_static[0, 5] = 45.0    # Runs last 6 overs 
    raw_static[0, 6] = 2.0     # Wickets last 6 overs
    raw_static[0, 7] = 0.15    # Boundary rate
    raw_static[0, 8] = 0.35    # Dot pressure
    raw_static[0, 9] = 12.0    # Balls remaining (120 - 108)
    
    # Player Stats (Mocks)
    raw_static[0, 10] = 145.0  # bat_sr
    raw_static[0, 11] = 35.0   # bat_avg
    raw_static[0, 12] = 12.0   # bat_bp
    raw_static[0, 13] = 8.2    # bowl_econ
    raw_static[0, 14] = 24.0   # bowl_sr
    raw_static[0, 15] = 28.0   # bowl_avg
    
    # Weather context
    raw_static[0, 16] = 28.5   # temp
    raw_static[0, 17] = 65.0   # humidity
    raw_static[0, 18] = 21.0   # dew
    
    # ELO ratings
    raw_static[0, 19] = 1580.0 # bat_elo
    raw_static[0, 20] = 1520.0 # bowl_elo
    raw_static[0, 21] = 60.0   # elo_diff
    
    # 2. Mock 18-Ball LSTM Sequence [runs, wicket, over]
    # Representing the last 3 overs of deep temporal bowling momentum
    raw_sequence = []
    current_over = 15.1
    for i in range(18):
        # Mostly singles and dots, with 1 nasty wicket
        runs = np.random.choice([0, 1, 2, 4])
        wicket = 1 if i == 10 else 0
        raw_sequence.append([runs, wicket, current_over])
        
        # Advance ball count mathematically
        current_over = round(current_over + 0.1, 1)
        if round(current_over % 1, 1) == 0.7: 
            current_over = round(current_over + 0.4, 1)  # Roll over to next over (15.6 -> 16.1)
            
    seq_np = np.array(raw_sequence).reshape(1, 18, 3)

    print("🧠 Forcing the Hybrid Pipeline to analyze the scenario...")
    prediction = ensemble.predict(
        static_features=raw_static,
        sequence_features=seq_np,
        return_confidence=True
    )
    
    print("\n================ PREDICTION RESULTS ================")
    print(f"🏆 Overall Win Probability:  {prediction['win_probability']*100:.2f}%")
    print(f"🔍 XGBoost Brain (Stats):  {prediction['xgb_probability']*100:.2f}%")
    print(f"🧠 LSTM Brain (Momentum):  {prediction['lstm_probability']*100:.2f}%")
    print(f"⭐ Engine Confidence:        {prediction['confidence']*100:.2f}%")
    print("====================================================")

if __name__ == "__main__":
    test_mock_inference()
