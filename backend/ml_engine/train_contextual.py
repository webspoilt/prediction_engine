import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import torch

def train_context_model(csv_path="e:\\IDEAS\\ipl prediction engine\\ipl_2008_2024_complete.csv"):
    print("🧠 Initializing Pre-Match Contextual AI Training...")
    
    # 1. Load the Historical CSV
    if not os.path.exists(csv_path):
        print(f"❌ Could not find CSV at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} historical innings from 2008-2024.")
    
    # 2. Feature Engineering
    # We want to know if the CURRENT batting team won the match.
    df = df.dropna(subset=['winner', 'batting_team', 'bowling_team', 'venue', 'toss_decision'])
    
    df['is_batting_team_winner'] = (df['batting_team'] == df['winner']).astype(int)
    df['is_toss_winner'] = (df['toss_winner'] == df['batting_team']).astype(int)
    df['is_toss_bat'] = (df['toss_decision'] == 'bat').astype(int)
    
    # Filter out any weird D/L method matches to ensure pure math
    df = df[(df['method'] != 'D/L') | (df['method'].isna())]

    # 3. Categorical Encoding
    print("🔠 Encoding Strings (Teams & Stadiums) into AI Math...")
    from sklearn.preprocessing import LabelEncoder
    
    encoder_dict = {}
    features_to_encode = ['batting_team', 'bowling_team', 'venue']
    
    # Standardize team names slightly (e.g. Pune Warriors, etc.)
    for col in features_to_encode:
        le = LabelEncoder()
        df[col] = df[col].astype(str).str.strip()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le
        
    # 4. Define our Training Features
    # Feature Vector: [batting_team_id, bowling_team_id, venue_id, is_toss_winner, is_toss_bat, innings]
    X_cols = ['batting_team', 'bowling_team', 'venue', 'is_toss_winner', 'is_toss_bat', 'innings']
    y_col = 'is_batting_team_winner'
    
    X = df[X_cols].values
    y = df[y_col].values
    
    print(f"🎯 Training XGBoost Context Predictor on {len(X)} specific scenarios...")
    
    # 5. Model Configuration (Using your GPU if available!)
    tree_method = "hist"
    device = "cpu"
    if torch.cuda.is_available():
        tree_method = "hist"
        device = "cuda"
        print("🚀 GPU Configured for Context Model")
        
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 5, # Relatively shallow, we don't want to overfit historical bias
        'tree_method': tree_method,
        'device': device,
        'n_estimators': 200
    }
    
    model = xgb.XGBClassifier(**params)
    
    # 6. Train!
    model.fit(X, y)
    
    print("✅ Context Training Complete!")
    
    # 7. Save Assets for the Live Engine
    os.makedirs('models/hybrid_ensemble', exist_ok=True)
    
    # Save Model
    model_path = 'models/hybrid_ensemble/context_xgb.json'
    model.save_model(model_path)
    
    # Save Encoders so the Live Server can map "Chennai Super Kings" to its exact integer ID during a match
    encoders_path = 'models/hybrid_ensemble/context_encoders.pkl'
    joblib.dump(encoder_dict, encoders_path)
    
    print(f"💾 Model Saved to: {model_path}")
    print(f"💾 Encoders Saved to: {encoders_path}")
    print("🚀 You are now ready to fuse this into the Live Engine.")
    
if __name__ == "__main__":
    train_context_model()
