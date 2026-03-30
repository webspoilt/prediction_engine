import pytest
import numpy as np
import pandas as pd
from backend.ml_engine.hybrid_model import CricsheetNormalizer

@pytest.fixture
def sample_match_data():
    """Mock match data for normalizer tests - using exact date format"""
    return pd.DataFrame([
        {
            'match_id': '2024-05-01',
            'inning': 1,
            'over': 0.1,
            'batsman': 'Virat Kohli',
            'bowler': 'Jasprit Bumrah',
            'runs': 4,
            'wicket': False,
            'batting_team': 'RCB',
            'bowling_team': 'MI'
        },
        {
            'match_id': '2024-05-01',
            'inning': 1,
            'over': 0.2,
            'batsman': 'Virat Kohli',
            'bowler': 'Jasprit Bumrah',
            'runs': 0,
            'wicket': True,
            'batting_team': 'RCB',
            'bowling_team': 'MI'
        }
    ])

def test_normalizer_feature_count(normalizer, sample_match_data):
    """
    Ensure the standardized feature set always contains exactly 22 columns
    regardless of input data richness.
    """
    df = normalizer.create_match_features(sample_match_data)
    
    # Check numeric feature count against defined STATIC_FEATURE_COLS
    numeric_cols = df[normalizer.STATIC_FEATURE_COLS]
    assert numeric_cols.shape[1] == 22
    assert 'bat_elo' in numeric_cols.columns
    assert 'temp' in numeric_cols.columns

def test_inference_shape_matching(ensemble):
    """
    Verify high-fidelity shape matching between static XGBoost vectors
    and LSTM temporal sequences.
    """
    # 22 features, 1 row
    X_static = np.zeros((1, 22))
    
    # 18 balls, 3 features [runs, wickets, overs]
    X_seq = np.zeros((1, 18, 3))
    
    # This shouldn't crash if scalers are correctly initialized
    prediction = ensemble.predict(X_static, X_seq)
    
    assert 'win_probability' in prediction
    assert isinstance(prediction['win_probability'], (float, np.float32))

def test_elo_differential_calculation(normalizer, sample_match_data):
    """
    Verify that ELO differentials are correctly calculated from mock lookup.
    """
    df = normalizer.create_match_features(sample_match_data)
    latest = df.iloc[-1]
    
    # RCB (1550) vs MI (1600) -> elo_diff = -50.0
    assert float(latest['bat_elo']) == 1550.0
    assert float(latest['bowl_elo']) == 1600.0
    assert float(latest['elo_diff']) == -50.0
