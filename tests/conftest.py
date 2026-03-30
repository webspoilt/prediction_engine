import pytest
import pandas as pd
import numpy as np
import json
from backend.ml_engine.hybrid_model import CricsheetNormalizer, HybridEnsemble

@pytest.fixture
def sample_match_data():
    """Mock match data for normalizer tests"""
    return pd.DataFrame([
        {
            'match_id': '2024-05-01',
            'inning': 1,
            'over': 0.1,
            'batsman': 'Virat Kohli',
            'bowler': 'Jasprit Bumrah',
            'runs': 4,
            'wicket': False,
            'batting_team': 'Royal Challengers Bengaluru',
            'bowling_team': 'Mumbai Indians'
        },
        {
            'match_id': '2024-05-01',
            'inning': 1,
            'over': 0.2,
            'batsman': 'Virat Kohli',
            'bowler': 'Jasprit Bumrah',
            'runs': 0,
            'wicket': True,
            'batting_team': 'Royal Challengers Bengaluru',
            'bowling_team': 'Mumbai Indians'
        }
    ])

@pytest.fixture
def normalizer():
    """Instantiated normalizer without loading large files"""
    n = CricsheetNormalizer()
    # Mock lookup tables to avoid file I/O
    n.weather_lookup = {
        '2024-05-01': {'temp_mean': 30.5, 'humidity_mean': 65.0, 'dew_point_mean': 22.0}
    }
    n.elo_lookup = {
        '2024-05-01': {
            'RCB': 1550.0,
            'MI': 1600.0
        }
    }
    return n

@pytest.fixture
def ensemble():
    """HybridEnsemble instance with mock weights"""
    e = HybridEnsemble()
    # Fake scaling params for tests
    e.static_scaler.mean_ = np.zeros(22)
    e.static_scaler.scale_ = np.ones(22)
    return e
