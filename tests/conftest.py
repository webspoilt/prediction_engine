import pytest
import pandas as pd
import numpy as np
import json
import torch
from unittest.mock import Mock, MagicMock
from backend.ml_engine.hybrid_model import CricsheetNormalizer, HybridEnsemble, MomentumLSTM

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
    """HybridEnsemble instance with mock XGBoost and LSTM models"""
    e = HybridEnsemble()
    # Fake scaling params for tests
    e.static_scaler.mean_ = np.zeros(22)
    e.static_scaler.scale_ = np.ones(22)

    # Mock XGBoost model — returns a 2-column probability array
    mock_xgb = Mock()
    mock_xgb.predict_proba = Mock(return_value=np.array([[0.35, 0.65]]))
    e.xgb_model = mock_xgb

    # Mock LSTM model — returns a (1, 32) tensor
    mock_lstm = MagicMock(spec=MomentumLSTM)
    mock_lstm.eval = Mock()
    mock_lstm.return_value = torch.tensor([[0.6] + [0.0] * 31])
    e.lstm_model = mock_lstm

    return e
