"""
TASK 3: HYBRID ML MODEL (THE "BRAIN")
IPL Win Probability Prediction Engine - Ensemble Model Architecture

This module implements a hybrid ensemble combining:
1. XGBoost Layer: Static features (venue, toss, XIs, basic stats)
2. LSTM/GRU Layer: Dynamic temporal features (last 18 balls, momentum)

Training data: Cricsheet.org ball-by-ball data (normalized)
Inference: <50ms on GTX 1650 Ti
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import redis
import time
import os
from huggingface_hub import hf_hub_download, HfApi, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the hybrid model"""
    # XGBoost params
    xgb_params: Dict = None
    
    # LSTM params
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    sequence_length: int = 18  # Last 18 balls
    # Transformer params (New)
    use_transformer: bool = True
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 3
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    
    # Training params
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }


class CricsheetNormalizer:
    """
    Normalizes Cricsheet.org ball-by-ball data for model training.
    Handles various data formats and creates standardized features.
    """
    
    # Venue mappings for standardization
    VENUE_MAPPING = {
        'M Chinnaswamy Stadium': 'Bangalore',
        'M.Chinnaswamy Stadium': 'Bangalore',
        'MA Chidambaram Stadium': 'Chennai',
        'Wankhede Stadium': 'Mumbai',
        'Eden Gardens': 'Kolkata',
        'Arun Jaitley Stadium': 'Delhi',
        'Feroz Shah Kotla': 'Delhi',
        'Rajiv Gandhi International Stadium': 'Hyderabad',
        'Sawai Mansingh Stadium': 'Jaipur',
        'Punjab Cricket Association Stadium': 'Mohali',
        'Narendra Modi Stadium': 'Ahmedabad',
    }
    
    # Team name standardization
    TEAM_MAPPING = {
        'Royal Challengers Bangalore': 'RCB',
        'Chennai Super Kings': 'CSK',
        'Mumbai Indians': 'MI',
        'Kolkata Knight Riders': 'KKR',
        'Delhi Capitals': 'DC',
        'Delhi Daredevils': 'DC',
        'Sunrisers Hyderabad': 'SRH',
        'Rajasthan Royals': 'RR',
        'Punjab Kings': 'PBKS',
        'Kings XI Punjab': 'PBKS',
        'Gujarat Titans': 'GT',
        'Lucknow Super Giants': 'LSG',
    }
    
    # Standard list of numeric features for model input
    STATIC_FEATURE_COLS = [
        'inning', 'over', 'total_runs', 'total_wickets', 'crr', 
        'runs_last_6', 'wickets_last_6', 'boundary_rate', 'dot_pressure', 
        'balls_remaining', 'bat_sr', 'bat_avg', 'bat_bp', 'bowl_econ', 
        'bowl_sr', 'bowl_avg', 'temp', 'humidity', 'dew', 
        'bat_elo', 'bowl_elo', 'elo_diff'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.venue_encoder = LabelEncoder()
        self.player_stats = None
        self.weather_lookup = {}
        self.elo_lookup = {}
        
    def load_enhanced_features(self, weather_path='dataset/enhanced/weather_dataset.csv', elo_path='dataset/enhanced/team_elo.csv'):
        """Load external context datasets (Weather and Team Form ELOs)"""
        import os, pandas as pd
        if os.path.exists(weather_path):
            try:
                wdf = pd.read_csv(weather_path)
                for _, r in wdf.iterrows():
                    self.weather_lookup[r['api_date']] = r.to_dict()
                logger.info(f"Loaded {len(self.weather_lookup)} weather entries")
            except Exception as e:
                logger.error(f"Failed to load weather data: {e}")
                
        if os.path.exists(elo_path):
            try:
                edf = pd.read_csv(elo_path)
                for _, r in edf.iterrows():
                    d = str(r['parsed_date'])
                    if d not in self.elo_lookup:
                        self.elo_lookup[d] = {}
                    t1, t2 = self.normalize_team(r['team1']), self.normalize_team(r['team2'])
                    self.elo_lookup[d][t1] = r['team1_elo_pre']
                    self.elo_lookup[d][t2] = r['team2_elo_pre']
                logger.info(f"Loaded ELO data for {len(self.elo_lookup)} match dates")
            except Exception as e:
                logger.error(f"Failed to load ELO data: {e}")

    def load_player_stats(self, db_path='backend/ml_engine/player_stats_db.json'):
        """Load offline player stats lookup table"""
        import os, json
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                self.player_stats = json.load(f)
        else:
            self.player_stats = {
                "batsmen": {}, "bowlers": {},
                "global_baselines": {
                    "bat_sr": 120.0, "bat_avg": 20.0, "bat_bound_pct": 10.0,
                    "bowl_econ": 8.0, "bowl_sr": 24.0, "bowl_avg": 30.0
                }
            }
        
    def load_cricsheet_data(self, filepath: str) -> pd.DataFrame:
        """Load and parse Cricsheet JSON data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Determine teams for inferring bowling team
        teams = data.get('info', {}).get('teams', [])
        
        # Extract ball-by-ball data
        balls = []
        
        for inning_idx, inning in enumerate(data.get('innings', []), 1):
            inning_team = inning.get('team', '')
            
            # Infer bowling team as the other team in the match
            if len(teams) > 1:
                bowling_team = teams[0] if teams[1] == inning_team else teams[1]
            else:
                bowling_team = 'Unknown'
            
            for over_data in inning.get('overs', []):
                over_num = over_data.get('over', 0)
                
                for ball_idx, delivery in enumerate(over_data.get('deliveries', [])):
                    ball = {
                        'match_id': data.get('info', {}).get('dates', [''])[0],
                        'inning': inning_idx,
                        'over': over_num + ball_idx / 6,
                        'batsman': delivery.get('batter', ''),
                        'bowler': delivery.get('bowler', ''),
                        'runs': delivery.get('runs', {}).get('total', 0),
                        'batter_runs': delivery.get('runs', {}).get('batter', 0),
                        'extras': delivery.get('runs', {}).get('extras', 0),
                        'wicket': 'wickets' in delivery,
                        'wicket_type': delivery.get('wickets', [{}])[0].get('kind') if 'wickets' in delivery else None,
                        'batting_team': inning_team,
                        'bowling_team': bowling_team,
                    }
                    balls.append(ball)
        
        return pd.DataFrame(balls)
    
    def normalize_venue(self, venue: str) -> str:
        """Standardize venue names"""
        venue = venue.strip()
        return self.VENUE_MAPPING.get(venue, venue)
    
    def normalize_team(self, team: str) -> str:
        """Standardize team names"""
        team = team.strip()
        return self.TEAM_MAPPING.get(team, team)
    
    def create_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create match-level features from ball-by-ball data"""
        features = []
        
        for match_id in df['match_id'].unique():
            match_df = df[df['match_id'] == match_id].sort_values('over')
            
            # Process each ball with historical context
            for idx, row in match_df.iterrows():
                # Get context up to this ball
                context = match_df[match_df['over'] <= row['over']]
                
                # Basic stats
                total_runs = context['runs'].sum()
                total_wickets = context['wicket'].sum()
                balls_faced = len(context)
                overs = row['over']
                
                # Run rates
                crr = total_runs / max(overs, 0.1)
                
                # Recent performance (last 6 balls)
                recent = match_df[
                    (match_df['over'] > row['over'] - 1) & 
                    (match_df['over'] <= row['over'])
                ]
                runs_last_6 = recent['runs'].sum()
                wickets_last_6 = recent['wicket'].sum()
                
                # Boundary analysis
                boundaries = context[context['runs'] >= 4]
                boundary_rate = len(boundaries) / max(len(context), 1)
                
                # Dot ball pressure
                dot_balls = context[context['runs'] == 0]
                dot_pressure = len(dot_balls) / max(len(context), 1)
                
                # Fetch player stats dynamically
                batter = row['batsman']
                bowler = row['bowler']
                
                if self.player_stats:
                    bat_stats = self.player_stats['batsmen'].get(batter, {})
                    bat_sr = bat_stats.get('strike_rate', self.player_stats['global_baselines']['bat_sr'])
                    bat_avg = bat_stats.get('average', self.player_stats['global_baselines']['bat_avg'])
                    bat_bp = bat_stats.get('boundary_pct', self.player_stats['global_baselines']['bat_bound_pct'])
                    
                    bowl_stats = self.player_stats['bowlers'].get(bowler, {})
                    bowl_econ = bowl_stats.get('economy', self.player_stats['global_baselines']['bowl_econ'])
                    bowl_sr = bowl_stats.get('strike_rate', self.player_stats['global_baselines']['bowl_sr'])
                    bowl_avg = bowl_stats.get('average', self.player_stats['global_baselines']['bowl_avg'])
                else:
                    bat_sr, bat_avg, bat_bp = 120.0, 20.0, 10.0
                    bowl_econ, bowl_sr, bowl_avg = 8.0, 24.0, 30.0

                # Extract Enhanced Features Context
                # Cricsheet match_id is typically the date string `YYYY-MM-DD`
                m_date = str(match_id)
                weather = self.weather_lookup.get(m_date, {})
                temp = weather.get('temp_mean', 25.0)  # Default values
                humidity = weather.get('humidity_mean', 60.0)
                dew = weather.get('dew_point_mean', 18.0)
                
                elo_date_map = self.elo_lookup.get(m_date, {})
                # Some files might not have batting_team/bowling_team explicitly in row
                batting_team = self.normalize_team(row.get('batting_team', ''))
                bowling_team = self.normalize_team(row.get('bowling_team', ''))
                bat_elo = elo_date_map.get(batting_team, 1500.0)
                bowl_elo = elo_date_map.get(bowling_team, 1500.0)

                feature = {
                    'match_id': match_id,
                    'inning': row['inning'],
                    'over': overs,
                    'total_runs': total_runs,
                    'total_wickets': total_wickets,
                    'crr': crr,
                    'runs_last_6': runs_last_6,
                    'wickets_last_6': wickets_last_6,
                    'boundary_rate': boundary_rate,
                    'dot_pressure': dot_pressure,
                    'balls_remaining': 120 - balls_faced if row['inning'] == 1 else None,
                    'bat_sr': bat_sr,
                    'bat_avg': bat_avg,
                    'bat_bp': bat_bp,
                    'bowl_econ': bowl_econ,
                    'bowl_sr': bowl_sr,
                    'bowl_avg': bowl_avg,
                    'temp': temp,
                    'humidity': humidity,
                    'dew': dew,
                    'bat_elo': bat_elo,
                    'bowl_elo': bowl_elo,
                    'elo_diff': bat_elo - bowl_elo
                }
                features.append(feature)
        
        return pd.DataFrame(features)
    
    def prepare_training_data(self, 
                              cricsheet_files: List[str],
                              match_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete training dataset from Cricsheet files.
        
        Returns:
            static_features: DataFrame for XGBoost
            sequence_features: DataFrame for LSTM
        """
        all_balls = []
        
        for filepath in cricsheet_files:
            try:
                df = self.load_cricsheet_data(filepath)
                all_balls.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        combined_df = pd.concat(all_balls, ignore_index=True)
        
        # Create static features
        static_features = self.create_match_features(combined_df)
        
        # Create sequence features (last 18 balls for each point)
        sequence_features = self._create_sequences(combined_df, sequence_length=18)
        
        return static_features, sequence_features
    
    def _create_sequences(self, df: pd.DataFrame, sequence_length: int = 18) -> pd.DataFrame:
        """Create time-series sequences for LSTM"""
        sequences = []
        
        for match_id in df['match_id'].unique():
            match_df = df[df['match_id'] == match_id].sort_values('over')
            
            for i in range(len(match_df)):
                end_idx = i + 1
                start_idx = max(0, end_idx - sequence_length)
                
                seq = match_df.iloc[start_idx:end_idx]
                
                # Pad if needed
                if len(seq) < sequence_length:
                    padding = pd.DataFrame([seq.iloc[0]] * (sequence_length - len(seq)))
                    seq = pd.concat([padding, seq], ignore_index=True)
                
                sequence_data = {
                    'match_id': match_id,
                    'position': i,
                    'sequence': seq[['runs', 'wicket', 'over']].values.tolist()
                }
                sequences.append(sequence_data)
        
        return pd.DataFrame(sequences)


class MomentumLSTM(nn.Module):
    """
    LSTM/GRU model for capturing momentum and time-series dependencies.
    Processes sequences of last 18 balls to extract temporal patterns.
    """
    
    def __init__(self, 
                 input_size: int = 3,  # runs, wicket, over
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 output_size: int = 32):  # Feature vector size
        super(MomentumLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Feature vector of shape (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Final projection
        output = self.fc(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer to inject sequence order info.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Attention-based Transformer Encoder for IPL momentum tracking.
    """
    def __init__(self, 
                 input_dim: int, 
                 d_model: int = 128, 
                 nhead: int = 8, 
                 num_layers: int = 3, 
                 dim_feedforward: int = 256, 
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Dense head to produce 32-dim latent embedding (matching LSTM output size)
        self.fc = nn.Linear(d_model, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)            # (batch, seq_len, d_model)
        x = self.pos_encoder(x)           # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)   # (batch, seq_len, d_model)
        
        # Use average pooling across sequence as alternative to last timestep
        x = x.mean(dim=1)                 # (batch, d_model)
        return self.fc(x)                 # (batch, 32)


class HybridEnsemble:
    """
    Hybrid Ensemble Model combining XGBoost and LSTM.
    XGBoost: Static features (venue, toss, team strength, basic stats)
    LSTM: Dynamic features (last 18 balls, momentum patterns)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lstm_model: Optional[MomentumLSTM] = None
        self.transformer_model: Optional[TransformerModel] = None
        
        # Scalers
        self.static_scaler = StandardScaler()
        self.lstm_scaler = StandardScaler()
        
        # Feature importance tracking
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"Hybrid Ensemble initialized on {self.device}")
        
    def build_xgb_model(self) -> xgb.XGBClassifier:
        """Build XGBoost classifier for static features"""
        return xgb.XGBClassifier(**self.config.xgb_params)
    
    def build_lstm_model(self) -> MomentumLSTM:
        """Build LSTM model for sequence features"""
        model = MomentumLSTM(
            input_size=3,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout,
            output_size=32
        )
        return model.to(self.device)
    
    def build_transformer_model(self) -> TransformerModel:
        """Build Transformer model for sequence features"""
        model = TransformerModel(
            input_dim=3,
            d_model=self.config.transformer_d_model,
            nhead=self.config.transformer_nhead,
            num_layers=self.config.transformer_num_layers,
            dim_feedforward=self.config.transformer_dim_feedforward,
            dropout=self.config.transformer_dropout
        )
        return model.to(self.device)
    
    def train_xgb(self, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        self.xgb_model = self.build_xgb_model()
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_patience,
            verbose=False
        )
        
        # Store feature importance
        importance = self.xgb_model.feature_importances_
        self.feature_importance['xgb'] = float(np.mean(importance))
        
        logger.info(f"XGBoost training complete. Best iteration: {self.xgb_model.best_iteration}")
        
    def train_lstm(self,
                   train_sequences: np.ndarray,
                   train_labels: np.ndarray,
                   val_sequences: Optional[np.ndarray] = None,
                   val_labels: Optional[np.ndarray] = None):
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        self.lstm_model = self.build_lstm_model()
        
        # Create data loaders
        train_dataset = SequenceDataset(train_sequences, train_labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.lstm_model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.lstm_model.train()
            train_loss = 0.0
            
            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.lstm_model(batch_seq)
                
                # Simple classifier head (would be separate in production)
                predictions = torch.sigmoid(features[:, 0])
                
                loss = criterion(predictions, batch_labels.float())
                loss.backward()
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            if val_sequences is not None and val_labels is not None:
                val_loss = self._validate_lstm(val_sequences, val_labels, criterion)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.lstm_model.state_dict(), 'best_lstm.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                               f"Val Loss = {val_loss:.4f}")
        
        # Load best model
        self.lstm_model.load_state_dict(torch.load('best_lstm.pth'))
        logger.info("LSTM training complete")
        
    def train_transformer(self,
                          train_sequences: np.ndarray,
                          train_labels: np.ndarray,
                          val_sequences: Optional[np.ndarray] = None,
                          val_labels: Optional[np.ndarray] = None):
        """Train Transformer model"""
        logger.info("Training Transformer model...")
        
        self.transformer_model = self.build_transformer_model()
        
        train_dataset = SequenceDataset(train_sequences, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=self.config.learning_rate)
        
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            self.transformer_model.train()
            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                features = self.transformer_model(batch_seq)
                predictions = features[:, 0]
                loss = criterion(predictions, batch_labels.float())
                loss.backward()
                optimizer.step()
            
            if val_sequences is not None:
                val_loss = self._validate_transformer(val_sequences, val_labels, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.transformer_model.state_dict(), 'best_transformer.pth')
                    
        self.transformer_model.load_state_dict(torch.load('best_transformer.pth'))
        logger.info("Transformer training complete")

    def _validate_transformer(self, 
                              val_sequences: np.ndarray, 
                              val_labels: np.ndarray, 
                              criterion) -> float:
        self.transformer_model.eval()
        val_loader = DataLoader(SequenceDataset(val_sequences, val_labels), batch_size=self.config.batch_size)
        total_loss = 0.0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(self.device), labels.to(self.device)
                features = self.transformer_model(seq)
                loss = criterion(features[:, 0], labels.float())
                total_loss += loss.item()
        return total_loss / len(val_loader)
        
    def _validate_lstm(self, 
                       val_sequences: np.ndarray,
                       val_labels: np.ndarray,
                       criterion) -> float:
        """Validate LSTM model"""
        self.lstm_model.eval()
        
        val_dataset = SequenceDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        total_loss = 0.0
        with torch.no_grad():
            for batch_seq, batch_labels in val_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                features = self.lstm_model(batch_seq)
                predictions = torch.sigmoid(features[:, 0])
                
                loss = criterion(predictions, batch_labels.float())
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, 
                static_features: np.ndarray,
                sequence_features: np.ndarray,
                return_confidence: bool = True) -> Dict:
        """
        Make prediction using ensemble.
        
        Args:
            static_features: XGBoost input features
            sequence_features: LSTM input sequences
            return_confidence: Whether to return confidence scores
        
        Returns:
            Dictionary with prediction and confidence
        """
        # XGBoost prediction
        xgb_prob = self.xgb_model.predict_proba(static_features)[:, 1]
        
        # LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_features).to(self.device)
            lstm_features = self.lstm_model(seq_tensor)
            lstm_prob = torch.sigmoid(lstm_features[:, 0]).cpu().numpy()
        
        # Ensemble (weighted average based on validation performance)
        # XGBoost typically more reliable for cricket
        xgb_weight = 0.6
        lstm_weight = 0.4
        
        ensemble_prob = xgb_weight * xgb_prob + lstm_weight * lstm_prob
        
        result = {
            'win_probability': float(ensemble_prob[0]),
            'xgb_probability': float(xgb_prob[0]),
            'lstm_probability': float(lstm_prob[0]),
        }
        
        if return_confidence:
            # Confidence based on model agreement
            agreement = 1 - abs(xgb_prob[0] - lstm_prob[0])
            result['confidence'] = float(agreement)
            result['uncertainty'] = float(1 - agreement)
        
        return result
    
    def save_models(self, path_prefix: str):
        """Save all models and scalers"""
        # Save XGBoost
        self.xgb_model.save_model(f"{path_prefix}_xgb.json")
        
        # Save LSTM
        torch.save(self.lstm_model.state_dict(), f"{path_prefix}_lstm.pth")
        
        # Save scalers
        with open(f"{path_prefix}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'static': self.static_scaler,
                'lstm': self.lstm_scaler
            }, f)
        
        logger.info(f"Models saved to {path_prefix}")

    def save_to_hub(self, repo_id: str, path_prefix: str, commit_message: str = "Update IPL models"):
        """Save models locally first, then upload to Hugging Face Hub"""
        self.save_models(path_prefix)
        
        api = HfApi()
        files_to_upload = [
            f"{path_prefix}_xgb.json",
            f"{path_prefix}_lstm.pth",
            f"{path_prefix}_scalers.pkl"
        ]
        
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_id,
                    commit_message=commit_message
                )
        logger.info(f"✅ Models successfully pushed to Hub: {repo_id}")

    def load_from_hub(self, repo_id: str, path_prefix: str):
        """Download models from Hugging Face Hub then load them"""
        files_to_download = [
            f"{os.path.basename(path_prefix)}_xgb.json",
            f"{os.path.basename(path_prefix)}_lstm.pth",
            f"{os.path.basename(path_prefix)}_scalers.pkl"
        ]
        
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        for file_name in files_to_download:
            # We must specify the folder path in the repository
            repo_file_path = f"models/{file_name}"
            local_path = hf_hub_download(repo_id=repo_id, filename=repo_file_path)
            # Copy to our expected path_prefix location
            target_name = f"{path_prefix}_{file_name.split('_')[-1]}"
            import shutil
            shutil.copy2(local_path, target_name)
            
        self.load_models(path_prefix)
        logger.info(f"✅ Models successfully loaded from Hub: {repo_id}")
        
    def load_models(self, path_prefix: str):
        """Load all models and scalers"""
        # Load XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f"{path_prefix}_xgb.json")
        
        # Load LSTM
        self.lstm_model = self.build_lstm_model()
        self.lstm_model.load_state_dict(torch.load(f"{path_prefix}_lstm.pth", map_location=self.device))
        self.lstm_model.eval()
        
        # Load scalers
        with open(f"{path_prefix}_scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
            self.static_scaler = scalers['static']
            self.lstm_scaler = scalers['lstm']
        
        logger.info(f"Models loaded from {path_prefix}")


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class RealTimePredictor:
    """
    Real-time prediction interface for live matches.
    Connects to Redis for live data and produces predictions every 2-5 minutes.
    """
    
    def __init__(self, model_path: str = 'models/hybrid_ensemble', repo_id: Optional[str] = None):
        self.model = HybridEnsemble()
        if repo_id:
            try:
                self.model.load_from_hub(repo_id, model_path)
            except Exception as e:
                logger.warning(f"Failed to load from Hub, trying local: {e}")
                self.model.load_models(model_path)
        else:
            self.model.load_models(model_path)
        self.redis_client = redis.Redis(decode_responses=True)
        self.normalizer = CricsheetNormalizer()
        self.normalizer.load_enhanced_features()
        
        # Performance tracking
        self.inference_times = []
        
    def predict_live_match(self, match_id: str) -> Dict:
        """
        Generate prediction for a live match.
        
        Args:
            match_id: Unique match identifier
        
        Returns:
            Prediction result with win probability
        """
        start_time = time.time()
        
        # Fetch full match data thus far for accurate static feature aggregation
        ball_stream = f"ipl:balls:{match_id}"
        all_balls = self.redis_client.xrevrange(ball_stream)
        
        if not all_balls:
            return {'error': 'No live data available'}
        
        # Convert to DataFrame
        ball_data = [b[1] for b in reversed(all_balls)]
        df = pd.DataFrame(ball_data)
        
        # Ensure datatypes (Redis stores as strings)
        for col in ['inning', 'over', 'runs', 'wicket']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # Inject match_id for the normalizer
        df['match_id'] = match_id
        
        # Ensure required columns for normalizer exist
        required_cols = ['batsman', 'bowler', 'batting_team', 'bowling_team']
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
                
        # Extract features
        static_features = self._extract_static_features(df)
        sequence_features = self._extract_sequence_features(df)
        
        # Make prediction
        result = self.model.predict(
            static_features.reshape(1, -1),
            sequence_features.reshape(1, 18, 3)
        )
        
        # Add metadata
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        result['match_id'] = match_id
        result['inference_time_ms'] = inference_time
        result['timestamp'] = time.time()
        result['balls_analyzed'] = len(ball_data)
        
        # Publish prediction
        self._publish_prediction(match_id, result)
        
        return result
    
    def _extract_static_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract static features dynamically matching training structure"""
        # We leverage the identical normalization pipeline used in training
        # to guarantee the shape perfectly matches our massive multi-dimension frame.
        features_df = self.normalizer.create_match_features(df)
        
        # Target the final current state (the latest ball)
        current_state = features_df.tail(1)
        
        # Isolate exactly the numeric columns XGBoost was trained on
        # Standardized to our STATIC_FEATURE_COLS
        X_live = current_state[self.normalizer.STATIC_FEATURE_COLS].fillna(0).values
        
        # Scale
        scaled_features = self.model.static_scaler.transform(X_live)
        
        return scaled_features.flatten()
    
    def _extract_sequence_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sequence features for LSTM"""
        # Pad or truncate to 18 balls
        seq_len = 18
        
        if len(df) < seq_len:
            # Pad with zeros
            padding = pd.DataFrame(0, index=range(seq_len - len(df)), columns=df.columns)
            df = pd.concat([padding, df], ignore_index=True)
        else:
            df = df.tail(seq_len)
        
        # Extract features (runs, wicket, over)
        features = df[['runs', 'wicket', 'over']].values
        
        return features
    
    def _publish_prediction(self, match_id: str, prediction: Dict):
        """Publish prediction to Redis"""
        stream_key = f"ipl:predictions:{match_id}"
        self.redis_client.xadd(stream_key, prediction, maxlen=100)
        
        # Publish to pub/sub
        self.redis_client.publish(
            f"ipl:prediction:{match_id}",
            json.dumps(prediction)
        )
    
    def get_performance_stats(self) -> Dict:
        """Get prediction performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_inference_ms': float(np.mean(times)),
            'p95_inference_ms': float(np.percentile(times, 95)),
            'max_inference_ms': float(np.max(times)),
            'total_predictions': len(times)
        }


# ==================== USAGE EXAMPLE ====================

def train_example():
    """Example: Train the hybrid model on Cricsheet data"""
    
    # Initialize
    normalizer = CricsheetNormalizer()
    config = ModelConfig()
    ensemble = HybridEnsemble(config)
    
    # Load and prepare data
    cricsheet_files = [
        'data/cricsheet/ipl_2023.json',
        'data/cricsheet/ipl_2022.json',
        # Add more files
    ]
    
    static_df, sequence_df = normalizer.prepare_training_data(
        cricsheet_files,
        match_metadata=pd.DataFrame()  # Would contain match outcomes
    )
    
    # Prepare labels (match outcomes)
    # This would come from match metadata
    labels = np.random.randint(0, 2, len(static_df))  # Placeholder
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        static_df.select_dtypes(include=[np.number]).values,
        labels,
        test_size=0.2,
        random_state=42
    )
    
    # Train models
    ensemble.train_xgb(X_train, y_train, X_val, y_val)
    
    # Prepare sequence data
    seq_train, seq_val, _, _ = train_test_split(
        sequence_df['sequence'].values,
        labels,
        test_size=0.2,
        random_state=42
    )
    
    seq_train = np.array([np.array(s) for s in seq_train])
    seq_val = np.array([np.array(s) for s in seq_val])
    
    ensemble.train_lstm(seq_train, y_train, seq_val, y_val)
    
    # Save models
    ensemble.save_models('models/hybrid_ensemble')


def predict_example():
    """Example: Real-time prediction"""
    predictor = RealTimePredictor('models/hybrid_ensemble')
    
    # Predict for live match
    result = predictor.predict_live_match('match_12345')
    print(f"Win Probability: {result['win_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference Time: {result['inference_time_ms']:.1f}ms")


if __name__ == "__main__":
    # train_example()
    predict_example()
