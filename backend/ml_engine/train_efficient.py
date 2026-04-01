import os
import gc
import glob
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import xgboost as xgb
from sklearn.model_selection import train_test_split
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ mlflow not found. Training will proceed without remote tracking.")
    MLFLOW_AVAILABLE = False
from backend.ml_engine.hybrid_model import HybridEnsemble, ModelConfig, CricsheetNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CricsheetIterableDataset(IterableDataset):
    """
    An IterableDataset that loads and yields JSON match data one by one 
    preventing entire dataset from being loaded into RAM simultaneously.
    """
    def __init__(self, file_paths, normalizer):
        self.file_paths = file_paths
        self.normalizer = normalizer

    def __iter__(self):
        for filepath in self.file_paths:
            try:
                with open(filepath, 'r') as file:
                    match_data = json.load(file)
                    winner = match_data.get('info', {}).get('outcome', {}).get('winner')

                # Load one match at a time
                df = self.normalizer.load_cricsheet_data(filepath)
                # Create match features
                static_features = self.normalizer.create_match_features(df)
                sequence_features = self.normalizer._create_sequences(df, sequence_length=18)
                
                # Yield ball-by-ball samples
                for idx in range(len(sequence_features)):
                    seq = sequence_features.iloc[idx]['sequence']
                    batting_team = df.iloc[idx]['batting_team']
                    
                    if pd.isna(winner) or not batting_team:
                        label = 0
                    else:
                        label = 1 if self.normalizer.normalize_team(batting_team) == self.normalizer.normalize_team(winner) else 0
                    
                    yield torch.FloatTensor(seq), torch.FloatTensor([label])
            except Exception as e:
                logger.warning(f"Failed processing {filepath}: {e}")
            finally:
                # Force garbage collection to free RAM
                for var_name in ['df', 'static_features', 'sequence_features']:
                    if var_name in locals():
                        del locals()[var_name]
                gc.collect()

def train_efficiently(data_dir='data'):
    """
    Trains the XGBoost and LSTM models incrementally using low-RAM techniques.
    """
    logger.info("Initializing Memory-Efficient Training Pipeline...")
    normalizer = CricsheetNormalizer()
    normalizer.load_player_stats('backend/ml_engine/player_stats_db.json')
    normalizer.load_enhanced_features(
        weather_path='dataset/enhanced/weather_dataset.csv',
        elo_path='dataset/enhanced/team_elo.csv'
    )
    config = ModelConfig()
    ensemble = HybridEnsemble(config)


def _run_training(config, ensemble, normalizer, train_files, data_dir):
    """Inner training function — runs inside MLflow context."""

    # Training limit configurable via env var (0 = no limit)
    max_matches = int(os.getenv('MAX_TRAIN_MATCHES', '0'))
    if max_matches > 0:
        train_files = train_files[:max_matches]
        logger.info(f"Training limited to {max_matches} matches (set MAX_TRAIN_MATCHES=0 for full)")

    # ---------------------------------------------------------
    # PART A: Incremental Training for XGBoost
    # ---------------------------------------------------------
    logger.info("Starting XGBoost Incremental Training...")
    xgb_model = None
    params = config.xgb_params
    
    # Process files in chunks of 50 matches (very low RAM usage)
    chunk_size = 50
    for i in range(0, len(train_files), chunk_size):
        chunk_files = train_files[i:i + chunk_size]
        logger.info(f"XGBoost - Processing chunk {i//chunk_size + 1}/{(len(train_files)//chunk_size) + 1}")
        
        chunk_dfs = []
        for f in chunk_files:
            try:
                with open(f, 'r') as file:
                    match_data = json.load(file)
                    winner = match_data.get('info', {}).get('outcome', {}).get('winner')
                
                match_df = normalizer.load_cricsheet_data(f)
                match_df['winner'] = winner
                chunk_dfs.append(match_df)
            except json.JSONDecodeError:
                continue
        
        if not chunk_dfs:
            continue
            
        combined_df = pd.concat(chunk_dfs, ignore_index=True)
        static_features = normalizer.create_match_features(combined_df)
        
        # DEBUG: Check types
        print("Feature Types:")
        print(static_features[normalizer.STATIC_FEATURE_COLS].dtypes)
        
        # Extract Actual Labels
        def get_label(row):
            w = row.get('winner')
            b = row.get('batting_team')
            if pd.isna(w) or not b: return 0
            return 1 if normalizer.normalize_team(b) == normalizer.normalize_team(w) else 0
            
        static_features['label'] = static_features.apply(get_label, axis=1)
        
        # Use Standardized Columns List
        X_chunk = static_features[normalizer.STATIC_FEATURE_COLS].fillna(0).values
        y_chunk = static_features['label'].values
        
        dtrain = xgb.DMatrix(X_chunk, label=y_chunk)
        
        # Train incrementally (xgb_model handles starting from previous state)
        if xgb_model is None:
            xgb_model = xgb.train(params, dtrain, num_boost_round=10)
        else:
            xgb_model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=xgb_model)
            
        # Log XGBoost status to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric("xgb_chunk_processed", i//chunk_size + 1)
            except Exception:
                pass
            
        # Free memory!
        del chunk_dfs, combined_df, static_features, X_chunk, y_chunk, dtrain
        gc.collect()

    ensemble.xgb_model = xgb_model

    # ---------------------------------------------------------
    # PART B: Streaming Training for PyTorch LSTM
    # ---------------------------------------------------------
    logger.info("Starting LSTM Streaming Training...")
    ensemble.lstm_model = ensemble.build_lstm_model()
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(ensemble.lstm_model.parameters(), lr=config.learning_rate)
    
    # Iterable datasets automatically stream from disk
    train_dataset = CricsheetIterableDataset(train_files, normalizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    ensemble.lstm_model.train()
    batch_count = 0
    
    for seq, label in train_loader:
        seq = seq.to(ensemble.device)
        label = label.to(ensemble.device)
        
        optimizer.zero_grad()
        features = ensemble.lstm_model(seq)
        predictions = features[:, 0]  # First feature logic
        
        loss = criterion(predictions, label.squeeze())
        loss.backward()
        optimizer.step()
        
        batch_count += 1
        if batch_count % 100 == 0:
            logger.info(f"LSTM Batch {batch_count}: Loss = {loss.item():.4f}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("lstm_loss", loss.item(), step=batch_count)
                except Exception:
                    pass
            gc.collect() # Periodically clean garbage within epoch
            
    # ---------------------------------------------------------
    # PART C: Streaming Training for PyTorch Transformer
    # ---------------------------------------------------------
    logger.info("Starting Transformer Streaming Training...")
    ensemble.transformer_model = ensemble.build_transformer_model()
    
    t_optimizer = torch.optim.Adam(ensemble.transformer_model.parameters(), lr=config.learning_rate)
    ensemble.transformer_model.train()
    
    t_batch_count = 0
    t_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    for seq, label in t_loader:
        seq, label = seq.to(ensemble.device), label.to(ensemble.device)
        t_optimizer.zero_grad()
        
        features = ensemble.transformer_model(seq)
        predictions = features[:, 0]
        
        loss = criterion(predictions, label.squeeze())
        loss.backward()
        t_optimizer.step()
        
        t_batch_count += 1
        if t_batch_count % 100 == 0:
            logger.info(f"Transformer Batch {t_batch_count}: Loss = {loss.item():.4f}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("transformer_loss", loss.item(), step=t_batch_count)
                except Exception:
                    pass
            gc.collect()

    # Save the models
    logger.info("Training complete! Saving models...")
    os.makedirs('models', exist_ok=True)
    ensemble.save_models('models/hybrid_ensemble')
    
    # Log models to MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.xgboost.log_model(ensemble.xgb_model, "xgb_static")
            mlflow.pytorch.log_model(ensemble.lstm_model, "lstm_sequence")
            mlflow.pytorch.log_model(ensemble.transformer_model, "transformer_attention")
        except Exception as e:
            logger.warning(f"MLflow model logging failed: {e}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Efficient Training Pipeline")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to cricsheet JSONs')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    train_efficiently(args.data_dir)
