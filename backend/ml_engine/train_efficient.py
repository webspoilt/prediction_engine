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
import mlflow
import mlflow.pytorch
import mlflow.xgboost
from hybrid_model import HybridEnsemble, ModelConfig, CricsheetNormalizer

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
                # Load one match at a time
                df = self.normalizer.load_cricsheet_data(filepath)
                # Create match features
                static_features = self.normalizer.create_match_features(df)
                sequence_features = self.normalizer._create_sequences(df, sequence_length=18)
                
                # Yield ball-by-ball samples
                for idx in range(len(sequence_features)):
                    seq = sequence_features.iloc[idx]['sequence']
                    
                    # Placeholder label generation for demonstration
                    # Should be replaced by actual match outcome extraction
                    label = np.random.randint(0, 2) 
                    
                    yield torch.FloatTensor(seq), torch.FloatTensor([label])
            except Exception as e:
                logger.warning(f"Failed processing {filepath}: {e}")
            finally:
                # Force garbage collection to free RAM
                del df, static_features, sequence_features
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

    # Initialize MLflow
    mlflow.set_experiment("IPL_Prediction_Engine_V2")
    
    with mlflow.start_run(run_name=f"Hybrid_Train_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log Hyperparameters
        mlflow.log_params({
            "xgb_max_depth": config.xgb_params['max_depth'],
            "xgb_lr": config.xgb_params['learning_rate'],
            "lstm_hidden": config.lstm_hidden_size,
            "transformer_layers": config.transformer_num_layers,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate
        })

    # 1. Get all JSON files (matches)
    all_files = glob.glob(os.path.join(data_dir, '*.json'))
    if not all_files:
        logger.error(f"No JSON files found in {data_dir}. Please download them first.")
        return

    logger.info(f"Found {len(all_files)} match files. Processing in chunks to save RAM.")

    # Split files into training and validation sets
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
    
    # ---------------------------------------------------------
    # TESTING: Limit to 30 matches to quickly verify player stats 
    # integration. REMOVE THIS FOR FULL PRODUCTION TRAINING.
    # ---------------------------------------------------------
    train_files = train_files[:30]

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
                chunk_dfs.append(normalizer.load_cricsheet_data(f))
            except json.JSONDecodeError:
                continue
        
        if not chunk_dfs:
            continue
            
        combined_df = pd.concat(chunk_dfs, ignore_index=True)
        static_features = normalizer.create_match_features(combined_df)
        
        # DEBUG: Check types
        print("Feature Types:")
        print(static_features[normalizer.STATIC_FEATURE_COLS].dtypes)
        
        # Use Standardized Columns List
        X_chunk = static_features[normalizer.STATIC_FEATURE_COLS].fillna(0).values
        y_chunk = np.random.randint(0, 2, len(static_features)) # Placeholder
        
        dtrain = xgb.DMatrix(X_chunk, label=y_chunk)
        
        # Train incrementally (xgb_model handles starting from previous state)
        if xgb_model is None:
            xgb_model = xgb.train(params, dtrain, num_boost_round=10)
        else:
            xgb_model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=xgb_model)
            
        # Log XGBoost status to MLflow
        mlflow.log_metric("xgb_chunk_processed", i//chunk_size + 1)
            
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
            mlflow.log_metric("lstm_loss", loss.item(), step=batch_count)
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
            mlflow.log_metric("transformer_loss", loss.item(), step=t_batch_count)
            gc.collect()

    # Save the models
    logger.info("Training complete! Saving models...")
    os.makedirs('models', exist_ok=True)
    ensemble.save_models('models/hybrid_ensemble')
    
    # Log models to MLflow
    mlflow.xgboost.log_model(ensemble.xgb_model, "xgb_static")
    mlflow.pytorch.log_model(ensemble.lstm_model, "lstm_sequence")
    mlflow.pytorch.log_model(ensemble.transformer_model, "transformer_attention")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Efficient Training Pipeline")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to cricsheet JSONs')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    train_efficiently(args.data_dir)
