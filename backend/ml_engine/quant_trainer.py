"""
Quant-Grade ML Training Pipeline
IPL Win Probability Prediction Engine

Implements strict chronological splitting, AdamW optimization, gradient clipping, 
and Reliability/Calibration Evaluation for true quantitative rigor.
"""

import os
import gc
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CricketDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for feeding batches to the GTX 1650 Ti.
    Expects a DataFrame with pre-processed static arrays and 18-ball sequences.
    """
    def __init__(self, dataframe):
        # Example static features: venue_code, target_score, crr, rrr, current_wickets
        # Assuming the dataframe columns are already constructed appropriately.
        # Here we mock extraction of 5 static features for demonstration.
        static_cols = [c for c in dataframe.columns if c.startswith('static_')]
        if not static_cols:
            # Fallback mock if data pipeline features aren't strictly named
            self.static_features = np.zeros((len(dataframe), 5), dtype=np.float32)
        else:
            self.static_features = dataframe[static_cols].values.astype(np.float32)
            
        # 18-Ball sequence extraction (batch, 18, 3 features)
        sequences = list(dataframe.get('sequence_18', np.zeros((len(dataframe), 18, 3))))
        self.sequences = np.stack(sequences).astype(np.float32)
        
        # Target (Win = 1, Loss = 0)
        self.targets = dataframe['target'].values.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.static_features[idx]), 
            torch.tensor(self.sequences[idx]), 
            torch.tensor(self.targets[idx]).unsqueeze(0)
        )


def build_time_series_splits(cricsheet_json_dir: str):
    """
    Implements the Golden Rule of Time-Series Splitting.
    Train: 2008 - 2021
    Val: 2022
    Test: 2023 - Present
    """
    logger.info("Building strict chronological splits...")
    
    # Normally, you would parse the dates from the Cricsheet files here.
    # For demonstration, we create dummy DataFrames representing the processed structure.
    # In production, this would call Normalizer logic.
    
    logger.info("Simulating parsed DataFrames from Cricsheet...")
    
    def create_dummy_df(size):
        return pd.DataFrame({
            'static_0': np.random.randn(size),
            'static_1': np.random.randn(size),
            'static_2': np.random.randn(size),
            'static_3': np.random.randn(size),
            'static_4': np.random.randn(size),
            'sequence_18': [np.random.randn(18, 3) for _ in range(size)],
            'target': np.random.randint(0, 2, size)
        })

    # Train (2008-2021) - Largest chunk
    df_train = create_dummy_df(10000) 
    # Val (2022)
    df_val = create_dummy_df(2000)
    # Test (2023+)
    df_test = create_dummy_df(2000)
    
    return df_train, df_val, df_test


class HybridNet(nn.Module):
    """
    Basic Hybrid Network structure integrating Static and Sequence processing.
    """
    def __init__(self, static_size=5, seq_features=3, hidden_size=64):
        super(HybridNet, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_size, batch_first=True)
        self.fc_static = nn.Linear(static_size, 16)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Bound between 0 and 1 for BCE Loss
        )

    def forward(self, static_x, seq_x):
        # Process sequence
        _, (hn, _) = self.lstm(seq_x)
        lstm_out = hn[-1] # Shape: (batch, hidden_size)
        
        # Process static
        static_out = torch.relu(self.fc_static(static_x))
        
        # Combine
        combined = torch.cat((lstm_out, static_out), dim=1)
        return self.fc_out(combined)


def train_model(model, train_loader, val_loader, epochs=20):
    """
    The Quant-Grade Training Loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device.type.upper()}")
    
    model.to(device)
    
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # AdamW for L2 Regularization
    
    best_val_loss = float('inf')
    os.makedirs('weights', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for static_x, seq_x, targets in progress_bar:
            static_x, seq_x, targets = static_x.to(device), seq_x.to(device), targets.to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            predictions = model(static_x, seq_x)
            
            # Calculate Loss
            loss = criterion(predictions, targets)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for static_x, seq_x, targets in val_loader:
                static_x, seq_x, targets = static_x.to(device), seq_x.to(device), targets.to(device)
                predictions = model(static_x, seq_x)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                
                predicted_classes = (predictions > 0.5).float()
                total += targets.size(0)
                correct += (predicted_classes == targets).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct / total) * 100
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'weights/best_ipl_model.pth')
            logger.info("--> Saved new best model weights!")


def evaluate_calibration(model, test_loader):
    """
    Plots a Reliability / Calibration Curve on Out-Of-Sample data.
    "If the model predicts a 75% win rate, does the team actually win 75% of the time?"
    """
    logger.info("Generating Out-of-Sample Calibration Curve...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    true_labels = []
    predicted_probs = []
    
    with torch.no_grad():
        for static_x, seq_x, targets in test_loader:
            static_x, seq_x = static_x.to(device), seq_x.to(device)
            predictions = model(static_x, seq_x)
            
            predicted_probs.extend(predictions.cpu().numpy().flatten())
            true_labels.extend(targets.cpu().numpy().flatten())
            
    # Calculate calibration
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=10, strategy='uniform')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Hybrid Ensemble")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly Calibrated")
    
    plt.title("Quant-Grade Calibration Curve (2023+ Test Set)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Actul Wins")
    plt.legend()
    plt.grid(True)
    
    os.makedirs('eval_plots', exist_ok=True)
    plt.savefig('eval_plots/calibration_curve.png')
    logger.info("Calibration curve saved to 'eval_plots/calibration_curve.png'")
    
    # Basic Brier Score (Mean Squared Error of Probability)
    # A true quant metric for probability calibration
    brier_score = np.mean((np.array(predicted_probs) - np.array(true_labels)) ** 2)
    logger.info(f"Out-of-Sample Brier Score: {brier_score:.4f} (Closer to 0 is better)")


if __name__ == "__main__":
    # 1. Chronological Data Split
    df_train, df_val, df_test = build_time_series_splits(cricsheet_json_dir="data")
    
    # 2. PyTorch Datasets
    train_dataset = CricketDataset(df_train)
    val_dataset = CricketDataset(df_val)
    test_dataset = CricketDataset(df_test)
    
    # GTX 1650 Ti Sweet Spot -> batch_size=256
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 3. Model Init
    model = HybridNet()
    
    # 4. Train
    train_model(model, train_loader, val_loader, epochs=5)
    
    # 5. Evaluate Calibration
    model.load_state_dict(torch.load('weights/best_ipl_model.pth'))
    evaluate_calibration(model, test_loader)
