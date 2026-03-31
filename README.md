---
title: IPL Prediction Engine (Hybrid XGBoost + LSTM)
emoji: 🏏
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# IPL Prediction Engine (Hybrid XGBoost + LSTM)

A high-performance machine learning engine designed to predict IPL match outcomes and ball-by-ball win probabilities using a hybrid ensemble architecture.

## 🚀 Key Features
- **Hybrid Architecture**: Combines **XGBoost** (for static match context) and **LSTM** (for temporal momentum).
- **Enriched Context**: Real-time integration of historical **Open-Meteo Weather Data** (dew, temperature, humidity).
- **Dynamic Form Tracking**: Rolling **ELO ratings** for teams and players calculated from 2008 to 2024.
- **Memory Efficient**: Optimized training pipeline using `IterableDataset` to handle 1000+ match files on low-RAM hardware.
- **Real-Time Ready**: Built-in support for Redis-based live match streaming.

## 🛠 Project Structure
- `backend/ml_engine/hybrid_model.py`: Core model architecture and normalization.
- `backend/ml_engine/train_efficient.py`: Memory-efficient training pipeline.
- `backend/data_pipeline/fetch_weather_data.py`: Historical weather data harvester.
- `backend/data_pipeline/feature_engineer_elo.py`: ELO and form calculation engine.

## 📦 Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage
### 1. Enrich the Data
First, generate the weather and ELO datasets:
```bash
python backend/data_pipeline/fetch_weather_data.py
python backend/data_pipeline/feature_engineer_elo.py
```

### 2. Train the Model
```bash
python backend/ml_engine/train_efficient.py
```

### 3. Run Inference Test
```bash
python backend/ml_engine/test_inference.py
```

## ⚖️ License
MIT
