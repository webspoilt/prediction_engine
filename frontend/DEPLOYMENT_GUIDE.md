# IPL Win Predictor - Deployment & Quick Start Guide

## Overview

This guide covers deployment of the IPL Win Probability Prediction Engine across Asus TUF (primary) and Poco X3 (failover) systems, with complete instructions for data pipeline, vision backup, ML model, and reliability infrastructure setup.

---

## Prerequisites

### Hardware Requirements

**Primary (Asus TUF):**
- CPU: Intel i5 10th Gen or better
- RAM: 16GB minimum
- GPU: NVIDIA GTX 1650 Ti (or equivalent with CUDA support)
- Storage: 500GB SSD
- Network: Stable internet connection (10+ Mbps)

**Failover (Poco X3):**
- Termux/Linux environment
- 6GB RAM minimum
- 128GB storage
- SSH access to primary system

### Software Requirements

```bash
# Asus TUF
- Python 3.11+
- CUDA 11.8 + cuDNN 8.6
- Node.js 18+
- Redis 7.0+
- PM2 5.3+

# Poco X3
- Termux or Linux chroot
- Python 3.9+
- SSH client
```

---

## Installation Steps

### Step 1: Asus TUF Setup

#### 1.1 Install Python and Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Create virtual environment
python3.11 -m venv ~/ipl_predictor_env
source ~/ipl_predictor_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install playwright opencv-python ultralytics pytesseract
pip install xgboost tensorflow scikit-learn pandas numpy
pip install redis flask gunicorn
pip install psutil pynvml
```

#### 1.2 Install CUDA and cuDNN

```bash
# Download CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

#### 1.3 Install Redis

```bash
# Install Redis
sudo apt install -y redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping  # Should return PONG
```

#### 1.4 Install PM2

```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2 globally
sudo npm install -g pm2

# Enable PM2 startup
pm2 startup
pm2 save
```

### Step 2: Clone and Configure Project

```bash
# Clone the project
git clone <repository_url> ~/ipl_predictor
cd ~/ipl_predictor

# Create directories
mkdir -p logs data models configs

# Download Cricsheet data
wget https://cricsheet.org/downloads/ipl_csv2.zip
unzip ipl_csv2.zip -d data/

# Download YOLOv8 model
python3.11 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 3: Configure Environment Variables

```bash
# Create .env file
cat > ~/.env << EOF
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Data Sources
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
CRICBUZZ_API_KEY=your_api_key

# GPU
CUDA_VISIBLE_DEVICES=0
TF_CPP_MIN_LOG_LEVEL=2

# Failover
FAILOVER_IP=poco_x3_ip
FAILOVER_SSH_KEY=~/.ssh/poco_key

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
EOF

# Load environment
source ~/.env
```

### Step 4: Train ML Models

```bash
# Activate virtual environment
source ~/ipl_predictor_env/bin/activate

# Run training script
python3.11 scripts/train_models.py \
    --data data/ipl_csv2/ \
    --epochs 100 \
    --batch_size 32 \
    --output models/

# This will generate:
# - models/xgboost_model.pkl
# - models/lstm_model.h5
# - models/scaler.pkl
```

### Step 5: Deploy with PM2

```bash
# Copy PM2 configuration
cp configs/pm2.config.js ~/

# Start all processes
pm2 start pm2.config.js

# Verify processes
pm2 status

# View logs
pm2 logs

# Save PM2 configuration
pm2 save
```

---

## Failover Setup (Poco X3)

### Step 1: Termux Environment

```bash
# In Termux on Poco X3
pkg update && pkg upgrade -y
pkg install -y python git openssh

# Create SSH key on Asus TUF
ssh-keygen -t ed25519 -f ~/.ssh/poco_key -N ""

# Copy public key to Poco X3
ssh-copy-id -i ~/.ssh/poco_key.pub user@poco_x3_ip
```

### Step 2: Deploy Failover Scraper

```bash
# On Poco X3
git clone <repository_url> ~/ipl_predictor_failover
cd ~/ipl_predictor_failover

# Install dependencies
pip install playwright redis

# Create failover script
cat > ~/failover_scraper.py << 'EOF'
import redis
import time
import subprocess

redis_client = redis.Redis(host='asus_tuf_ip', port=6379)

def failover_scraper():
    while True:
        try:
            if not redis_client.get('heartbeat:asus_tuf'):
                print("Primary down, activating failover...")
                subprocess.run(['python', 'scraper.py'])
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(30)

if __name__ == '__main__':
    failover_scraper()
EOF

# Run failover scraper
nohup python failover_scraper.py > failover.log 2>&1 &
```

---

## Monitoring and Maintenance

### Real-time Monitoring

```bash
# Monitor processes
pm2 monit

# View specific logs
pm2 logs data-scraper
pm2 logs vision-pipeline
pm2 logs ml-server

# Check system health
watch -n 1 'nvidia-smi'  # GPU metrics
watch -n 1 'free -h'     # Memory usage
```

### Health Checks

```bash
# Check Redis connection
redis-cli ping

# Check data pipeline status
redis-cli GET 'heartbeat:asus_tuf'

# Check vision pipeline
redis-cli GET 'gpu:metrics'

# Check ML model
redis-cli GET 'model:latest_prediction'
```

### Restart Procedures

```bash
# Restart all processes
pm2 restart all

# Restart specific process
pm2 restart data-scraper

# Restart with graceful shutdown
pm2 gracefulReload all

# Hard restart (if stuck)
pm2 kill && pm2 start pm2.config.js
```

---

## Testing

### Unit Tests

```bash
# Test data scraper
python3.11 -m pytest tests/test_scraper.py -v

# Test vision pipeline
python3.11 -m pytest tests/test_vision.py -v

# Test ML model
python3.11 -m pytest tests/test_model.py -v
```

### Integration Tests

```bash
# Test full pipeline with mock data
python3.11 tests/integration_test.py

# Test failover mechanism
python3.11 tests/test_failover.py

# Load test (100 concurrent predictions)
python3.11 tests/load_test.py --concurrency 100
```

### Live Testing

```bash
# Start monitoring
pm2 monit

# In another terminal, trigger prediction
python3.11 scripts/predict_live_match.py --match_id 12345

# Check prediction in Redis
redis-cli GET 'match:12345:prediction'
```

---

## Troubleshooting

### Issue: WebSocket Scraper Blocked

**Symptoms:** Data feed stops updating, heartbeat missing

**Solution:**
```bash
# Check if primary is still running
pm2 status

# Check logs for errors
pm2 logs data-scraper

# Manually trigger failover
redis-cli SET 'failover:active' 'true' EX 300

# Verify failover on Poco X3
ssh -i ~/.ssh/poco_key user@poco_x3_ip 'ps aux | grep scraper'
```

### Issue: GPU Thermal Throttling

**Symptoms:** Vision pipeline FPS drops, GPU temp > 85°C

**Solution:**
```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Reduce vision pipeline FPS
# Edit vision.py and increase sleep interval

# Improve cooling
# - Clean GPU fans
# - Improve case airflow
# - Reduce ambient temperature

# Restart vision pipeline
pm2 restart vision-pipeline
```

### Issue: Redis Connection Timeout

**Symptoms:** All services fail to connect to Redis

**Solution:**
```bash
# Check Redis status
sudo systemctl status redis-server

# Restart Redis
sudo systemctl restart redis-server

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log

# Verify port is open
netstat -tlnp | grep 6379
```

### Issue: Model Prediction Drift

**Symptoms:** Predictions become inaccurate over time

**Solution:**
```bash
# Retrain model on latest data
python3.11 scripts/train_models.py --data data/latest_matches/ --retrain

# Validate against live matches
python3.11 scripts/validate_model.py --matches 100

# If accuracy < 75%, rollback to previous model
cp models/xgboost_model.pkl.backup models/xgboost_model.pkl
```

---

## Performance Optimization

### Data Pipeline Optimization

```python
# Use connection pooling
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=10
)

# Batch updates
pipeline = redis_client.pipeline()
for i in range(100):
    pipeline.set(f'key:{i}', value)
pipeline.execute()
```

### Vision Pipeline Optimization

```python
# Use lower resolution for faster processing
frame = cv2.resize(frame, (640, 480))

# Use GPU acceleration
cuda_device = cv2.cuda_GpuMat()
cuda_device.upload(frame)

# Batch inference
results = model(frames, batch_size=8)
```

### ML Model Optimization

```python
# Use model quantization
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Use inference optimization
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## Backup and Recovery

### Backup Strategy

```bash
# Daily backup of models and data
0 2 * * * /home/ubuntu/backup.sh

# Backup script
#!/bin/bash
BACKUP_DIR="/mnt/backup/ipl_predictor"
mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models_$(date +%Y%m%d).tar.gz ~/ipl_predictor/models/

# Backup Redis data
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_$(date +%Y%m%d).rdb

# Backup logs
tar -czf $BACKUP_DIR/logs_$(date +%Y%m%d).tar.gz ~/ipl_predictor/logs/

# Upload to cloud (optional)
# gsutil -m cp -r $BACKUP_DIR gs://ipl-predictor-backups/
```

### Recovery Procedure

```bash
# Restore from backup
tar -xzf /mnt/backup/ipl_predictor/models_20260327.tar.gz -C ~/ipl_predictor/

# Restore Redis data
redis-cli SHUTDOWN
cp /mnt/backup/ipl_predictor/redis_20260327.rdb /var/lib/redis/dump.rdb
sudo systemctl start redis-server

# Restart all services
pm2 restart all
```

---

## Security Considerations

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 6379/tcp  # Redis (local only)
sudo ufw allow 5000/tcp  # API (if exposed)
sudo ufw enable
```

### SSH Security

```bash
# Use key-based authentication only
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Restrict SSH access
echo "AllowUsers ubuntu@asus_tuf_ip" >> /etc/ssh/sshd_config
```

### Redis Security

```bash
# Set Redis password
redis-cli CONFIG SET requirepass "your_strong_password"

# Bind to localhost only
redis-cli CONFIG SET bind "127.0.0.1"

# Disable dangerous commands
redis-cli CONFIG SET rename-command FLUSHDB ""
redis-cli CONFIG SET rename-command FLUSHALL ""
```

---

## Monitoring Dashboard

Create a simple monitoring dashboard:

```python
# monitor.py
from flask import Flask, jsonify
import redis
import psutil
import pynvml

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'redis': redis_client.ping(),
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'gpu_temp': pynvml.nvmlDeviceGetTemperature(handle, 0),
        'gpu_util': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Access at: `http://asus_tuf_ip:5000/health`

---

## Support and Documentation

- **System Architecture**: See `SYSTEM_ARCHITECTURE.md`
- **API Documentation**: See `API_DOCS.md`
- **Troubleshooting**: See section above
- **Contact**: For issues, check logs at `~/ipl_predictor/logs/`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-27 | Initial release |
| 1.1.0 | TBD | Model optimization, improved failover |
| 2.0.0 | TBD | Multi-GPU support, cloud sync |

