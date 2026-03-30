# IPL Win Probability Prediction Engine
## Deployment Roadmap & Integration Guide

**Target Uptime: 99.99%** (52.56 minutes downtime/year)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Hardware Setup](#hardware-setup)
3. [Software Prerequisites](#software-prerequisites)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Poco X3 (Termux) Setup](#poco-x3-termux-setup)
6. [Integration Guide](#integration-guide)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASUS TUF (Primary)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │  WebSocket  │  │   Vision    │  │   Hybrid ML Model   │    │
│  │   Sniffer   │  │   Backup    │  │  (XGBoost + LSTM)   │    │
│  │  (Task 1)   │  │  (Task 2)   │  │     (Task 3)        │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘    │
│         └─────────────────┴────────────────────┘                │
│                           │                                     │
│                    ┌──────┴──────┐                             │
│                    │    Redis    │                             │
│                    │   (State)   │                             │
│                    └──────┬──────┘                             │
│                           │                                     │
│                    ┌──────┴──────┐                             │
│                    │     PM2     │                             │
│                    │ (Self-Heal) │                             │
│                    └─────────────┘                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Heartbeat
┌───────────────────────────┼─────────────────────────────────────┐
│                    POCO X3 (Backup/Failover)                   │
│  ┌────────────────────────┘                                     │
│  │  Termux + Linux Deploy                                       │
│  │  - Backup Scraper                                            │
│  │  - Emergency API                                             │
│  │  - Health Monitor                                            │
│  └────────────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Setup

### Asus TUF Configuration

| Component | Specification | Purpose |
|-----------|--------------|---------|
| CPU | Intel i5 10th Gen | Data processing, ML inference |
| RAM | 16GB DDR4 | Model loading, Redis cache |
| GPU | GTX 1650 Ti 4GB | LSTM inference, YOLOv8 detection |
| Storage | 512GB SSD | Model storage, logs |
| Network | WiFi 6 / Ethernet | WebSocket connections |

### Poco X3 Configuration

| Component | Specification | Purpose |
|-----------|--------------|---------|
| RAM | 6GB | Backup scraper, lightweight API |
| Storage | 128GB + SD Card | Termux environment |
| OS | Android 12 + Termux | Linux environment |
| Network | WiFi / Mobile Data | Failover connectivity |

---

## Software Prerequisites

### Asus TUF (Windows/Linux)

```bash
# 1. Install Python 3.10+
python --version  # Should be 3.10 or higher

# 2. Install CUDA (for GTX 1650 Ti)
# Download from: https://developer.nvidia.com/cuda-downloads
nvcc --version  # Verify CUDA installation

# 3. Install Redis
# Windows: Use WSL2 or download from https://github.com/microsoftarchive/redis/releases
# Linux:
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl enable redis
sudo systemctl start redis

# 4. Install Node.js & PM2
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# 5. Install Tesseract OCR
# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Linux:
sudo apt-get install tesseract-ocr

# 6. Install Python dependencies
pip install -r requirements.txt
```

### requirements.txt

```
# Core
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# ML Models
xgboost>=1.7.0

# Web Scraping
playwright>=1.35.0
beautifulsoup4>=4.12.0
aiohttp>=3.8.0

# Computer Vision
opencv-python>=4.8.0
pytesseract>=0.3.10
Pillow>=10.0.0

# Data & State
redis>=4.6.0

# Monitoring
pynvml>=11.5.0
psutil>=5.9.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## Step-by-Step Deployment

### Phase 1: Environment Setup (Day 1)

```bash
# 1. Create project directory
mkdir -p ~/ipl-predictor/{logs,models,data,config}
cd ~/ipl-predictor

# 2. Clone/download all task files
cp /path/to/task*.py ./
cp /path/to/ecosystem.config.js ./

# 3. Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or
venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Playwright browsers
playwright install chromium

# 6. Download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 7. Test Redis connection
redis-cli ping  # Should return PONG
```

### Phase 2: Model Training (Day 2-3)

```bash
# 1. Download Cricsheet data
mkdir -p data/cricsheet
cd data/cricsheet
wget https://cricsheet.org/downloads/ipl_json.zip
unzip ipl_json.zip

# 2. Train the hybrid model
cd ~/ipl-predictor
python task3_hybrid_ml_model.py --mode train \
    --data-dir ./data/cricsheet \
    --output-dir ./models/hybrid_ensemble \
    --epochs 100

# 3. Verify model files
ls -la ./models/hybrid_ensemble/
# Should show: hybrid_ensemble_xgb.json, hybrid_ensemble_lstm.pth, hybrid_ensemble_scalers.pkl
```

### Phase 3: System Deployment (Day 4)

```bash
# 1. Generate PM2 config
python task4_infrastructure.py

# 2. Create logs directory
mkdir -p logs

# 3. Configure PM2 log rotation
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 100M
pm2 set pm2-logrotate:retain 10

# 4. Start all services
pm2 start ecosystem.config.js

# 5. Save PM2 config
pm2 save
pm2 startup  # Follow instructions to enable auto-start on boot

# 6. Monitor processes
pm2 status
pm2 logs
```

### Phase 4: Failover Setup (Day 5)

```bash
# On Asus TUF - Configure heartbeat
export BACKUP_HOST=192.168.1.100  # Poco X3 IP
export HEARTBEAT_PORT=7777

# Test connectivity
ping $BACKUP_HOST
telnet $BACKUP_HOST $HEARTBEAT_PORT
```

---

## Poco X3 (Termux) Setup

### Step 1: Install Termux

```bash
# On Poco X3:
# 1. Download Termux from F-Droid (NOT Play Store)
# https://f-droid.org/packages/com.termux/

# 2. Update packages
pkg update && pkg upgrade -y

# 3. Install essential packages
pkg install -y git python redis nodejs

# 4. Install Python packages
pip install redis aiohttp fastapi uvicorn
```

### Step 2: Install Linux Deploy (Optional but Recommended)

```bash
# 1. Install Linux Deploy from F-Droid
# https://f-droid.org/packages/ru.meefik.linuxdeploy/

# 2. Configure Ubuntu container:
# - Distribution: Ubuntu
# - Architecture: arm64
# - Version: jammy
# - Installation type: Directory
# - Username: ipl
# - Password: [secure password]

# 3. Start container and SSH into it
```

### Step 3: Deploy Backup Services

```bash
# In Termux or Linux Deploy:
mkdir -p ~/ipl-backup
cd ~/ipl-backup

# Create minimal backup scraper
cat > backup_scraper.py << 'EOF'
import asyncio
import redis
import aiohttp
from aiohttp import web
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupScraper:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.is_active = False
        
    async def health_handler(self, request):
        return web.json_response({
            'status': 'healthy',
            'mode': 'backup',
            'active': self.is_active
        })
    
    async def activate_handler(self, request):
        data = await request.json()
        self.is_active = True
        logger.info(f"Activated as primary for matches: {data}")
        # Start backup scraping logic
        return web.json_response({'status': 'activated'})
    
    async def standby_handler(self, request):
        self.is_active = False
        logger.info("Switched to standby mode")
        return web.json_response({'status': 'standby'})
    
    async def start(self):
        app = web.Application()
        app.router.add_get('/health', self.health_handler)
        app.router.add_post('/activate', self.activate_handler)
        app.router.add_post('/standby', self.standby_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        
        logger.info("Backup scraper started on port 8080")
        
        while True:
            await asyncio.sleep(3600)

if __name__ == '__main__':
    scraper = BackupScraper()
    asyncio.run(scraper.start())
EOF

# Start backup service
python backup_scraper.py &
```

### Step 4: Configure Auto-Start

```bash
# Add to ~/.bashrc or ~/.zshrc
if ! pgrep -f "backup_scraper.py" > /dev/null; then
    cd ~/ipl-backup && python backup_scraper.py &
fi

# For Linux Deploy, create systemd service
sudo tee /etc/systemd/system/ipl-backup.service << 'EOF'
[Unit]
Description=IPL Backup Scraper
After=network.target

[Service]
Type=simple
User=ipl
WorkingDirectory=/home/ipl/ipl-backup
ExecStart=/usr/bin/python3 /home/ipl/ipl-backup/backup_scraper.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable ipl-backup
sudo systemctl start ipl-backup
```

---

## Integration Guide

### API Endpoints

Once deployed, the system exposes these endpoints:

#### Prediction API

```bash
# Get win probability for a match
GET http://localhost:8080/predict/{match_id}

# Response:
{
    "match_id": "ipl_2024_match_45",
    "win_probability": 0.73,
    "confidence": 0.89,
    "xgb_probability": 0.71,
    "lstm_probability": 0.76,
    "inference_time_ms": 42.5,
    "timestamp": 1711531200.123
}
```

#### Health Check

```bash
# System health
GET http://localhost:8080/health

# Response:
{
    "status": "healthy",
    "components": {
        "scraper": "online",
        "vision_backup": "standby",
        "predictor": "online",
        "redis": "connected"
    },
    "uptime_seconds": 86400
}
```

#### Live Matches

```bash
# Get all active matches
GET http://localhost:8080/matches

# Response:
[
    {
        "match_id": "ipl_2024_match_45",
        "teams": ["MI", "CSK"],
        "status": "live",
        "current_over": 12.3,
        "score": "145/2"
    }
]
```

### WebSocket Subscription

```javascript
// Subscribe to live predictions
const ws = new WebSocket('ws://localhost:8080/ws/predictions');

ws.onmessage = (event) => {
    const prediction = JSON.parse(event.data);
    console.log(`Win Probability: ${prediction.win_probability}`);
};

// Subscribe to specific match
ws.send(JSON.stringify({
    action: 'subscribe',
    match_id: 'ipl_2024_match_45'
}));
```

---

## Monitoring & Maintenance

### PM2 Commands

```bash
# View process status
pm2 status

# View logs
pm2 logs
pm2 logs ipl-scraper --lines 100

# Restart specific service
pm2 restart ipl-scraper

# Monitor resources
pm2 monit

# Flush logs
pm2 flush
```

### Redis Commands

```bash
# Check Redis memory usage
redis-cli info memory

# Monitor live operations
redis-cli monitor

# View stream data
redis-cli xread streams ipl:stream:balls:match_123 0

# Check keys
redis-cli keys 'ipl:*'
```

### Health Metrics

```bash
# View system health
curl http://localhost:8080/health

# View metrics history
curl http://localhost:8080/metrics
```

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Failed

```bash
# Check if target site is accessible
curl -I https://www.cricbuzz.com

# Check Playwright installation
playwright install --force chromium

# Verify in headful mode for debugging
# Edit task1_websocket_pipeline.py: headless=False
```

#### 2. GPU Out of Memory

```bash
# Check GPU usage
nvidia-smi

# Reduce batch size in task3_hybrid_ml_model.py
# config.batch_size = 32  # Reduce from 64

# Enable gradient checkpointing for LSTM
```

#### 3. Redis Connection Refused

```bash
# Check Redis status
sudo systemctl status redis

# Restart Redis
sudo systemctl restart redis

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

#### 4. High Inference Latency

```bash
# Profile model inference
python -m torch.utils.bottleneck task3_hybrid_ml_model.py

# Check thermal throttling
nvidia-smi -q -d TEMPERATURE

# Reduce LSTM sequence length
# config.sequence_length = 12  # Reduce from 18
```

#### 5. Failover Not Triggering

```bash
# Test heartbeat connectivity
ping 192.168.1.100  # Poco X3 IP

# Check firewall rules
sudo ufw status
sudo ufw allow 7777/tcp
sudo ufw allow 8080/tcp

# Verify backup service
ssh poco-x3 "curl http://localhost:8080/health"
```

### Log Locations

| Component | Log Path |
|-----------|----------|
| WebSocket Scraper | `~/ipl-predictor/logs/scraper-*.log` |
| Vision Backup | `~/ipl-predictor/logs/vision-*.log` |
| ML Predictor | `~/ipl-predictor/logs/predictor-*.log` |
| Heartbeat | `~/ipl-predictor/logs/heartbeat-*.log` |
| API Server | `~/ipl-predictor/logs/api-*.log` |
| Redis | `/var/log/redis/redis-server.log` |
| PM2 | `~/.pm2/logs/` |

---

## Performance Benchmarks

### Expected Performance (Asus TUF)

| Metric | Target | Notes |
|--------|--------|-------|
| Inference Latency | < 50ms | GTX 1650 Ti |
| Data Ingestion | < 2s | WebSocket sniffing |
| Throughput | 30 req/min | Per match |
| GPU Temp | < 75°C | Thermal management |
| Memory Usage | < 8GB | All services |

### Failover Times

| Scenario | Detection Time | Failover Time | Total |
|----------|---------------|---------------|-------|
| Scraper Failure | 15s | 5s | 20s |
| Redis Failure | 5s | 10s | 15s |
| Network Partition | 15s | 5s | 20s |
| Complete Primary Failure | 15s | 30s | 45s |

---

## Security Considerations

1. **Redis Security**
   ```bash
   # Enable Redis password
   redis-cli CONFIG SET requirepass "your_secure_password"
   
   # Bind to localhost only
   redis-cli CONFIG SET bind 127.0.0.1
   ```

2. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 6379/tcp  # Redis (localhost only)
   sudo ufw allow 8080/tcp  # API
   sudo ufw allow 7777/tcp  # Heartbeat
   sudo ufw enable
   ```

3. **API Authentication**
   ```python
   # Add to api_server.py
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.get("/predict/{match_id}")
   async def predict(match_id: str, token: str = Depends(security)):
       # Validate token
       ...
   ```

---

## Cost Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| Asus TUF | Sunk | Existing hardware |
| Poco X3 | Sunk | Existing hardware |
| Redis | Free | Open source |
| PM2 | Free | Open source |
| Playwright | Free | Open source |
| YOLOv8 | Free | Open source |
| Cricsheet Data | Free | Open data |
| **Total** | **$0** | Zero ongoing costs |

---

## Next Steps

1. **Model Improvement**
   - Collect more training data
   - Fine-tune on IPL-specific patterns
   - Add weather/pitch condition features

2. **Feature Enhancements**
   - Player-specific models
   - Partnership analysis
   - Powerplay/Death over specialists

3. **Trading Integration**
   - Betfair API integration (with API key)
   - Automated betting strategies
   - Risk management module

---

## Support & Resources

- **Redis Documentation**: https://redis.io/documentation
- **PM2 Documentation**: https://pm2.keymetrics.io/docs/
- **Playwright Docs**: https://playwright.dev/python/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Cricsheet Data**: https://cricsheet.org/

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: IPL Prediction Engine Team
