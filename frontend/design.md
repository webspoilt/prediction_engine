# IPL Win Probability Prediction Engine - Mobile App Design

## Design Philosophy

This app is designed for **portrait orientation (9:16)** with **one-handed usage** in mind. Following **Apple Human Interface Guidelines (HIG)**, the app feels like a first-party iOS application with clean typography, intuitive navigation, and real-time data visualization.

---

## Screen List

### 1. **Dashboard (Home Screen)**
The primary entry point displaying real-time match data and win probability predictions.

**Primary Content:**
- Live match card with team names, logos, and current score
- Win probability gauge (circular progress indicator)
- Match status badge (Live, Upcoming, Completed)
- Quick stats: Current Run Rate (CRR), Required Run Rate (RRR), Wickets remaining
- Upcoming matches list (horizontal scroll)

**Functionality:**
- Tap match card to navigate to detailed match view
- Swipe to refresh live data
- Tap upcoming match to set as active

---

### 2. **Match Detail Screen**
Comprehensive view of a single match with all prediction data.

**Primary Content:**
- Full scoreboard (Team A vs Team B)
- Ball-by-ball commentary
- Win probability chart (line graph over time)
- Player statistics
- Prediction confidence score
- System health status (Data Pipeline, Vision Backup, ML Model)

**Functionality:**
- Real-time updates every 2-5 minutes
- Tap on player cards to see detailed stats
- Swipe between innings
- View prediction history

---

### 3. **Data Pipeline Module Screen**
Displays WebSocket scraping status and data feed information.

**Primary Content:**
- Primary scraper status (Asus TUF connection)
- Fallback scraper status (Poco X3 connection)
- Data sources (Betfair, Cricbuzz, etc.)
- Last update timestamp
- WebSocket connection indicator
- Code viewer for scraping logic
- Redundancy status

**Functionality:**
- View live WebSocket traffic
- Toggle between data sources
- Inspect payload samples
- View scraper logs

---

### 4. **Vision Backup Module Screen**
Shows OpenCV + YOLOv8 scoreboard detection status.

**Primary Content:**
- Live stream region capture preview
- OCR extracted data (Runs, Wickets, Overs)
- Confidence scores for each extracted value
- GPU utilization (GTX 1650 Ti)
- Thermal status
- Vision model performance metrics
- Code viewer for vision pipeline

**Functionality:**
- Start/stop vision capture
- Adjust capture region
- View OCR confidence scores
- Monitor GPU temperature
- Inspect extracted data samples

---

### 5. **Hybrid ML Model Screen**
Displays the ensemble model architecture and predictions.

**Primary Content:**
- Model architecture diagram (XGBoost + LSTM/GRU layers)
- Feature importance chart
- Model performance metrics (Accuracy, Precision, Recall)
- Prediction breakdown (XGBoost vs LSTM contribution)
- Training data statistics
- Model version and last update
- Code viewer for model logic

**Functionality:**
- View real-time feature importance
- Inspect prediction components
- See model training history
- Download model weights

---

### 6. **Reliability Infrastructure Screen**
Monitors Redis, PM2, and heartbeat mechanisms.

**Primary Content:**
- Redis connection status
- PM2 process manager status
- Heartbeat indicator (Asus TUF ↔ Poco X3)
- System uptime
- Memory usage (both devices)
- Process logs
- Failover history

**Functionality:**
- Restart processes
- View detailed logs
- Monitor heartbeat latency
- Check failover triggers
- View system health history

---

### 7. **Settings Screen**
Configuration and app preferences.

**Primary Content:**
- Notification preferences
- Data refresh interval
- Theme selection (Light/Dark)
- About app
- System logs export
- API configuration (if needed)

**Functionality:**
- Toggle notifications
- Adjust refresh rate
- Export logs
- View version info

---

## Primary User Flows

### Flow 1: Check Live Match Prediction
1. User opens app → Dashboard
2. Dashboard shows live match card with win probability
3. User taps match card → Match Detail Screen
4. Sees real-time updates, ball-by-ball commentary, and prediction chart
5. User can swipe to refresh or navigate to other matches

### Flow 2: Monitor System Health
1. User navigates to Data Pipeline/Vision/ML Model/Reliability screens via tab bar
2. Each screen shows real-time status of its respective module
3. User can inspect logs, view code, and monitor performance metrics
4. Alerts appear if any module degrades

### Flow 3: Analyze Prediction Confidence
1. User on Match Detail Screen
2. Taps on prediction confidence score
3. Views breakdown of XGBoost vs LSTM contributions
4. Sees feature importance and model reasoning
5. Can export prediction data for analysis

---

## Color Scheme

### Primary Colors
- **Brand Blue**: `#0a7ea4` (Primary accent, buttons, highlights)
- **Success Green**: `#22C55E` (Win probability, positive indicators)
- **Warning Orange**: `#F59E0B` (Alerts, caution states)
- **Error Red**: `#EF4444` (System failures, critical alerts)

### Neutral Colors
- **Background**: `#ffffff` (Light mode), `#151718` (Dark mode)
- **Surface**: `#f5f5f5` (Light mode), `#1e2022` (Dark mode)
- **Foreground**: `#11181C` (Light mode), `#ECEDEE` (Dark mode)
- **Muted**: `#687076` (Light mode), `#9BA1A6` (Dark mode)
- **Border**: `#E5E7EB` (Light mode), `#334155` (Dark mode)

### Data Visualization
- **Team A**: `#1E40AF` (Blue)
- **Team B**: `#DC2626` (Red)
- **Neutral**: `#6B7280` (Gray)

---

## Key User Flows Summary

| Flow | Start | End | Key Actions |
|------|-------|-----|-------------|
| Check Prediction | Dashboard | Match Detail | Tap match, view probability, refresh |
| Monitor System | Dashboard | Module Screens | Navigate tabs, inspect status, view logs |
| Analyze Confidence | Match Detail | Prediction Breakdown | Tap score, view components, export |
| Configure App | Dashboard | Settings | Adjust preferences, export logs |

---

## Design Principles

1. **Real-time Focus**: All data updates within 2-5 minutes; loading states clearly indicated
2. **System Transparency**: Users can see data sources, model components, and system health
3. **One-handed Usage**: Primary actions (refresh, navigate) accessible with thumb
4. **Minimal Cognitive Load**: Complex data simplified into visual indicators (gauges, charts)
5. **Accessibility**: High contrast colors, readable fonts (min 16pt for body text)
6. **Performance**: Smooth animations, no jank, responsive to touch
