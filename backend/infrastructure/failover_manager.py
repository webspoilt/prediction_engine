"""
TASK 4: 99.99% RELIABILITY INFRASTRUCTURE
IPL Win Probability Prediction Engine - High Availability Architecture

This module implements:
1. Redis for in-memory state management
2. PM2 for self-healing process management
3. Heartbeat mechanism between Asus TUF and Poco X3
4. Health monitoring and automatic failover

Target: 99.99% uptime (52.56 minutes downtime/year)
"""

import asyncio
import json
import logging
import time
import socket
import subprocess
import os
import signal
import sys
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import requests

# Redis
import redis
from redis.asyncio import Redis as AsyncRedis

# For HTTP health checks
from aiohttp import web, ClientSession, ClientTimeout
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILOVER = "failover"
    RECOVERING = "recovering"
    DOWN = "down"


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: float
    state: str
    primary_healthy: bool
    backup_healthy: bool
    redis_healthy: bool
    last_ball_timestamp: float
    data_latency_seconds: float
    inference_latency_ms: float
    error_count: int
    uptime_seconds: float


class RedisStateManager:
    """
    Redis-based state management for high-performance data handling.
    Uses Redis Streams for event sourcing and Pub/Sub for real-time updates.
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        # Sync client for blocking operations
        self.sync_client: Optional[redis.Redis] = None
        # Async client for non-blocking operations
        self.async_client: Optional[AsyncRedis] = None
        
        # Connection pool settings
        self.pool_kwargs = {
            'host': host,
            'port': port,
            'db': db,
            'password': password,
            'decode_responses': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'health_check_interval': 30,
            'retry_on_timeout': True,
        }
        
    def connect(self) -> bool:
        """Establish Redis connections"""
        try:
            self.sync_client = redis.Redis(**self.pool_kwargs)
            self.sync_client.ping()
            logger.info(f"Redis connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
            
    async def connect_async(self) -> bool:
        """Establish async Redis connection"""
        try:
            self.async_client = AsyncRedis(**self.pool_kwargs)
            await self.async_client.ping()
            return True
        except Exception as e:
            logger.error(f"Async Redis connection failed: {e}")
            return False
    
    # ==================== STREAM OPERATIONS ====================
    
    def publish_ball(self, match_id: str, ball_data: Dict) -> str:
        """Publish ball data to Redis Stream"""
        stream_key = f"ipl:stream:balls:{match_id}"
        
        # Add timestamp if not present
        if 'timestamp' not in ball_data:
            ball_data['timestamp'] = time.time()
        
        entry_id = self.sync_client.xadd(
            stream_key,
            ball_data,
            maxlen=10000,  # Keep last 10k balls per match
            approximate=True
        )
        
        # Also publish to pub/sub for real-time consumers
        self.sync_client.publish(
            f"ipl:live:{match_id}",
            json.dumps(ball_data)
        )
        
        return entry_id
    
    def get_recent_balls(self, match_id: str, count: int = 18) -> List[Dict]:
        """Get recent balls for a match"""
        stream_key = f"ipl:stream:balls:{match_id}"
        
        entries = self.sync_client.xrevrange(stream_key, count=count)
        
        # Reverse to get chronological order
        return [{**entry[1], 'id': entry[0]} for entry in reversed(entries)]
    
    def create_consumer_group(self, stream_key: str, group_name: str) -> bool:
        """Create a consumer group for distributed processing"""
        try:
            self.sync_client.xgroup_create(stream_key, group_name, id='0', mkstream=True)
            logger.info(f"Consumer group '{group_name}' created for '{stream_key}'")
            return True
        except redis.ResponseError as e:
            if 'already exists' in str(e):
                return True
            logger.error(f"Failed to create consumer group: {e}")
            return False
    
    def read_group(self, 
                   group_name: str, 
                   consumer_name: str,
                   streams: List[str],
                   count: int = 10,
                   block: int = 5000) -> List:
        """Read from consumer group"""
        return self.sync_client.xreadgroup(
            group_name,
            consumer_name,
            {s: '>' for s in streams},
            count=count,
            block=block
        )
    
    def ack_message(self, stream_key: str, group_name: str, message_id: str):
        """Acknowledge message processing"""
        self.sync_client.xack(stream_key, group_name, message_id)
    
    # ==================== STATE MANAGEMENT ====================
    
    def set_match_state(self, match_id: str, state: Dict):
        """Set current match state"""
        key = f"ipl:state:{match_id}"
        state['updated_at'] = time.time()
        self.sync_client.hset(key, mapping=state)
        self.sync_client.expire(key, 86400)  # 24 hour TTL
    
    def get_match_state(self, match_id: str) -> Optional[Dict]:
        """Get current match state"""
        key = f"ipl:state:{match_id}"
        return self.sync_client.hgetall(key)
    
    def update_prediction(self, match_id: str, prediction: Dict):
        """Store latest prediction"""
        key = f"ipl:prediction:{match_id}"
        prediction['timestamp'] = time.time()
        self.sync_client.hset(key, mapping=prediction)
        
        # Add to time-series
        self.sync_client.xadd(
            f"ipl:predictions:history:{match_id}",
            prediction,
            maxlen=1000
        )
    
    # ==================== HEALTH & MONITORING ====================
    
    def record_heartbeat(self, component: str, metadata: Dict = None):
        """Record component heartbeat"""
        key = f"ipl:heartbeat:{component}"
        data = {
            'timestamp': time.time(),
            'status': 'alive'
        }
        if metadata:
            data.update(metadata)
        
        self.sync_client.hset(key, mapping=data)
        self.sync_client.expire(key, 60)  # 60 second TTL
    
    def check_heartbeat(self, component: str, max_age: int = 30) -> bool:
        """Check if component heartbeat is recent"""
        key = f"ipl:heartbeat:{component}"
        data = self.sync_client.hgetall(key)
        
        if not data:
            return False
        
        last_timestamp = float(data.get('timestamp', 0))
        return (time.time() - last_timestamp) < max_age
    
    def get_system_health(self) -> Dict:
        """Get overall system health status"""
        components = ['scraper', 'vision_backup', 'predictor', 'failover_manager']
        
        health = {
            'timestamp': time.time(),
            'redis_connected': self.sync_client.ping(),
            'components': {}
        }
        
        for comp in components:
            health['components'][comp] = {
                'healthy': self.check_heartbeat(comp),
                'last_seen': self.sync_client.hget(f"ipl:heartbeat:{comp}", 'timestamp')
            }
        
        return health


class HeartbeatManager:
    """
    Manages heartbeat communication between Asus TUF (primary) and Poco X3 (backup).
    Implements bidirectional health monitoring and automatic failover.
    """
    
    def __init__(self,
                 primary_host: str = '192.168.1.50',  # Asus TUF
                 backup_host: str = '192.168.1.100',  # Poco X3
                 heartbeat_port: int = 7777,
                 heartbeat_interval: int = 5):
        self.primary_host = primary_host
        self.backup_host = backup_host
        self.heartbeat_port = heartbeat_port
        self.heartbeat_interval = heartbeat_interval
        
        self.redis = RedisStateManager()
        self.is_primary = True  # Set based on hostname
        self.peer_host = backup_host if self.is_primary else primary_host
        
        # State tracking
        self.last_peer_heartbeat = 0
        self.missed_heartbeats = 0
        self.missed_threshold = 3
        self.is_failover_active = False
        
        # HTTP server for heartbeat endpoint
        self.app = web.Application()
        self.app.router.add_get('/health', self._health_handler)
        self.app.router.add_post('/heartbeat', self._heartbeat_handler)
        self.app.router.add_post('/failover', self._failover_handler)
        
    async def start(self):
        """Start heartbeat manager"""
        self.redis.connect()
        await self.redis.connect_async()
        
        # Start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.heartbeat_port)
        await site.start()
        
        logger.info(f"Heartbeat manager started on port {self.heartbeat_port}")
        
        # Start heartbeat loops
        await asyncio.gather(
            self._send_heartbeats(),
            self._monitor_peer(),
            self._health_check_loop()
        )
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to peer"""
        timeout = ClientTimeout(total=5)
        
        while True:
            try:
                async with ClientSession(timeout=timeout) as session:
                    health_data = {
                        'timestamp': time.time(),
                        'host': socket.gethostname(),
                        'role': 'primary' if self.is_primary else 'backup',
                        'system_health': self.redis.get_system_health()
                    }
                    
                    url = f"http://{self.peer_host}:{self.heartbeat_port}/heartbeat"
                    async with session.post(url, json=health_data) as resp:
                        if resp.status == 200:
                            self.missed_heartbeats = 0
                        else:
                            self.missed_heartbeats += 1
                            
            except Exception as e:
                logger.warning(f"Heartbeat send failed: {e}")
                self.missed_heartbeats += 1
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _monitor_peer(self):
        """Monitor peer health based on received heartbeats"""
        while True:
            time_since_last = time.time() - self.last_peer_heartbeat
            
            if time_since_last > self.heartbeat_interval * self.missed_threshold:
                logger.warning(f"Peer missed {self.missed_threshold} heartbeats!")
                
                if not self.is_primary and not self.is_failover_active:
                    await self._activate_failover()
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _health_check_loop(self):
        """Periodic health check and state recording"""
        while True:
            self.redis.record_heartbeat('failover_manager', {
                'missed_heartbeats': self.missed_heartbeats,
                'failover_active': self.is_failover_active,
                'peer_last_seen': self.last_peer_heartbeat
            })
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _activate_failover(self):
        """Activate failover mode"""
        logger.warning("!!! ACTIVATING FAILOVER MODE !!!")
        self.is_failover_active = True
        
        # Update Redis
        self.redis.sync_client.set('ipl:system:mode', 'failover')
        self.redis.sync_client.set('ipl:system:failover_time', time.time())
        
        # Start backup services
        await self._start_backup_services()
    
    async def _start_backup_services(self):
        """Start services in backup mode"""
        # This would start the scraper and predictor on Poco X3
        logger.info("Starting backup services...")
        
        # Signal to PM2 to start backup processes
        subprocess.run(['pm2', 'start', 'ecosystem.config.js', '--env', 'backup'])
    
    # ==================== HTTP HANDLERS ====================
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'role': 'primary' if self.is_primary else 'backup',
            'failover_active': self.is_failover_active
        }
        return web.json_response(health)
    
    async def _heartbeat_handler(self, request: web.Request) -> web.Response:
        """Receive heartbeat from peer"""
        try:
            data = await request.json()
            self.last_peer_heartbeat = time.time()
            
            # Store peer health in Redis
            peer_role = data.get('role', 'unknown')
            self.redis.sync_client.hset(
                f"ipl:peer:{peer_role}",
                mapping={
                    'last_heartbeat': data.get('timestamp'),
                    'host': data.get('host'),
                    'health': json.dumps(data.get('system_health', {}))
                }
            )
            
            return web.json_response({'status': 'ok'})
            
        except Exception as e:
            logger.error(f"Heartbeat processing error: {e}")
            return web.json_response({'status': 'error'}, status=500)
    
    async def _failover_handler(self, request: web.Request) -> web.Response:
        """Handle failover command"""
        if self.is_primary:
            return web.json_response(
                {'status': 'error', 'message': 'Primary cannot failover'},
                status=400
            )
        
        await self._activate_failover()
        return web.json_response({'status': 'failover_activated'})


class PM2ProcessManager:
    """
    PM2 integration for self-healing process management.
    Handles process lifecycle, auto-restart, and log management.
    """
    
    PROCESS_DEFINITIONS = {
        'ipl-scraper': {
            'script': 'task1_websocket_pipeline.py',
            'instances': 1,
            'autorestart': True,
            'max_restarts': 10,
            'min_uptime': '10s',
            'env': {
                'NODE_ENV': 'production'
            },
            'error_file': './logs/scraper-error.log',
            'out_file': './logs/scraper-out.log',
            'log_date_format': 'YYYY-MM-DD HH:mm:ss Z',
            'watch': False,
            'max_memory_restart': '1G',
            'kill_timeout': 5000,
            'listen_timeout': 10000,
        },
        'ipl-vision': {
            'script': 'task2_vision_backup.py',
            'instances': 1,
            'autorestart': True,
            'max_restarts': 5,
            'min_uptime': '10s',
            'env': {
                'CUDA_VISIBLE_DEVICES': '0'
            },
            'error_file': './logs/vision-error.log',
            'out_file': './logs/vision-out.log',
            'max_memory_restart': '2G',
        },
        'ipl-predictor': {
            'script': 'task3_hybrid_ml_model.py',
            'instances': 1,
            'autorestart': True,
            'max_restarts': 5,
            'env': {
                'MODEL_PATH': './models/hybrid_ensemble'
            },
            'error_file': './logs/predictor-error.log',
            'out_file': './logs/predictor-out.log',
            'max_memory_restart': '4G',
        },
        'ipl-heartbeat': {
            'script': 'task4_infrastructure.py',
            'instances': 1,
            'autorestart': True,
            'args': ['--mode', 'heartbeat'],
            'error_file': './logs/heartbeat-error.log',
            'out_file': './logs/heartbeat-out.log',
        },
        'ipl-api': {
            'script': 'api_server.py',
            'instances': 2,  # Cluster mode
            'exec_mode': 'cluster',
            'autorestart': True,
            'env': {
                'PORT': 8080
            },
            'error_file': './logs/api-error.log',
            'out_file': './logs/api-out.log',
        }
    }
    
    def __init__(self):
        self.redis = RedisStateManager()
        
    def generate_ecosystem_config(self, output_path: str = 'ecosystem.config.js'):
        """Generate PM2 ecosystem configuration file"""
        config = {
            'apps': []
        }
        
        for name, settings in self.PROCESS_DEFINITIONS.items():
            app_config = {
                'name': name,
                **settings
            }
            config['apps'].append(app_config)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('module.exports = ')
            json.dump(config, f, indent=2)
            f.write(';\n')
        
        logger.info(f"PM2 ecosystem config written to {output_path}")
        
    def start_all(self):
        """Start all processes with PM2"""
        subprocess.run(['pm2', 'start', 'ecosystem.config.js'])
        logger.info("All processes started")
        
    def stop_all(self):
        """Stop all processes"""
        subprocess.run(['pm2', 'stop', 'all'])
        
    def restart_process(self, process_name: str):
        """Restart a specific process"""
        subprocess.run(['pm2', 'restart', process_name])
        
    def get_status(self) -> Dict:
        """Get PM2 process status"""
        result = subprocess.run(
            ['pm2', 'jlist'],
            capture_output=True,
            text=True
        )
        
        try:
            processes = json.loads(result.stdout)
            status = {}
            
            for proc in processes:
                status[proc['name']] = {
                    'status': proc['pm2_env']['status'],
                    'uptime': proc['pm2_env']['pm_uptime'],
                    'restarts': proc['pm2_env']['restart_time'],
                    'memory': proc['monit']['memory'],
                    'cpu': proc['monit']['cpu']
                }
            
            return status
            
        except json.JSONDecodeError:
            logger.error("Failed to parse PM2 status")
            return {}
    
    def setup_log_rotation(self):
        """Configure PM2 log rotation"""
        # Install pm2-logrotate if not present
        subprocess.run(['pm2', 'install', 'pm2-logrotate'])
        
        # Configure rotation
        subprocess.run([
            'pm2', 'set', 'pm2-logrotate:max_size', '100M'
        ])
        subprocess.run([
            'pm2', 'set', 'pm2-logrotate:retain', '10'
        ])
        subprocess.run([
            'pm2', 'set', 'pm2-logrotate:compress', 'true'
        ])
        
        logger.info("Log rotation configured")


class HealthMonitor:
    """
    Comprehensive health monitoring with alerting.
    Tracks system metrics and triggers alerts on anomalies.
    """
    
    def __init__(self):
        self.redis = RedisStateManager()
        self.redis.connect()
        
        self.alert_callbacks: List[Callable] = []
        self.metrics_history: List[HealthMetrics] = []
        
        # Thresholds
        self.thresholds = {
            'max_data_latency': 120,  # seconds
            'max_inference_latency': 100,  # ms
            'max_error_rate': 0.05,  # 5%
        }
        
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
        
    async def start_monitoring(self, interval: int = 10):
        """Start continuous health monitoring"""
        while True:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Store in Redis
            self.redis.sync_client.xadd(
                'ipl:metrics:history',
                {
                    'timestamp': metrics.timestamp,
                    'state': metrics.state,
                    'data': json.dumps(asdict(metrics))
                },
                maxlen=10000
            )
            
            # Check thresholds
            await self._check_thresholds(metrics)
            
            await asyncio.sleep(interval)
            
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics"""
        # Check component health
        primary_healthy = self.redis.check_heartbeat('scraper', max_age=30)
        backup_healthy = self.redis.check_heartbeat('vision_backup', max_age=60)
        redis_healthy = self.redis.sync_client.ping()
        
        # Get last ball timestamp
        last_ball = self.redis.sync_client.xrevrange('ipl:stream:balls:*', count=1)
        last_ball_ts = float(last_ball[0][1].get('timestamp', 0)) if last_ball else 0
        
        # Calculate data latency
        data_latency = time.time() - last_ball_ts if last_ball_ts > 0 else float('inf')
        
        # Get inference latency from recent predictions
        recent_pred = self.redis.sync_client.xrevrange('ipl:predictions:history:*', count=1)
        inference_latency = 0
        if recent_pred:
            pred_data = recent_pred[0][1]
            inference_latency = float(pred_data.get('inference_time_ms', 0))
        
        # Determine system state
        if primary_healthy and redis_healthy and data_latency < 60:
            state = SystemState.HEALTHY
        elif backup_healthy and data_latency < 120:
            state = SystemState.DEGRADED
        elif backup_healthy:
            state = SystemState.FAILOVER
        else:
            state = SystemState.DOWN
        
        # Calculate uptime
        uptime = self._calculate_uptime()
        
        return HealthMetrics(
            timestamp=time.time(),
            state=state.value,
            primary_healthy=primary_healthy,
            backup_healthy=backup_healthy,
            redis_healthy=redis_healthy,
            last_ball_timestamp=last_ball_ts,
            data_latency_seconds=data_latency,
            inference_latency_ms=inference_latency,
            error_count=0,  # Would track from logs
            uptime_seconds=uptime
        )
        
    def _calculate_uptime(self) -> float:
        """Calculate system uptime"""
        # Get first heartbeat timestamp
        first_beat = self.redis.sync_client.hget('ipl:heartbeat:scraper', 'timestamp')
        if first_beat:
            return time.time() - float(first_beat)
        return 0
        
    async def _check_thresholds(self, metrics: HealthMetrics):
        """Check metrics against thresholds and trigger alerts"""
        alerts = []
        
        if metrics.data_latency_seconds > self.thresholds['max_data_latency']:
            alerts.append(f"Data latency critical: {metrics.data_latency_seconds:.1f}s")
            
        if metrics.inference_latency_ms > self.thresholds['max_inference_latency']:
            alerts.append(f"Inference latency high: {metrics.inference_latency_ms:.1f}ms")
            
        if metrics.state == SystemState.DOWN.value:
            alerts.append("SYSTEM DOWN - All components failed")
            
        for alert in alerts:
            await self._send_alert(alert)
            
    async def _send_alert(self, message: str):
        """Send alert through all registered channels"""
        logger.warning(f"ALERT: {message}")
        
        # Store in Redis
        self.redis.sync_client.xadd(
            'ipl:alerts',
            {
                'timestamp': time.time(),
                'message': message,
                'severity': 'critical' if 'DOWN' in message else 'warning'
            },
            maxlen=1000
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


class SelfHealingManager:
    """
    Implements self-healing logic for automatic recovery.
    Restarts failed components and manages failover.
    """
    
    def __init__(self):
        self.redis = RedisStateManager()
        self.redis.connect()
        self.pm2 = PM2ProcessManager()
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        
    async def start_healing_loop(self, interval: int = 30):
        """Start self-healing monitoring loop"""
        while True:
            await self._check_and_heal()
            await asyncio.sleep(interval)
            
    async def _check_and_heal(self):
        """Check component health and attempt recovery"""
        # Get PM2 status
        pm2_status = self.pm2.get_status()
        
        for process_name, status in pm2_status.items():
            if status['status'] != 'online':
                logger.warning(f"Process {process_name} is {status['status']}")
                await self._attempt_recovery(process_name)
                
        # Check Redis health
        if not self.redis.sync_client.ping():
            logger.error("Redis connection lost - attempting recovery")
            await self._recover_redis()
            
    async def _attempt_recovery(self, process_name: str):
        """Attempt to recover a failed process"""
        attempts = self.recovery_attempts.get(process_name, 0)
        
        if attempts >= self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {process_name}")
            await self._escalate_recovery(process_name)
            return
            
        logger.info(f"Attempting recovery of {process_name} (attempt {attempts + 1})")
        
        # Restart process
        self.pm2.restart_process(process_name)
        
        self.recovery_attempts[process_name] = attempts + 1
        
        # Wait and verify
        await asyncio.sleep(10)
        
        status = self.pm2.get_status()
        if status.get(process_name, {}).get('status') == 'online':
            logger.info(f"Process {process_name} recovered successfully")
            self.recovery_attempts[process_name] = 0
            
    async def _recover_redis(self):
        """Attempt to recover Redis connection"""
        # Try to reconnect
        if self.redis.connect():
            logger.info("Redis reconnected")
        else:
            logger.error("Redis recovery failed - activating backup")
            # Would activate backup Redis instance
            
    async def _escalate_recovery(self, process_name: str):
        """Escalate recovery to human intervention"""
        # Log for manual intervention
        logger.critical(f"CRITICAL: Process {process_name} requires manual intervention")
        
        # Store escalation in Redis
        self.redis.sync_client.xadd(
            'ipl:escalations',
            {
                'timestamp': time.time(),
                'process': process_name,
                'message': f'Max recovery attempts exceeded for {process_name}'
            },
            maxlen=100
        )


# ==================== PM2 ECOSYSTEM CONFIG ====================

ECOSYSTEM_CONFIG = """
module.exports = {
  apps: [
    {
      name: 'ipl-scraper',
      script: './task1_websocket_pipeline.py',
      instances: 1,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      env: {
        NODE_ENV: 'production',
        REDIS_HOST: 'localhost',
        REDIS_PORT: 6379
      },
      error_file: './logs/scraper-error.log',
      out_file: './logs/scraper-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      max_memory_restart: '1G',
      kill_timeout: 5000,
      listen_timeout: 10000,
    },
    {
      name: 'ipl-vision',
      script: './task2_vision_backup.py',
      instances: 1,
      autorestart: true,
      max_restarts: 5,
      min_uptime: '10s',
      env: {
        CUDA_VISIBLE_DEVICES: '0',
        REDIS_HOST: 'localhost'
      },
      error_file: './logs/vision-error.log',
      out_file: './logs/vision-out.log',
      max_memory_restart: '2G',
    },
    {
      name: 'ipl-predictor',
      script: './task3_hybrid_ml_model.py',
      instances: 1,
      autorestart: true,
      max_restarts: 5,
      env: {
        MODEL_PATH: './models/hybrid_ensemble',
        REDIS_HOST: 'localhost'
      },
      error_file: './logs/predictor-error.log',
      out_file: './logs/predictor-out.log',
      max_memory_restart: '4G',
    },
    {
      name: 'ipl-heartbeat',
      script: './task4_infrastructure.py',
      args: ['--mode', 'heartbeat'],
      instances: 1,
      autorestart: true,
      error_file: './logs/heartbeat-error.log',
      out_file: './logs/heartbeat-out.log',
    },
    {
      name: 'ipl-api',
      script: './api_server.py',
      instances: 2,
      exec_mode: 'cluster',
      autorestart: true,
      env: {
        PORT: 8080
      },
      error_file: './logs/api-error.log',
      out_file: './logs/api-out.log',
    }
  ],
  
  deploy: {
    production: {
      user: 'ipl-trader',
      host: ['192.168.1.50'],
      ref: 'origin/main',
      repo: 'git@github.com:yourrepo/ipl-predictor.git',
      path: '/home/ipl-trader/ipl-predictor',
      'post-deploy': 'pip install -r requirements.txt && pm2 reload ecosystem.config.js --env production'
    }
  }
};
"""


def generate_pm2_config():
    """Generate PM2 ecosystem configuration file"""
    with open('/mnt/okcomputer/output/ecosystem.config.js', 'w') as f:
        f.write(ECOSYSTEM_CONFIG)
    print("PM2 ecosystem config generated: ecosystem.config.js")


# ==================== USAGE EXAMPLE ====================

async def main():
    """Main entry point for infrastructure"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['heartbeat', 'monitor', 'healing'], 
                       default='heartbeat')
    args = parser.parse_args()
    
    if args.mode == 'heartbeat':
        heartbeat = HeartbeatManager()
        await heartbeat.start()
    elif args.mode == 'monitor':
        monitor = HealthMonitor()
        await monitor.start_monitoring()
    elif args.mode == 'healing':
        healing = SelfHealingManager()
        await healing.start_healing_loop()


if __name__ == "__main__":
    generate_pm2_config()
    # asyncio.run(main())
