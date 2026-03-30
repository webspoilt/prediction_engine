"""
TASK 2: VISION-BASED BACKUP SYSTEM (THE "SAFETY NET")
IPL Win Probability Prediction Engine - Computer Vision Module

This module provides a backup data ingestion system using OpenCV + YOLOv8
to extract scoreboard data from live stream captures.

Hardware: Asus TUF (i5 10th Gen, 16GB RAM, GTX 1650 Ti)
Optimization: GPU acceleration with thermal throttling protection
"""

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image
import logging
import time
import threading
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import json

# YOLOv8 for scoreboard detection
from ultralytics import YOLO

# For GPU monitoring
import pynvml

# Redis for data publishing
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoreboardData:
    """Extracted scoreboard data"""
    runs: int
    wickets: int
    overs: float
    crr: float
    rrr: Optional[float]
    target: Optional[int]
    batting_team: str
    timestamp: float
    confidence: float


class GPUMonitor:
    """Monitor GTX 1650 Ti to prevent thermal throttling"""
    
    def __init__(self, temp_threshold: int = 75, utilization_threshold: int = 90):
        self.temp_threshold = temp_threshold
        self.utilization_threshold = utilization_threshold
        self.is_throttling = False
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
            logger.info("GPU monitoring initialized")
        except Exception as e:
            logger.warning(f"GPU monitoring not available: {e}")
            self.available = False
    
    def get_stats(self) -> Dict:
        """Get current GPU statistics"""
        if not self.available:
            return {'available': False}
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'available': True,
                'temperature': temp,
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used': memory.used / 1024**2,  # MB
                'memory_total': memory.total / 1024**2,  # MB
                'is_throttling': temp > self.temp_threshold
            }
        except Exception as e:
            logger.error(f"GPU stats error: {e}")
            return {'available': False, 'error': str(e)}
    
    def should_throttle(self) -> bool:
        """Check if processing should be throttled"""
        if not self.available:
            return False
        
        stats = self.get_stats()
        return stats.get('temperature', 0) > self.temp_threshold


class ScoreboardDetector:
    """
    YOLOv8-based scoreboard region detection.
    Trained to identify scoreboard regions in cricket broadcast frames.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', use_gpu: bool = True):
        self.gpu_monitor = GPUMonitor()
        
        # Load YOLOv8 model
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        logger.info(f"Loading YOLOv8 on {self.device}")
        
        # Use nano model for speed on GTX 1650 Ti
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        
        # Scoreboard class (would be trained specifically for cricket scoreboards)
        self.scoreboard_class = 'scoreboard'
        
        # Detection parameters
        self.conf_threshold = 0.6
        self.iou_threshold = 0.45
        
    def detect_scoreboard(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect scoreboard region in frame.
        Returns (x1, y1, x2, y2) bounding box or None.
        """
        # Check GPU temperature
        if self.gpu_monitor.should_throttle():
            logger.warning("GPU throttling - reducing inference frequency")
            time.sleep(0.5)
        
        # Run YOLOv8 inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )
        
        # Extract scoreboard bounding box
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if this is a scoreboard detection
                # In production, use custom trained model
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Filter by aspect ratio (scoreboards are typically wide)
                aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
                if 2.0 < aspect_ratio < 8.0 and conf > self.conf_threshold:
                    return (x1, y1, x2, y2)
        
        return None
    
    def detect_scoreboard_heuristic(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback heuristic method for scoreboard detection.
        Uses traditional CV techniques when YOLO fails.
        """
        height, width = frame.shape[:2]
        
        # Scoreboards are typically in top portion of frame
        roi = frame[0:int(height*0.25), :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular regions with text-like characteristics
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            
            # Filter for wide rectangles
            if w > width * 0.3 and 2.0 < aspect_ratio < 8.0:
                return (x, y, x + w, y + h)
        
        return None


class ScoreboardOCR:
    """
    OCR engine for extracting text from scoreboard regions.
    Optimized for cricket scoreboard text (numbers and team names).
    """
    
    def __init__(self):
        # Configure Tesseract for scoreboard reading
        self.tesseract_config = (
            '--oem 3 '  # LSTM engine
            '--psm 7 '  # Single text line
            '-c tessedit_char_whitelist=0123456789./-ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        )
        
        # Preprocessing parameters
        self.denoise_strength = 10
        self.contrast_alpha = 1.5
        self.contrast_beta = 0
        
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal OCR"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for better OCR (scale up small regions)
        scale_factor = 2
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength, 7, 21)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=self.contrast_alpha, 
                                       beta=self.contrast_beta)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from preprocessed image"""
        preprocessed = self.preprocess_for_ocr(image)
        
        # Run OCR
        text = pytesseract.image_to_string(
            preprocessed, 
            config=self.tesseract_config
        )
        
        return text.strip()
    
    def extract_score(self, scoreboard_img: np.ndarray) -> Dict:
        """
        Extract score components from scoreboard image.
        Assumes standard cricket scoreboard layout.
        """
        height, width = scoreboard_img.shape[:2]
        
        # Define regions of interest (adjust based on broadcast format)
        regions = {
            'runs_wickets': (int(width*0.1), int(height*0.2), int(width*0.3), int(height*0.6)),
            'overs': (int(width*0.35), int(height*0.2), int(width*0.5), int(height*0.6)),
            'crr': (int(width*0.55), int(height*0.2), int(width*0.7), int(height*0.6)),
            'rrr': (int(width*0.75), int(height*0.2), int(width*0.9), int(height*0.6)),
        }
        
        results = {}
        
        for key, (x1, y1, x2, y2) in regions.items():
            roi = scoreboard_img[y1:y2, x1:x2]
            if roi.size > 0:
                text = self.extract_text(roi)
                results[key] = text
        
        return results
    
    def parse_score_text(self, text_dict: Dict) -> ScoreboardData:
        """Parse OCR text into structured scoreboard data"""
        try:
            # Parse runs/wickets (format: "150/3" or "150-3")
            runs_wickets = text_dict.get('runs_wickets', '0/0')
            runs_wickets = runs_wickets.replace('-', '/')
            parts = runs_wickets.split('/')
            runs = int(parts[0]) if parts[0].isdigit() else 0
            wickets = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            
            # Parse overs (format: "15.3")
            overs_text = text_dict.get('overs', '0.0')
            try:
                overs = float(overs_text)
            except ValueError:
                overs = 0.0
            
            # Parse CRR
            crr_text = text_dict.get('crr', '0.0')
            try:
                crr = float(crr_text)
            except ValueError:
                crr = runs / max(overs, 0.1)
            
            # Parse RRR (second innings only)
            rrr_text = text_dict.get('rrr', '')
            rrr = None
            try:
                rrr = float(rrr_text) if rrr_text else None
            except ValueError:
                pass
            
            return ScoreboardData(
                runs=runs,
                wickets=wickets,
                overs=overs,
                crr=crr,
                rrr=rrr,
                target=None,  # Would need additional parsing
                batting_team='',  # Would need team recognition
                timestamp=time.time(),
                confidence=0.8  # Placeholder confidence
            )
            
        except Exception as e:
            logger.error(f"Score parsing error: {e}")
            return ScoreboardData(
                runs=0, wickets=0, overs=0.0, crr=0.0,
                rrr=None, target=None, batting_team='',
                timestamp=time.time(), confidence=0.0
            )


class StreamCapture:
    """
    Capture video stream from various sources.
    Supports screen capture, video files, and streaming URLs.
    """
    
    def __init__(self, source: str = 'screen', region: Optional[Tuple] = None):
        self.source = source
        self.region = region  # (x, y, w, h) for screen capture
        self.capture = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=30)  # 1 second at 30fps
        
    def start(self):
        """Start video capture"""
        if self.source == 'screen':
            self._start_screen_capture()
        else:
            self._start_video_capture()
            
    def _start_screen_capture(self):
        """Start screen capture using mss or d3d"""
        try:
            import mss
            self.sct = mss.mss()
            self.is_running = True
            logger.info("Screen capture started")
        except ImportError:
            logger.error("mss not installed. Install with: pip install mss")
            raise
            
    def _start_video_capture(self):
        """Start video file/stream capture"""
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")
        self.is_running = True
        logger.info(f"Video capture started: {self.source}")
        
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame"""
        if self.source == 'screen':
            return self._read_screen_frame()
        else:
            return self._read_video_frame()
            
    def _read_screen_frame(self) -> Optional[np.ndarray]:
        """Read frame from screen"""
        try:
            if self.region:
                monitor = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3]
                }
            else:
                monitor = self.sct.monitors[1]  # Primary monitor
            
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
            
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
            
    def _read_video_frame(self) -> Optional[np.ndarray]:
        """Read frame from video source"""
        ret, frame = self.capture.read()
        return frame if ret else None
        
    def stop(self):
        """Stop capture"""
        self.is_running = False
        if self.capture:
            self.capture.release()
        logger.info("Capture stopped")


class VisionPipeline:
    """
    Complete vision-based data extraction pipeline.
    Integrates detection, OCR, and data publishing.
    """
    
    def __init__(self, 
                 capture_source: str = 'screen',
                 capture_region: Optional[Tuple] = None,
                 process_interval: float = 5.0):
        self.capture = StreamCapture(capture_source, capture_region)
        self.detector = ScoreboardDetector()
        self.ocr = ScoreboardOCR()
        self.redis_client = redis.Redis(decode_responses=True)
        
        self.process_interval = process_interval
        self.is_running = False
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_process_time = 0
        
    def start(self):
        """Start the vision pipeline"""
        self.is_running = True
        self.capture.start()
        
        logger.info("Vision pipeline started")
        
        # Main processing loop
        while self.is_running:
            loop_start = time.time()
            
            # Read frame
            frame = self.capture.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # Process at specified intervals
            if time.time() - self.last_process_time >= self.process_interval:
                self._process_frame(frame)
                self.last_process_time = time.time()
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1/30) - elapsed)  # Target 30fps capture
            time.sleep(sleep_time)
            
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame for score extraction"""
        try:
            # Detect scoreboard
            bbox = self.detector.detect_scoreboard(frame)
            
            if bbox is None:
                # Try heuristic fallback
                bbox = self.detector.detect_scoreboard_heuristic(frame)
            
            if bbox:
                self.detection_count += 1
                
                # Extract scoreboard region
                x1, y1, x2, y2 = bbox
                scoreboard_img = frame[y1:y2, x1:x2]
                
                # Run OCR
                ocr_results = self.ocr.extract_score(scoreboard_img)
                score_data = self.ocr.parse_score_text(ocr_results)
                
                # Publish to Redis
                self._publish_score(score_data)
                
                logger.info(f"Extracted score: {score_data.runs}/{score_data.wickets} "
                           f"at {score_data.overs} overs (conf: {score_data.confidence:.2f})")
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            
    def _publish_score(self, score_data: ScoreboardData):
        """Publish extracted score to Redis"""
        # Create unique match ID (would be passed as parameter in production)
        match_id = 'vision_backup_match'
        
        # Convert to dict
        data = {
            'runs': score_data.runs,
            'wickets': score_data.wickets,
            'overs': score_data.overs,
            'crr': score_data.crr,
            'rrr': score_data.rrr,
            'target': score_data.target,
            'timestamp': score_data.timestamp,
            'confidence': score_data.confidence,
            'source': 'vision'
        }
        
        # Publish to Redis Stream
        stream_key = f"ipl:vision:{match_id}"
        self.redis_client.xadd(stream_key, data, maxlen=100)
        
        # Publish to pub/sub
        self.redis_client.publish('ipl:vision:updates', json.dumps(data))
        
    def stop(self):
        """Stop the pipeline"""
        self.is_running = False
        self.capture.stop()
        
        # Log performance stats
        detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
        logger.info(f"Vision pipeline stopped. Detection rate: {detection_rate:.1f}%")


class ThermalManager:
    """
    Manages GPU temperature to prevent throttling on GTX 1650 Ti.
    Implements dynamic frequency scaling and cooling pauses.
    """
    
    def __init__(self, 
                 temp_warning: int = 70,
                 temp_critical: int = 80,
                 cooldown_period: int = 10):
        self.gpu_monitor = GPUMonitor()
        self.temp_warning = temp_warning
        self.temp_critical = temp_critical
        self.cooldown_period = cooldown_period
        self.is_cooling = False
        
    def check_and_adjust(self) -> bool:
        """
        Check GPU temperature and adjust processing if needed.
        Returns True if processing should continue, False if cooling.
        """
        if not self.gpu_monitor.available:
            return True
            
        stats = self.gpu_monitor.get_stats()
        temp = stats.get('temperature', 0)
        
        if temp >= self.temp_critical:
            logger.warning(f"CRITICAL GPU TEMP: {temp}°C - Initiating cooldown")
            self.is_cooling = True
            time.sleep(self.cooldown_period)
            self.is_cooling = False
            return False
            
        elif temp >= self.temp_warning:
            logger.warning(f"High GPU temp: {temp}°C - Throttling processing")
            return False
            
        return True
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on GPU temperature"""
        if not self.gpu_monitor.available:
            return 1
            
        stats = self.gpu_monitor.get_stats()
        temp = stats.get('temperature', 0)
        
        if temp > 75:
            return 1
        elif temp > 65:
            return 2
        else:
            return 4


# ==================== USAGE EXAMPLE ====================

def main():
    """Example usage of vision backup system"""
    
    # Define screen region for scoreboard (adjust based on your stream layout)
    # Format: (x, y, width, height)
    scoreboard_region = (100, 50, 800, 150)
    
    # Initialize pipeline
    pipeline = VisionPipeline(
        capture_source='screen',
        capture_region=scoreboard_region,
        process_interval=5.0  # Process every 5 seconds
    )
    
    try:
        pipeline.start()
    except KeyboardInterrupt:
        logger.info("Stopping vision pipeline...")
        pipeline.stop()


if __name__ == "__main__":
    main()
