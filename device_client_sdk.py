"""
Federated Learning IoT Security - Device Client SDK
Lightweight client library for IoT devices to connect to the FL platform
"""

import requests
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
import threading
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Configuration for IoT device"""
    device_id: str
    device_type: str
    name: str
    organization_id: str
    api_base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    
    # Baseline metrics for anomaly detection
    baseline_cpu: float = 10.0
    baseline_network: float = 30.0
    baseline_requests: float = 10.0
    baseline_memory: float = 50.0
    
    # Reporting intervals (seconds)
    metrics_interval: int = 30
    model_update_interval: int = 3600  # 1 hour
    
    # Local model settings
    local_model_size: int = 10  # Number of weights
    training_samples_threshold: int = 100


@dataclass
class DeviceMetrics:
    """Current device metrics"""
    cpu_usage: float
    memory_usage: float
    network_traffic: float
    request_count: int
    connection_attempts: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class LocalAnomalyDetector:
    """Local anomaly detection model running on device"""
    
    def __init__(self, baseline: Dict[str, float]):
        self.baseline = baseline
        self.weights = np.random.randn(10).tolist()  # Simple model
        self.accuracy = 0.65
        self.training_data = []
        
    def predict(self, metrics: DeviceMetrics) -> tuple[bool, float]:
        """
        Predict if metrics indicate an anomaly
        Returns: (is_anomaly, confidence)
        """
        # Calculate deviations from baseline
        cpu_dev = abs(metrics.cpu_usage - self.baseline['cpu']) / self.baseline['cpu']
        mem_dev = abs(metrics.memory_usage - self.baseline['memory']) / self.baseline['memory']
        net_dev = abs(metrics.network_traffic - self.baseline['network']) / self.baseline['network']
        
        # Simple scoring
        anomaly_score = cpu_dev * 0.4 + mem_dev * 0.3 + net_dev * 0.3
        
        is_anomaly = anomaly_score > 1.0
        confidence = min(0.95, 0.6 + anomaly_score * 0.2)
        
        return is_anomaly, confidence
    
    def train(self, metrics_batch: List[DeviceMetrics]):
        """Train local model on recent metrics"""
        if len(metrics_batch) < 10:
            return
        
        # Simple training simulation - in production use actual ML
        # Update weights based on recent patterns
        for _ in range(5):  # 5 local epochs
            # Simulate gradient descent
            noise = np.random.randn(len(self.weights)) * 0.01
            self.weights = [w + n for w, n in zip(self.weights, noise)]
        
        # Improve accuracy slightly
        self.accuracy = min(0.95, self.accuracy + 0.01)
        
        logger.info(f"Local model trained. New accuracy: {self.accuracy:.3f}")
    
    def get_model_update(self, training_samples: int) -> Dict:
        """Prepare model update for federated aggregation"""
        weights_str = json.dumps(self.weights)
        update_hash = hashlib.sha256(weights_str.encode()).hexdigest()[:16]
        
        return {
            "weights": self.weights,
            "training_samples": training_samples,
            "local_accuracy": self.accuracy,
            "update_hash": update_hash
        }
    
    def apply_global_model(self, global_weights: List[float]):
        """Apply global model weights to local model"""
        if len(global_weights) == len(self.weights):
            self.weights = global_weights
            logger.info("Global model applied to local model")


class FLIoTClient:
    """Main client class for IoT devices"""
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.session = requests.Session()
        self.is_registered = False
        self.is_running = False
        self.current_model_version = 1
        
        # Set up API headers
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        
        # Initialize local model
        baseline = {
            "cpu": config.baseline_cpu,
            "memory": config.baseline_memory,
            "network": config.baseline_network,
            "requests": config.baseline_requests
        }
        self.local_model = LocalAnomalyDetector(baseline)
        
        # Metrics buffer
        self.metrics_buffer = []
        self.max_buffer_size = 1000
        
        # Callbacks
        self.on_threat_detected: Optional[Callable] = None
        self.on_model_updated: Optional[Callable] = None
        
        # Background threads
        self.metrics_thread = None
        self.model_thread = None
    
    def register(self) -> bool:
        """Register device with the platform"""
        try:
            url = f"{self.config.api_base_url}/api/v1/devices/register"
            data = {
                "device_id": self.config.device_id,
                "device_type": self.config.device_type,
                "name": self.config.name,
                "organization_id": self.config.organization_id,
                "baseline_metrics": {
                    "cpu": self.config.baseline_cpu,
                    "network": self.config.baseline_network,
                    "requests": self.config.baseline_requests
                }
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            result = response.json()
            self.current_model_version = result.get("model_version", 1)
            self.is_registered = True
            
            logger.info(f"Device {self.config.device_id} registered successfully")
            logger.info(f"Current model version: {self.current_model_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def submit_metrics(self, metrics: DeviceMetrics) -> bool:
        """Submit metrics to platform and check for anomalies"""
        try:
            url = f"{self.config.api_base_url}/api/v1/metrics"
            data = {
                "device_id": self.config.device_id,
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "network_traffic": metrics.network_traffic,
                "request_count": metrics.request_count,
                "connection_attempts": metrics.connection_attempts
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if threat was detected
            if result.get("status") == "anomaly_detected":
                threat = result.get("threat")
                logger.warning(f"THREAT DETECTED: {threat.get('threat_type')}")
                
                if self.on_threat_detected:
                    self.on_threat_detected(threat)
            
            # Buffer metrics for local training
            self.metrics_buffer.append(metrics)
            if len(self.metrics_buffer) > self.max_buffer_size:
                self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit metrics: {e}")
            return False
    
    def local_anomaly_check(self, metrics: DeviceMetrics) -> Optional[Dict]:
        """Perform local anomaly detection"""
        is_anomaly, confidence = self.local_model.predict(metrics)
        
        if is_anomaly:
            logger.warning(f"Local anomaly detected (confidence: {confidence:.2f})")
            return {
                "type": "local_detection",
                "confidence": confidence,
                "metrics": asdict(metrics)
            }
        
        return None
    
    def train_local_model(self):
        """Train local model on buffered metrics"""
        if len(self.metrics_buffer) < self.config.training_samples_threshold:
            logger.info(f"Insufficient data for training: {len(self.metrics_buffer)}/{self.config.training_samples_threshold}")
            return
        
        logger.info("Training local model...")
        self.local_model.train(self.metrics_buffer)
    
    def submit_model_update(self) -> bool:
        """Submit local model update for federated aggregation"""
        try:
            if len(self.metrics_buffer) < self.config.training_samples_threshold:
                logger.info("Skipping model update - insufficient training data")
                return False
            
            # Train before submitting
            self.train_local_model()
            
            url = f"{self.config.api_base_url}/api/v1/model/update"
            update = self.local_model.get_model_update(len(self.metrics_buffer))
            
            data = {
                "device_id": self.config.device_id,
                "model_version": self.current_model_version,
                **update
            }
            
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Model update submitted. Pending updates: {result.get('pending_updates')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit model update: {e}")
            return False
    
    def fetch_global_model(self) -> bool:
        """Fetch and apply latest global model"""
        try:
            url = f"{self.config.api_base_url}/api/v1/model/current"
            response = self.session.get(url)
            response.raise_for_status()
            
            result = response.json()
            model = result.get("model", {})
            
            new_version = model.get("version", 1)
            if new_version > self.current_model_version:
                weights = json.loads(model.get("weights_blob", "[]"))
                self.local_model.apply_global_model(weights)
                self.current_model_version = new_version
                
                logger.info(f"Global model updated to version {new_version}")
                logger.info(f"Global accuracy: {model.get('accuracy', 0):.3f}")
                
                if self.on_model_updated:
                    self.on_model_updated(model)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to fetch global model: {e}")
            return False
    
    def _metrics_loop(self):
        """Background thread for periodic metrics submission"""
        while self.is_running:
            try:
                # In production, collect real metrics from system
                # For now, simulate with some variation
                metrics = self._collect_metrics()
                
                # Local anomaly check (fast, runs on device)
                local_threat = self.local_anomaly_check(metrics)
                
                # Submit to platform (slower, network required)
                self.submit_metrics(metrics)
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                time.sleep(self.config.metrics_interval)
    
    def _model_update_loop(self):
        """Background thread for periodic model updates"""
        while self.is_running:
            try:
                # Check for new global model
                self.fetch_global_model()
                
                # Submit local model update
                self.submit_model_update()
                
                time.sleep(self.config.model_update_interval)
                
            except Exception as e:
                logger.error(f"Error in model update loop: {e}")
                time.sleep(self.config.model_update_interval)
    
    def _collect_metrics(self) -> DeviceMetrics:
        """
        Collect current device metrics
        In production, this would read from actual system metrics
        """
        # Simulate metrics with some randomness around baseline
        import random
        
        cpu = self.config.baseline_cpu + random.uniform(-5, 5)
        memory = self.config.baseline_memory + random.uniform(-10, 10)
        network = self.config.baseline_network + random.uniform(-15, 15)
        requests = int(self.config.baseline_requests + random.uniform(-5, 5))
        connections = random.randint(0, 10)
        
        return DeviceMetrics(
            cpu_usage=max(0, cpu),
            memory_usage=max(0, memory),
            network_traffic=max(0, network),
            request_count=max(0, requests),
            connection_attempts=connections
        )
    
    def start(self):
        """Start the client (register and begin monitoring)"""
        if not self.is_registered:
            if not self.register():
                logger.error("Failed to register device. Cannot start.")
                return False
        
        self.is_running = True
        
        # Start background threads
        self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self.model_thread = threading.Thread(target=self._model_update_loop, daemon=True)
        
        self.metrics_thread.start()
        self.model_thread.start()
        
        logger.info(f"Device {self.config.device_id} started successfully")
        logger.info(f"Metrics interval: {self.config.metrics_interval}s")
        logger.info(f"Model update interval: {self.config.model_update_interval}s")
        
        return True
    
    def stop(self):
        """Stop the client"""
        self.is_running = False
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        if self.model_thread:
            self.model_thread.join(timeout=5)
        
        logger.info(f"Device {self.config.device_id} stopped")
    
    def get_status(self) -> Dict:
        """Get current device status"""
        return {
            "device_id": self.config.device_id,
            "is_registered": self.is_registered,
            "is_running": self.is_running,
            "model_version": self.current_model_version,
            "local_accuracy": self.local_model.accuracy,
            "metrics_buffered": len(self.metrics_buffer)
        }


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_client(
    device_id: str,
    device_type: str,
    name: str,
    organization_id: str,
    api_base_url: str = "http://localhost:8000",
    **kwargs
) -> FLIoTClient:
    """
    Convenience function to create a configured client
    
    Example:
        client = create_client(
            device_id="camera-001",
            device_type="camera",
            name="Front Door Camera",
            organization_id="org-001",
            baseline_cpu=20.0
        )
        client.start()
    """
    config = DeviceConfig(
        device_id=device_id,
        device_type=device_type,
        name=name,
        organization_id=organization_id,
        api_base_url=api_base_url,
        **kwargs
    )
    
    return FLIoTClient(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    config = DeviceConfig(
        device_id="test-camera-001",
        device_type="camera",
        name="Test Smart Camera",
        organization_id="org-demo-001",
        api_base_url="http://localhost:8000",
        metrics_interval=10,  # Report every 10 seconds for testing
        model_update_interval=60  # Update model every minute for testing
    )
    
    client = FLIoTClient(config)
    
    # Set up callbacks
    def on_threat(threat):
        print(f"\n🚨 THREAT ALERT: {threat['threat_type']}")
        print(f"   Severity: {threat['severity']}")
        print(f"   Confidence: {threat['confidence']}\n")
    
    def on_model_update(model):
        print(f"\n📊 MODEL UPDATED: Version {model['version']}")
        print(f"   Accuracy: {model['accuracy']:.3f}\n")
    
    client.on_threat_detected = on_threat
    client.on_model_updated = on_model_update
    
    # Start the client
    if client.start():
        print("✅ Device client started successfully!")
        print("Press Ctrl+C to stop...\n")
        
        try:
            # Keep running
            while True:
                time.sleep(5)
                status = client.get_status()
                print(f"Status: Running | Model v{status['model_version']} | "
                      f"Accuracy: {status['local_accuracy']:.3f} | "
                      f"Buffer: {status['metrics_buffered']}")
                
        except KeyboardInterrupt:
            print("\n\nStopping device...")
            client.stop()
            print("✅ Device stopped successfully")
    else:
        print("❌ Failed to start device client")