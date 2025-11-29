"""
Federated Learning IoT Security Platform - Backend Server
Main coordination server for managing devices, federated learning rounds, and threat detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np
import hashlib
import json
from collections import defaultdict
import uuid

# ============================================================================
# DATA MODELS
# ============================================================================

class DeviceType(str, Enum):
    CAMERA = "camera"
    SENSOR = "sensor"
    THERMOSTAT = "thermostat"
    LOCK = "lock"
    MEDICAL = "medical"
    INDUSTRIAL = "industrial"
    OTHER = "other"

class ThreatSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DeviceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPROMISED = "compromised"
    MAINTENANCE = "maintenance"

class DeviceRegistration(BaseModel):
    device_id: str
    device_type: DeviceType
    name: str
    organization_id: str
    location: Optional[str] = None
    baseline_metrics: Dict[str, float] = Field(
        default_factory=lambda: {"cpu": 10.0, "network": 30.0, "requests": 10.0}
    )

class DeviceMetrics(BaseModel):
    device_id: str
    timestamp: datetime
    cpu_usage: float
    network_traffic: float
    request_count: int
    connection_attempts: int
    memory_usage: float
    additional_metrics: Optional[Dict[str, Any]] = None

class ModelUpdate(BaseModel):
    device_id: str
    model_version: int
    weights: List[float]  # Simplified - in production would be actual model weights
    training_samples: int
    local_accuracy: float
    update_hash: str  # For integrity verification

class ThreatReport(BaseModel):
    device_id: str
    threat_type: str
    severity: ThreatSeverity
    confidence: float
    description: str
    metrics_snapshot: Dict[str, Any]
    timestamp: datetime

class FederatedRoundConfig(BaseModel):
    min_devices: int = 5
    aggregation_method: str = "fedavg"  # fedavg, krum, trimmed_mean
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    byzantine_tolerance: int = 2

# ============================================================================
# IN-MEMORY STORAGE (Replace with actual database in production)
# ============================================================================

class DataStore:
    def __init__(self):
        self.devices: Dict[str, Dict] = {}
        self.model_versions: List[Dict] = []
        self.pending_updates: Dict[int, List[ModelUpdate]] = defaultdict(list)
        self.threats: List[Dict] = []
        self.metrics_history: Dict[str, List[DeviceMetrics]] = defaultdict(list)
        self.current_model_version = 1
        self.current_round = 0
        
        # Initialize global model
        self.global_model = {
            "version": 1,
            "weights": self._initialize_weights(),
            "accuracy": 0.65,
            "created_at": datetime.utcnow(),
            "devices_contributed": 0
        }
    
    def _initialize_weights(self) -> List[float]:
        """Initialize random model weights - simplified for demo"""
        return np.random.randn(10).tolist()

store = DataStore()

# ============================================================================
# FEDERATED LEARNING LOGIC
# ============================================================================

class FederatedAggregator:
    """Handles aggregation of model updates from devices"""
    
    @staticmethod
    def federated_averaging(updates: List[ModelUpdate]) -> List[float]:
        """
        FedAvg: Weighted average of model updates based on training samples
        """
        total_samples = sum(u.training_samples for u in updates)
        
        if total_samples == 0:
            return store.global_model["weights"]
        
        # Weight by number of training samples
        weighted_weights = []
        num_weights = len(updates[0].weights)
        
        for i in range(num_weights):
            weighted_sum = sum(
                u.weights[i] * (u.training_samples / total_samples)
                for u in updates
            )
            weighted_weights.append(weighted_sum)
        
        return weighted_weights
    
    @staticmethod
    def krum_aggregation(updates: List[ModelUpdate], byzantine_tolerance: int = 2) -> List[float]:
        """
        Krum: Byzantine-robust aggregation
        Selects the update closest to others, excluding outliers
        """
        if len(updates) <= byzantine_tolerance:
            return FederatedAggregator.federated_averaging(updates)
        
        n = len(updates)
        f = byzantine_tolerance
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(
                    np.array(updates[i].weights) - np.array(updates[j].weights)
                )
                distances[i][j] = dist
                distances[j][i] = dist
        
        # For each update, sum distances to n-f-2 closest neighbors
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:n-f-1])  # Exclude self (0) and f furthest
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        return updates[selected_idx].weights
    
    @staticmethod
    def trimmed_mean_aggregation(updates: List[ModelUpdate], trim_ratio: float = 0.2) -> List[float]:
        """
        Trimmed Mean: Remove extreme values before averaging
        More robust to Byzantine attacks than simple averaging
        """
        num_weights = len(updates[0].weights)
        trimmed_weights = []
        
        for i in range(num_weights):
            weight_values = [u.weights[i] for u in updates]
            weight_values.sort()
            
            # Trim top and bottom
            trim_count = int(len(weight_values) * trim_ratio)
            if trim_count > 0:
                trimmed = weight_values[trim_count:-trim_count]
            else:
                trimmed = weight_values
            
            trimmed_weights.append(np.mean(trimmed))
        
        return trimmed_weights
    
    @staticmethod
    def add_differential_privacy(weights: List[float], privacy_budget: float) -> List[float]:
        """
        Add Gaussian noise for differential privacy
        Noise scale based on privacy budget (epsilon)
        """
        sensitivity = 1.0  # Simplified - would calculate based on model
        noise_scale = sensitivity / privacy_budget
        
        noise = np.random.normal(0, noise_scale, len(weights))
        return [w + n for w, n in zip(weights, noise)]

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

class AnomalyDetector:
    """Local and global anomaly detection"""
    
    @staticmethod
    def detect_anomaly(device_id: str, metrics: DeviceMetrics) -> Optional[ThreatReport]:
        """
        Detect anomalies based on baseline and historical patterns
        """
        device = store.devices.get(device_id)
        if not device:
            return None
        
        baseline = device["baseline_metrics"]
        
        # Calculate deviation from baseline
        cpu_deviation = abs(metrics.cpu_usage - baseline["cpu"]) / baseline["cpu"]
        network_deviation = abs(metrics.network_traffic - baseline["network"]) / baseline["network"]
        request_deviation = abs(metrics.request_count - baseline["requests"]) / baseline["requests"]
        
        # Anomaly scoring
        anomaly_score = (
            cpu_deviation * 0.4 +
            network_deviation * 0.4 +
            request_deviation * 0.2
        )
        
        # Thresholds
        if anomaly_score > 2.0:
            severity = ThreatSeverity.CRITICAL
            threat_type = "Severe Anomaly Detected"
        elif anomaly_score > 1.5:
            severity = ThreatSeverity.HIGH
            threat_type = "Suspicious Activity"
        elif anomaly_score > 1.0:
            severity = ThreatSeverity.MEDIUM
            threat_type = "Unusual Behavior"
        else:
            return None
        
        # Determine specific threat type
        if cpu_deviation > 2.0:
            threat_type = "Resource Hijacking (Potential Cryptomining)"
        elif network_deviation > 2.0:
            threat_type = "Data Exfiltration Attempt"
        elif request_deviation > 2.0:
            threat_type = "Botnet Activity (Scanning Detected)"
        
        return ThreatReport(
            device_id=device_id,
            threat_type=threat_type,
            severity=severity,
            confidence=min(0.95, 0.6 + anomaly_score * 0.15),
            description=f"Device metrics deviate significantly from baseline (score: {anomaly_score:.2f})",
            metrics_snapshot={
                "cpu_usage": metrics.cpu_usage,
                "network_traffic": metrics.network_traffic,
                "request_count": metrics.request_count,
                "baseline": baseline,
                "deviations": {
                    "cpu": cpu_deviation,
                    "network": network_deviation,
                    "requests": request_deviation
                }
            },
            timestamp=metrics.timestamp
        )

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Federated Learning IoT Security API",
    description="Privacy-preserving threat detection for IoT devices",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Federated Learning IoT Security Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/v1/devices/register")
async def register_device(device: DeviceRegistration):
    """Register a new IoT device"""
    
    if device.device_id in store.devices:
        raise HTTPException(status_code=400, detail="Device already registered")
    
    store.devices[device.device_id] = {
        "device_id": device.device_id,
        "device_type": device.device_type,
        "name": device.name,
        "organization_id": device.organization_id,
        "location": device.location,
        "baseline_metrics": device.baseline_metrics,
        "status": DeviceStatus.ACTIVE,
        "registered_at": datetime.utcnow(),
        "last_seen": datetime.utcnow(),
        "model_version": store.current_model_version,
        "total_contributions": 0
    }
    
    return {
        "status": "success",
        "device_id": device.device_id,
        "model_version": store.current_model_version,
        "message": "Device registered successfully"
    }

@app.get("/api/v1/devices")
async def list_devices(organization_id: Optional[str] = None):
    """List all registered devices"""
    
    devices = list(store.devices.values())
    
    if organization_id:
        devices = [d for d in devices if d["organization_id"] == organization_id]
    
    return {
        "total": len(devices),
        "devices": devices
    }

@app.get("/api/v1/devices/{device_id}")
async def get_device(device_id: str):
    """Get device details"""
    
    device = store.devices.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Include recent metrics
    recent_metrics = store.metrics_history.get(device_id, [])[-10:]
    
    return {
        "device": device,
        "recent_metrics": [m.dict() for m in recent_metrics]
    }

@app.post("/api/v1/metrics")
async def submit_metrics(metrics: DeviceMetrics):
    """Submit device metrics and check for anomalies"""
    
    device = store.devices.get(metrics.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not registered")
    
    # Store metrics
    store.metrics_history[metrics.device_id].append(metrics)
    
    # Keep only last 1000 metrics per device
    if len(store.metrics_history[metrics.device_id]) > 1000:
        store.metrics_history[metrics.device_id] = store.metrics_history[metrics.device_id][-1000:]
    
    # Update last seen
    device["last_seen"] = datetime.utcnow()
    
    # Detect anomalies
    threat = AnomalyDetector.detect_anomaly(metrics.device_id, metrics)
    
    if threat:
        threat_dict = threat.dict()
        threat_dict["id"] = str(uuid.uuid4())
        threat_dict["device_name"] = device["name"]
        store.threats.append(threat_dict)
        
        # Update device status if critical
        if threat.severity == ThreatSeverity.CRITICAL:
            device["status"] = DeviceStatus.COMPROMISED
        
        return {
            "status": "anomaly_detected",
            "threat": threat_dict
        }
    
    return {
        "status": "normal",
        "message": "Metrics recorded successfully"
    }

@app.post("/api/v1/model/update")
async def submit_model_update(update: ModelUpdate):
    """Submit local model update for federated aggregation"""
    
    device = store.devices.get(update.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not registered")
    
    # Verify update is for current or previous version
    if update.model_version not in [store.current_model_version, store.current_model_version - 1]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model version. Current version: {store.current_model_version}"
        )
    
    # Verify update hash (simplified integrity check)
    expected_hash = hashlib.sha256(
        json.dumps(update.weights).encode()
    ).hexdigest()[:16]
    
    if update.update_hash != expected_hash:
        raise HTTPException(status_code=400, detail="Invalid update hash")
    
    # Store update for aggregation
    store.pending_updates[store.current_round].append(update)
    device["total_contributions"] += 1
    device["last_contribution"] = datetime.utcnow()
    
    return {
        "status": "accepted",
        "current_round": store.current_round,
        "pending_updates": len(store.pending_updates[store.current_round])
    }

@app.get("/api/v1/model/current")
async def get_current_model():
    """Get current global model"""
    
    return {
        "model": store.global_model,
        "download_url": f"/api/v1/model/download/{store.current_model_version}"
    }

@app.post("/api/v1/federated/round")
async def trigger_federated_round(
    background_tasks: BackgroundTasks,
    config: FederatedRoundConfig
):
    """Trigger a new federated learning round"""
    
    updates = store.pending_updates[store.current_round]
    
    if len(updates) < config.min_devices:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient updates. Need {config.min_devices}, have {len(updates)}"
        )
    
    # Perform aggregation
    aggregator = FederatedAggregator()
    
    if config.aggregation_method == "fedavg":
        new_weights = aggregator.federated_averaging(updates)
    elif config.aggregation_method == "krum":
        new_weights = aggregator.krum_aggregation(updates, config.byzantine_tolerance)
    elif config.aggregation_method == "trimmed_mean":
        new_weights = aggregator.trimmed_mean_aggregation(updates)
    else:
        raise HTTPException(status_code=400, detail="Invalid aggregation method")
    
    # Add differential privacy if enabled
    if config.differential_privacy:
        new_weights = aggregator.add_differential_privacy(new_weights, config.privacy_budget)
    
    # Calculate new global accuracy (simplified)
    avg_local_accuracy = np.mean([u.local_accuracy for u in updates])
    new_accuracy = min(0.99, avg_local_accuracy + 0.02)
    
    # Update global model
    store.current_model_version += 1
    store.global_model = {
        "version": store.current_model_version,
        "weights": new_weights,
        "accuracy": new_accuracy,
        "created_at": datetime.utcnow(),
        "devices_contributed": len(updates)
    }
    
    store.model_versions.append(store.global_model.copy())
    
    # Clear pending updates and increment round
    store.pending_updates[store.current_round] = []
    store.current_round += 1
    
    return {
        "status": "success",
        "new_version": store.current_model_version,
        "accuracy": new_accuracy,
        "devices_contributed": len(updates),
        "aggregation_method": config.aggregation_method,
        "round": store.current_round
    }

@app.get("/api/v1/threats")
async def list_threats(
    severity: Optional[ThreatSeverity] = None,
    device_id: Optional[str] = None,
    limit: int = 50
):
    """List detected threats"""
    
    threats = store.threats
    
    if severity:
        threats = [t for t in threats if t["severity"] == severity]
    
    if device_id:
        threats = [t for t in threats if t["device_id"] == device_id]
    
    # Sort by timestamp descending
    threats = sorted(threats, key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "total": len(threats),
        "threats": threats[:limit]
    }

@app.get("/api/v1/stats")
async def get_statistics():
    """Get platform statistics"""
    
    active_devices = sum(1 for d in store.devices.values() if d["status"] == DeviceStatus.ACTIVE)
    compromised_devices = sum(1 for d in store.devices.values() if d["status"] == DeviceStatus.COMPROMISED)
    
    high_threats = sum(1 for t in store.threats if t["severity"] in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL])
    
    # Recent activity (last hour)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_threats = sum(1 for t in store.threats if t["timestamp"] > one_hour_ago)
    
    return {
        "devices": {
            "total": len(store.devices),
            "active": active_devices,
            "compromised": compromised_devices,
            "inactive": len(store.devices) - active_devices - compromised_devices
        },
        "model": {
            "current_version": store.current_model_version,
            "accuracy": store.global_model["accuracy"],
            "current_round": store.current_round,
            "pending_updates": len(store.pending_updates[store.current_round])
        },
        "threats": {
            "total": len(store.threats),
            "high_severity": high_threats,
            "last_hour": recent_threats
        }
    }

@app.delete("/api/v1/devices/{device_id}")
async def deregister_device(device_id: str):
    """Deregister a device"""
    
    if device_id not in store.devices:
        raise HTTPException(status_code=404, detail="Device not found")
    
    del store.devices[device_id]
    
    # Clean up associated data
    if device_id in store.metrics_history:
        del store.metrics_history[device_id]
    
    return {
        "status": "success",
        "message": f"Device {device_id} deregistered"
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the server"""
    print("🚀 Federated Learning IoT Security Platform Starting...")
    print(f"📊 Model Version: {store.current_model_version}")
    print(f"🔒 Privacy-Preserving: Enabled")
    print(f"🌐 API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)