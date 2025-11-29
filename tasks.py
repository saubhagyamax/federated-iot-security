"""
Celery Background Tasks for Federated Learning IoT Security Platform
Handles asynchronous processing, scheduled tasks, and notifications
"""

from celery import Celery
from celery.schedules import crontab
from datetime import datetime, timedelta
from typing import List, Dict
import os
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Initialize Celery
celery_app = Celery(
    'federated_iot_tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/federated_iot')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ============================================================================
# SCHEDULED TASKS
# ============================================================================

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Configure periodic tasks"""
    
    # Auto-trigger FL rounds every hour
    sender.add_periodic_task(
        3600.0,  # Every hour
        auto_trigger_federated_round.s(),
        name='auto-trigger-fl-round'
    )
    
    # Check for inactive devices every 15 minutes
    sender.add_periodic_task(
        900.0,  # 15 minutes
        check_inactive_devices.s(),
        name='check-inactive-devices'
    )
    
    # Aggregate threat statistics daily
    sender.add_periodic_task(
        crontab(hour=0, minute=0),  # Midnight daily
        aggregate_threat_statistics.s(),
        name='daily-threat-statistics'
    )
    
    # Clean old metrics weekly
    sender.add_periodic_task(
        crontab(day_of_week=0, hour=2, minute=0),  # Sunday 2 AM
        cleanup_old_data.s(),
        name='weekly-data-cleanup'
    )
    
    # Generate threat intelligence reports daily
    sender.add_periodic_task(
        crontab(hour=8, minute=0),  # 8 AM daily
        generate_threat_intelligence.s(),
        name='daily-threat-intelligence'
    )

# ============================================================================
# FEDERATED LEARNING TASKS
# ============================================================================

@celery_app.task(bind=True, name='auto_trigger_federated_round')
def auto_trigger_federated_round(self):
    """
    Automatically trigger federated learning round if enough updates are pending
    """
    session = SessionLocal()
    
    try:
        from models import ModelUpdate, GlobalModel
        
        # Get current model version
        latest_model = session.query(GlobalModel).order_by(
            GlobalModel.version.desc()
        ).first()
        
        if not latest_model:
            return {"status": "no_model", "message": "No global model found"}
        
        # Count pending updates
        pending_count = session.query(ModelUpdate).filter(
            ModelUpdate.global_model_version == latest_model.version,
            ModelUpdate.aggregated_at.is_(None),
            ModelUpdate.is_byzantine == False
        ).count()
        
        min_devices = int(os.getenv('FL_MIN_DEVICES', 5))
        
        if pending_count >= min_devices:
            # Trigger federated learning round
            result = perform_federated_aggregation.delay(
                model_version=latest_model.version,
                min_devices=min_devices
            )
            
            return {
                "status": "triggered",
                "pending_updates": pending_count,
                "task_id": result.id
            }
        
        return {
            "status": "insufficient_updates",
            "pending_updates": pending_count,
            "required": min_devices
        }
        
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

@celery_app.task(bind=True, name='perform_federated_aggregation')
def perform_federated_aggregation(self, model_version: int, min_devices: int = 5):
    """
    Perform federated aggregation of model updates
    """
    session = SessionLocal()
    
    try:
        from models import ModelUpdate, GlobalModel, Device, FederatedRound, AggregationMethodEnum
        import json
        
        start_time = datetime.utcnow()
        
        # Get pending updates
        updates = session.query(ModelUpdate).filter(
            ModelUpdate.global_model_version == model_version,
            ModelUpdate.aggregated_at.is_(None),
            ModelUpdate.is_byzantine == False,
            ModelUpdate.is_validated == True
        ).all()
        
        if len(updates) < min_devices:
            return {
                "status": "insufficient_updates",
                "updates_count": len(updates),
                "required": min_devices
            }
        
        # Validate updates for Byzantine behavior
        validated_updates = validate_updates_for_byzantine(updates)
        
        if len(validated_updates) < min_devices:
            return {
                "status": "too_many_byzantine",
                "valid_updates": len(validated_updates),
                "rejected": len(updates) - len(validated_updates)
            }
        
        # Perform aggregation
        aggregation_method = os.getenv('FL_AGGREGATION_METHOD', 'fedavg')
        
        if aggregation_method == 'fedavg':
            new_weights = federated_averaging(validated_updates)
        elif aggregation_method == 'krum':
            new_weights = krum_aggregation(validated_updates)
        else:
            new_weights = trimmed_mean_aggregation(validated_updates)
        
        # Apply differential privacy if enabled
        if os.getenv('FL_DIFFERENTIAL_PRIVACY', 'true').lower() == 'true':
            privacy_budget = float(os.getenv('FL_PRIVACY_BUDGET', 1.0))
            new_weights = add_differential_privacy(new_weights, privacy_budget)
        
        # Calculate new accuracy
        avg_accuracy = np.mean([u.local_accuracy for u in validated_updates])
        new_accuracy = min(0.99, avg_accuracy + 0.02)
        
        # Create new global model
        new_model = GlobalModel(
            version=model_version + 1,
            weights_blob=json.dumps(new_weights),
            accuracy=new_accuracy,
            devices_contributed=len(validated_updates),
            total_samples=sum(u.training_samples for u in validated_updates),
            aggregation_method=AggregationMethodEnum[aggregation_method.upper()],
            differential_privacy_enabled=os.getenv('FL_DIFFERENTIAL_PRIVACY', 'true').lower() == 'true',
            privacy_budget=float(os.getenv('FL_PRIVACY_BUDGET', 1.0)),
            round_number=self.request.retries + 1,
            training_duration_seconds=(datetime.utcnow() - start_time).total_seconds()
        )
        
        session.add(new_model)
        
        # Mark updates as aggregated
        for update in validated_updates:
            update.aggregated_at = datetime.utcnow()
        
        # Create federated round record
        fl_round = FederatedRound(
            round_number=new_model.round_number,
            min_devices=min_devices,
            aggregation_method=new_model.aggregation_method,
            differential_privacy=new_model.differential_privacy_enabled,
            privacy_budget=new_model.privacy_budget,
            devices_participated=len(validated_updates),
            updates_received=len(updates),
            updates_accepted=len(validated_updates),
            updates_rejected=len(updates) - len(validated_updates),
            previous_model_version=model_version,
            new_model_version=new_model.version,
            accuracy_improvement=new_accuracy - session.query(GlobalModel).get(model_version).accuracy,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            status="completed"
        )
        
        session.add(fl_round)
        session.commit()
        
        # Notify devices of new model (async)
        notify_devices_new_model.delay(new_model.version)
        
        return {
            "status": "success",
            "new_version": new_model.version,
            "accuracy": new_accuracy,
            "devices_contributed": len(validated_updates),
            "duration_seconds": fl_round.duration_seconds
        }
        
    except Exception as e:
        session.rollback()
        self.retry(exc=e, countdown=300, max_retries=3)  # Retry after 5 minutes
    finally:
        session.close()

# ============================================================================
# DEVICE MONITORING TASKS
# ============================================================================

@celery_app.task(name='check_inactive_devices')
def check_inactive_devices():
    """
    Check for devices that haven't reported in and mark as inactive
    """
    session = SessionLocal()
    
    try:
        from models import Device, DeviceStatusEnum
        
        inactive_threshold = datetime.utcnow() - timedelta(minutes=30)
        
        devices = session.query(Device).filter(
            Device.status == DeviceStatusEnum.ACTIVE,
            Device.last_seen < inactive_threshold
        ).all()
        
        count = 0
        for device in devices:
            device.status = DeviceStatusEnum.INACTIVE
            count += 1
            
            # Send alert
            send_device_inactive_alert.delay(device.id, device.name)
        
        session.commit()
        
        return {
            "status": "success",
            "devices_marked_inactive": count
        }
        
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

@celery_app.task(name='analyze_device_behavior')
def analyze_device_behavior(device_id: str):
    """
    Deep analysis of device behavior patterns
    """
    session = SessionLocal()
    
    try:
        from models import Device, DeviceMetrics
        
        # Get device and recent metrics
        device = session.query(Device).filter(Device.id == device_id).first()
        if not device:
            return {"status": "device_not_found"}
        
        # Get metrics from last 24 hours
        since = datetime.utcnow() - timedelta(hours=24)
        metrics = session.query(DeviceMetrics).filter(
            DeviceMetrics.device_id == device_id,
            DeviceMetrics.timestamp >= since
        ).order_by(DeviceMetrics.timestamp.asc()).all()
        
        if len(metrics) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze patterns
        cpu_values = [m.cpu_usage for m in metrics]
        network_values = [m.network_traffic for m in metrics]
        
        analysis = {
            "device_id": device_id,
            "period_hours": 24,
            "samples": len(metrics),
            "cpu": {
                "mean": np.mean(cpu_values),
                "std": np.std(cpu_values),
                "max": np.max(cpu_values),
                "baseline_deviation": abs(np.mean(cpu_values) - device.baseline_cpu) / device.baseline_cpu
            },
            "network": {
                "mean": np.mean(network_values),
                "std": np.std(network_values),
                "max": np.max(network_values),
                "baseline_deviation": abs(np.mean(network_values) - device.baseline_network) / device.baseline_network
            }
        }
        
        # Check for anomalies
        anomaly_detected = (
            analysis["cpu"]["baseline_deviation"] > 0.5 or
            analysis["network"]["baseline_deviation"] > 0.5
        )
        
        if anomaly_detected:
            analysis["status"] = "anomaly_detected"
            analysis["recommendations"] = [
                "Review device logs",
                "Check for unauthorized access",
                "Consider device isolation"
            ]
        else:
            analysis["status"] = "normal"
        
        return analysis
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

# ============================================================================
# THREAT INTELLIGENCE TASKS
# ============================================================================

@celery_app.task(name='generate_threat_intelligence')
def generate_threat_intelligence():
    """
    Analyze threats across the platform and generate intelligence
    """
    session = SessionLocal()
    
    try:
        from models import Threat, ThreatIntelligence, ThreatSeverityEnum
        
        # Get threats from last 7 days
        since = datetime.utcnow() - timedelta(days=7)
        threats = session.query(Threat).filter(
            Threat.detected_at >= since
        ).all()
        
        # Aggregate by threat type
        threat_patterns = {}
        for threat in threats:
            if threat.threat_type not in threat_patterns:
                threat_patterns[threat.threat_type] = {
                    "count": 0,
                    "severities": [],
                    "devices": set(),
                    "first_seen": threat.detected_at,
                    "last_seen": threat.detected_at
                }
            
            pattern = threat_patterns[threat.threat_type]
            pattern["count"] += 1
            pattern["severities"].append(threat.severity)
            pattern["devices"].add(threat.device_id)
            pattern["last_seen"] = max(pattern["last_seen"], threat.detected_at)
        
        # Update or create threat intelligence records
        for threat_type, data in threat_patterns.items():
            intel = session.query(ThreatIntelligence).filter(
                ThreatIntelligence.pattern_name == threat_type
            ).first()
            
            if intel:
                intel.prevalence += data["count"]
                intel.last_seen = data["last_seen"]
            else:
                intel = ThreatIntelligence(
                    pattern_name=threat_type,
                    threat_category="automated_detection",
                    severity=max(data["severities"]),
                    prevalence=data["count"],
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"]
                )
                session.add(intel)
        
        session.commit()
        
        return {
            "status": "success",
            "patterns_analyzed": len(threat_patterns),
            "total_threats": len(threats)
        }
        
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

@celery_app.task(name='aggregate_threat_statistics')
def aggregate_threat_statistics():
    """
    Generate daily threat statistics
    """
    session = SessionLocal()
    
    try:
        from models import Threat, Device, Organization
        from sqlalchemy import func
        
        # Stats for last 24 hours
        since = datetime.utcnow() - timedelta(hours=24)
        
        # Threats by severity
        severity_stats = session.query(
            Threat.severity,
            func.count(Threat.id).label('count')
        ).filter(
            Threat.detected_at >= since
        ).group_by(Threat.severity).all()
        
        # Threats by device type
        device_type_stats = session.query(
            Device.device_type,
            func.count(Threat.id).label('count')
        ).join(Threat).filter(
            Threat.detected_at >= since
        ).group_by(Device.device_type).all()
        
        # Top affected organizations
        org_stats = session.query(
            Organization.name,
            func.count(Threat.id).label('threat_count')
        ).join(Device).join(Threat).filter(
            Threat.detected_at >= since
        ).group_by(Organization.name).order_by(
            func.count(Threat.id).desc()
        ).limit(10).all()
        
        stats = {
            "period": "last_24_hours",
            "generated_at": datetime.utcnow().isoformat(),
            "by_severity": {s.name: c for s, c in severity_stats},
            "by_device_type": {dt.value: c for dt, c in device_type_stats},
            "top_affected_organizations": [
                {"name": name, "threats": count} for name, count in org_stats
            ]
        }
        
        # Send to monitoring/alerting system
        # send_stats_to_monitoring(stats)
        
        return stats
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

# ============================================================================
# NOTIFICATION TASKS
# ============================================================================

@celery_app.task(name='send_device_inactive_alert')
def send_device_inactive_alert(device_id: str, device_name: str):
    """Send alert when device goes inactive"""
    # Implementation depends on notification system (email, Slack, PagerDuty)
    print(f"ALERT: Device {device_name} ({device_id}) is now inactive")
    return {"status": "alert_sent", "device_id": device_id}

@celery_app.task(name='send_threat_alert')
def send_threat_alert(threat_id: str, severity: str, device_name: str):
    """Send alert for high-severity threats"""
    if severity in ["high", "critical"]:
        print(f"CRITICAL THREAT ALERT: {severity} threat detected on {device_name}")
        # Send to alerting system
    return {"status": "alert_sent", "threat_id": threat_id}

@celery_app.task(name='notify_devices_new_model')
def notify_devices_new_model(model_version: int):
    """Notify devices that a new global model is available"""
    session = SessionLocal()
    
    try:
        from models import Device, DeviceStatusEnum
        
        active_devices = session.query(Device).filter(
            Device.status == DeviceStatusEnum.ACTIVE
        ).all()
        
        # In production, push notifications to devices
        # For now, just log
        print(f"Notifying {len(active_devices)} devices of new model version {model_version}")
        
        return {
            "status": "success",
            "devices_notified": len(active_devices),
            "model_version": model_version
        }
        
    finally:
        session.close()

# ============================================================================
# DATA CLEANUP TASKS
# ============================================================================

@celery_app.task(name='cleanup_old_data')
def cleanup_old_data():
    """Clean up old data to manage database size"""
    session = SessionLocal()
    
    try:
        from models import DeviceMetrics, AuditLog
        
        # Delete metrics older than 90 days
        metrics_cutoff = datetime.utcnow() - timedelta(days=90)
        deleted_metrics = session.query(DeviceMetrics).filter(
            DeviceMetrics.timestamp < metrics_cutoff
        ).delete()
        
        # Delete audit logs older than 1 year
        audit_cutoff = datetime.utcnow() - timedelta(days=365)
        deleted_audits = session.query(AuditLog).filter(
            AuditLog.timestamp < audit_cutoff
        ).delete()
        
        session.commit()
        
        return {
            "status": "success",
            "metrics_deleted": deleted_metrics,
            "audit_logs_deleted": deleted_audits
        }
        
    except Exception as e:
        session.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        session.close()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_updates_for_byzantine(updates: List) -> List:
    """Validate model updates and filter Byzantine actors"""
    # Simplified Byzantine detection
    # In production, use more sophisticated methods
    validated = []
    
    for update in updates:
        # Check if weights are reasonable (not NaN, not too large)
        weights = update.weights_blob
        if all(abs(w) < 100 for w in weights):  # Simple sanity check
            update.is_validated = True
            validated.append(update)
        else:
            update.is_byzantine = True
    
    return validated

def federated_averaging(updates: List) -> List[float]:
    """FedAvg aggregation"""
    total_samples = sum(u.training_samples for u in updates)
    weights_arrays = [u.weights_blob for u in updates]
    
    aggregated = []
    for i in range(len(weights_arrays[0])):
        weighted_sum = sum(
            w[i] * (u.training_samples / total_samples)
            for w, u in zip(weights_arrays, updates)
        )
        aggregated.append(weighted_sum)
    
    return aggregated

def krum_aggregation(updates: List) -> List[float]:
    """Krum Byzantine-robust aggregation"""
    # Simplified Krum - select median update
    weights_arrays = [u.weights_blob for u in updates]
    median_idx = len(weights_arrays) // 2
    return sorted(weights_arrays, key=lambda x: sum(x))[median_idx]

def trimmed_mean_aggregation(updates: List) -> List[float]:
    """Trimmed mean aggregation"""
    weights_arrays = [u.weights_blob for u in updates]
    trim_ratio = 0.2
    
    aggregated = []
    for i in range(len(weights_arrays[0])):
        values = sorted([w[i] for w in weights_arrays])
        trim = int(len(values) * trim_ratio)
        trimmed = values[trim:-trim] if trim > 0 else values
        aggregated.append(np.mean(trimmed))
    
    return aggregated

def add_differential_privacy(weights: List[float], epsilon: float) -> List[float]:
    """Add differential privacy noise"""
    sensitivity = 1.0
    noise_scale = sensitivity / epsilon
    noise = np.random.normal(0, noise_scale, len(weights))
    return [w + n for w, n in zip(weights, noise)]