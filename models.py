"""
Database Models for Federated Learning IoT Security Platform
Using SQLAlchemy ORM with PostgreSQL
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    JSON, ForeignKey, Enum, Text, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

# ============================================================================
# ENUMS
# ============================================================================

class DeviceTypeEnum(enum.Enum):
    CAMERA = "camera"
    SENSOR = "sensor"
    THERMOSTAT = "thermostat"
    LOCK = "lock"
    MEDICAL = "medical"
    INDUSTRIAL = "industrial"
    OTHER = "other"

class DeviceStatusEnum(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPROMISED = "compromised"
    MAINTENANCE = "maintenance"

class ThreatSeverityEnum(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AggregationMethodEnum(enum.Enum):
    FEDAVG = "fedavg"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"

# ============================================================================
# MODELS
# ============================================================================

class Organization(Base):
    """Organization/Company using the platform"""
    __tablename__ = "organizations"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    industry = Column(String(100))
    contact_email = Column(String(255))
    api_key = Column(String(255), unique=True, index=True)
    
    # Subscription info
    subscription_tier = Column(String(50), default="free")  # free, basic, enterprise
    max_devices = Column(Integer, default=10)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    devices = relationship("Device", back_populates="organization", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_org_api_key', 'api_key'),
    )

class Device(Base):
    """IoT Device registered in the system"""
    __tablename__ = "devices"
    
    id = Column(String(36), primary_key=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    
    # Device info
    name = Column(String(255), nullable=False)
    device_type = Column(Enum(DeviceTypeEnum), nullable=False)
    status = Column(Enum(DeviceStatusEnum), default=DeviceStatusEnum.ACTIVE)
    
    # Location
    location = Column(String(255))
    ip_address = Column(String(45))
    mac_address = Column(String(17))
    
    # Baseline metrics for anomaly detection
    baseline_cpu = Column(Float, default=10.0)
    baseline_network = Column(Float, default=30.0)
    baseline_requests = Column(Float, default=10.0)
    baseline_memory = Column(Float, default=50.0)
    
    # Model info
    current_model_version = Column(Integer, default=1)
    local_model_accuracy = Column(Float)
    total_contributions = Column(Integer, default=0)
    last_contribution_at = Column(DateTime(timezone=True))
    
    # Metadata
    firmware_version = Column(String(50))
    hardware_info = Column(JSON)
    custom_metadata = Column(JSON)
    
    # Timestamps
    registered_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="devices")
    metrics = relationship("DeviceMetrics", back_populates="device", cascade="all, delete-orphan")
    threats = relationship("Threat", back_populates="device", cascade="all, delete-orphan")
    model_updates = relationship("ModelUpdate", back_populates="device", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_device_org', 'organization_id'),
        Index('idx_device_status', 'status'),
        Index('idx_device_last_seen', 'last_seen'),
    )

class DeviceMetrics(Base):
    """Time-series metrics from devices"""
    __tablename__ = "device_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(36), ForeignKey("devices.id"), nullable=False)
    
    # Metrics
    cpu_usage = Column(Float, nullable=False)
    memory_usage = Column(Float)
    network_traffic = Column(Float, nullable=False)
    request_count = Column(Integer, nullable=False)
    connection_attempts = Column(Integer, default=0)
    
    # Additional metrics (flexible)
    disk_usage = Column(Float)
    temperature = Column(Float)
    custom_metrics = Column(JSON)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Relationships
    device = relationship("Device", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metrics_device_time', 'device_id', 'timestamp'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )

class GlobalModel(Base):
    """Global federated model versions"""
    __tablename__ = "global_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False, index=True)
    
    # Model data (in production, store reference to model file)
    weights_blob = Column(Text)  # JSON serialized weights
    model_architecture = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training info
    devices_contributed = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    aggregation_method = Column(Enum(AggregationMethodEnum))
    differential_privacy_enabled = Column(Boolean, default=True)
    privacy_budget = Column(Float)
    
    # Round info
    round_number = Column(Integer)
    training_duration_seconds = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model_updates = relationship("ModelUpdate", back_populates="global_model")
    
    __table_args__ = (
        Index('idx_model_version', 'version'),
    )

class ModelUpdate(Base):
    """Individual model updates from devices"""
    __tablename__ = "model_updates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(36), ForeignKey("devices.id"), nullable=False)
    global_model_version = Column(Integer, ForeignKey("global_models.version"), nullable=False)
    
    # Update data
    weights_blob = Column(Text)  # JSON serialized weights
    update_hash = Column(String(64), nullable=False)
    
    # Training info
    training_samples = Column(Integer, nullable=False)
    local_accuracy = Column(Float)
    local_loss = Column(Float)
    epochs_trained = Column(Integer)
    
    # Validation
    is_validated = Column(Boolean, default=False)
    is_byzantine = Column(Boolean, default=False)  # Flagged as malicious
    validation_score = Column(Float)
    
    # Timestamps
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())
    aggregated_at = Column(DateTime(timezone=True))
    
    # Relationships
    device = relationship("Device", back_populates="model_updates")
    global_model = relationship("GlobalModel", back_populates="model_updates")
    
    __table_args__ = (
        Index('idx_update_device_version', 'device_id', 'global_model_version'),
        Index('idx_update_submitted', 'submitted_at'),
    )

class Threat(Base):
    """Detected security threats"""
    __tablename__ = "threats"
    
    id = Column(String(36), primary_key=True)
    device_id = Column(String(36), ForeignKey("devices.id"), nullable=False)
    
    # Threat details
    threat_type = Column(String(100), nullable=False)
    severity = Column(Enum(ThreatSeverityEnum), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    description = Column(Text)
    
    # Detection info
    detection_method = Column(String(50))  # local_model, global_model, rule_based
    model_version = Column(Integer)
    
    # Metrics snapshot at time of detection
    metrics_snapshot = Column(JSON)
    
    # Response
    is_acknowledged = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), nullable=False, index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    
    # Relationships
    device = relationship("Device", back_populates="threats")
    
    __table_args__ = (
        Index('idx_threat_severity_time', 'severity', 'detected_at'),
        Index('idx_threat_device', 'device_id'),
        Index('idx_threat_unresolved', 'is_resolved', 'severity'),
    )

class FederatedRound(Base):
    """Federated learning training rounds"""
    __tablename__ = "federated_rounds"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, unique=True, nullable=False, index=True)
    
    # Configuration
    min_devices = Column(Integer, default=5)
    aggregation_method = Column(Enum(AggregationMethodEnum), nullable=False)
    differential_privacy = Column(Boolean, default=True)
    privacy_budget = Column(Float)
    byzantine_tolerance = Column(Integer)
    
    # Results
    devices_participated = Column(Integer)
    updates_received = Column(Integer)
    updates_accepted = Column(Integer)
    updates_rejected = Column(Integer)
    
    # Model improvements
    previous_model_version = Column(Integer)
    new_model_version = Column(Integer)
    accuracy_improvement = Column(Float)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Status
    status = Column(String(20), default="in_progress")  # in_progress, completed, failed
    
    __table_args__ = (
        Index('idx_round_number', 'round_number'),
        Index('idx_round_status', 'status'),
    )

class AuditLog(Base):
    """Audit trail for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event info
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(50))  # authentication, model_update, threat_detection, etc.
    
    # Actor
    user_id = Column(String(36))
    device_id = Column(String(36))
    organization_id = Column(String(36))
    ip_address = Column(String(45))
    
    # Details
    description = Column(Text)
    details = Column(JSON)
    
    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_audit_type_time', 'event_type', 'timestamp'),
        Index('idx_audit_org', 'organization_id'),
    )

class ThreatIntelligence(Base):
    """Shared threat intelligence across the platform"""
    __tablename__ = "threat_intelligence"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Threat pattern
    pattern_name = Column(String(255), nullable=False)
    pattern_signature = Column(Text)  # Hash or signature of the threat pattern
    threat_category = Column(String(100))
    
    # Severity and impact
    severity = Column(Enum(ThreatSeverityEnum), nullable=False)
    prevalence = Column(Integer, default=1)  # Number of times seen
    
    # Detection criteria
    detection_rules = Column(JSON)
    indicators_of_compromise = Column(JSON)
    
    # Mitigation
    recommended_actions = Column(JSON)
    mitigation_effectiveness = Column(Float)
    
    # Timestamps
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_threat_intel_category', 'threat_category'),
        Index('idx_threat_intel_severity', 'severity'),
    )

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def init_database(database_url: str = "postgresql://user:password@localhost/federated_iot"):
    """Initialize database with all tables"""
    engine = create_engine(database_url, echo=True)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()

# ============================================================================
# SAMPLE QUERIES (for reference)
# ============================================================================

"""
# Get all active devices for an organization
session.query(Device).filter(
    Device.organization_id == org_id,
    Device.status == DeviceStatusEnum.ACTIVE
).all()

# Get recent high-severity threats
session.query(Threat).filter(
    Threat.severity.in_([ThreatSeverityEnum.HIGH, ThreatSeverityEnum.CRITICAL]),
    Threat.detected_at >= datetime.utcnow() - timedelta(hours=24)
).order_by(Threat.detected_at.desc()).all()

# Get device metrics for the last hour
session.query(DeviceMetrics).filter(
    DeviceMetrics.device_id == device_id,
    DeviceMetrics.timestamp >= datetime.utcnow() - timedelta(hours=1)
).order_by(DeviceMetrics.timestamp.asc()).all()

# Get model updates for a specific round
session.query(ModelUpdate).filter(
    ModelUpdate.global_model_version == version,
    ModelUpdate.is_validated == True
).all()

# Get threat statistics by device type
session.query(
    Device.device_type,
    func.count(Threat.id).label('threat_count')
).join(Threat).group_by(Device.device_type).all()
"""