"""
Database models for Cloud HAR application.

This module defines SQLAlchemy ORM models that represent database tables.
Models are used to interact with the database in an object-oriented way.

Tables:
- devices: Stores registered edge devices
- activity_events: Stores inference results (activity predictions)
"""
from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, UTC
import uuid

from app.database import Base


class Device(Base):
    """
    Device model representing an edge device.
    
    Devices are automatically registered when they send their first inference
    request. Each device can have multiple activity events.
    
    Attributes:
        id: Primary key (UUID)
        device_id: Unique identifier of the device (e.g., "pi-001")
        created_at: Timestamp when device was first registered
        events: Relationship to ActivityEvent objects (one-to-many)
    
    Table: devices
    """
    __tablename__ = "devices"

    # Primary key: UUID for unique identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Device identifier: unique string identifier from edge device
    # Indexed for fast lookups
    device_id = Column(String, unique=True, nullable=False, index=True)
    
    # Timestamp when device was first registered
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Relationship: one device can have many activity events
    events = relationship("ActivityEvent", back_populates="device")


class ActivityEvent(Base):
    """
    Activity event model representing an inference result.
    
    Each inference request can produce multiple events (one per person detected).
    Events store the predicted activity, confidence, and metadata about the
    time window and device that generated them.
    
    Attributes:
        id: Primary key (UUID)
        device_id: Foreign key to devices table
        camera_id: Identifier of the camera that captured the data
        track_id: Tracking ID of the person (for multi-person scenarios)
        ts_start_ms: Start timestamp of the time window (milliseconds)
        ts_end_ms: End timestamp of the time window (milliseconds)
        fps: Frames per second of the video
        window_size: Number of frames in the time window
        activity: Predicted activity label (e.g., "standing", "unknown")
        confidence: Confidence score of the prediction (0.0 to 1.0)
        created_at: Timestamp when event was created
        device: Relationship to Device object (many-to-one)
    
    Table: activity_events
    """
    __tablename__ = "activity_events"

    # Primary key: UUID for unique identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to devices table (references devices.device_id)
    # Indexed for fast filtering by device
    device_id = Column(String, ForeignKey("devices.device_id"), nullable=False, index=True)
    
    # Camera identifier (e.g., "cam-1", "cam-2")
    # Indexed for fast filtering by camera
    camera_id = Column(String, nullable=False, index=True)
    
    # Person tracking ID (for multi-person scenarios)
    # Indexed for fast filtering by track
    track_id = Column(Integer, nullable=False, index=True)
    
    # Time window start timestamp (milliseconds since epoch)
    ts_start_ms = Column(BigInteger, nullable=False)
    
    # Time window end timestamp (milliseconds since epoch)
    ts_end_ms = Column(BigInteger, nullable=False)
    
    # Video frames per second
    fps = Column(Integer, nullable=False)
    
    # Number of frames in the time window
    window_size = Column(Integer, nullable=False)
    
    # Predicted activity label (e.g., "standing", "unknown", "walking")
    # Indexed for fast filtering by activity type
    activity = Column(String, nullable=False, index=True)
    
    # Confidence score of the prediction (0.0 to 1.0)
    confidence = Column(Float, nullable=False)

    # Input quality (nullable for backward compatibility with existing rows)
    k_count = Column(Integer, nullable=True)  # 17 or 25 keypoints
    avg_pose_conf = Column(Float, nullable=True)  # average keypoint confidence
    frames_ok_ratio = Column(Float, nullable=True)  # 0..1, ratio of frames without NaN/invalid conf

    # Timestamp when event was created
    # Indexed for fast sorting by creation time
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    # Relationship: many events belong to one device
    device = relationship("Device", back_populates="events")
