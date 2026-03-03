"""
Database models for Cloud HAR application.

This module defines SQLAlchemy ORM models that represent database tables.
Models are used to interact with the database in an object-oriented way.

Tables:
- devices: Stores registered edge devices
- activity_events: Stores inference results (activity predictions)
- persons: Stores registered persons for face recognition
- person_faces: Stores face images and embeddings for each person
- gallery_versions: Tracks face gallery version changes
- pose_windows: Stores pose data windows with optional person identification
- window_predictions: Stores ONNX model predictions for pose windows
"""
from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, ForeignKey, Text, Boolean, JSON
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


class Person(Base):
    """
    Person model for face recognition management.
    
    Stores registered persons who can be identified via face recognition.
    Each person can have multiple face images/embeddings stored in PersonFace.
    
    Attributes:
        id: Primary key (UUID)
        name: Person's name (required)
        is_active: Whether person is active in face gallery (default True)
        created_at: Timestamp when person was registered
        faces: Relationship to PersonFace objects (one-to-many)
    """
    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    faces = relationship("PersonFace", back_populates="person", cascade="all, delete-orphan")
    windows = relationship("PoseWindow", back_populates="person")


class PersonFace(Base):
    """
    Face image and embedding for a person.
    
    Stores face images and their computed embeddings (512-dim vectors).
    Multiple faces can be stored per person for better recognition accuracy.
    
    Attributes:
        id: Primary key (UUID)
        person_id: Foreign key to persons table
        image_path: Relative path to stored face image
        embedding: Face embedding as JSON list (512 floats, L2 normalized)
        det_score: Face detection confidence score
        quality_score: Image quality score (optional)
        created_at: Timestamp when face was added
    """
    __tablename__ = "person_faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)  # Used to skip duplicate uploads by name
    embedding = Column(JSON, nullable=False)  # List of 512 floats, L2 normalized
    det_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    person = relationship("Person", back_populates="faces")


class GalleryVersion(Base):
    """
    Tracks face gallery version changes.
    
    A new version is created whenever faces are added/removed or
    persons are activated/deactivated. Edge devices use this to
    know when to refresh their local gallery cache.
    
    Attributes:
        id: Primary key (UUID)
        version: Version string (e.g., "v1", "v2")
        created_at: Timestamp when version was created
    """
    __tablename__ = "gallery_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)


class PoseWindow(Base):
    """
    Pose window: one time window of pose data.
    
    Can be labelled manually; ONNX predictions stored in WindowPrediction.
    Optionally linked to a Person when face recognition identifies someone.
    
    Attributes:
        person_id: Foreign key to persons table (nullable for unknown)
        person_name: Cached person name at time of ingest
        person_conf: Face recognition confidence from edge device
        gallery_version: Face gallery version used by edge device
    """
    __tablename__ = "pose_windows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    ts_start_ms = Column(BigInteger, nullable=False)
    ts_end_ms = Column(BigInteger, nullable=False)
    fps = Column(Integer, nullable=False)
    window_size = Column(Integer, nullable=False)
    label = Column(String, nullable=True, index=True)
    keypoints_json = Column(Text, nullable=True)  # JSON array of frames x keypoints (x,y,conf)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    # Person identification from edge face recognition
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True)
    person_name = Column(String, nullable=True)
    person_conf = Column(Float, nullable=True)
    gallery_version = Column(String, nullable=True)

    predictions = relationship("WindowPrediction", back_populates="window", order_by="WindowPrediction.created_at.desc()")
    person = relationship("Person", back_populates="windows")


class WindowPrediction(Base):
    """ONNX prediction for a pose window."""
    __tablename__ = "window_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_id = Column(UUID(as_uuid=True), ForeignKey("pose_windows.id", ondelete="CASCADE"), nullable=False, index=True)
    model_key = Column(String, nullable=False, index=True)
    pred_label = Column(String, nullable=False)
    pred_conf = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    window = relationship("PoseWindow", back_populates="predictions")
