"""
Service layer for database operations.

This module contains business logic functions for interacting with the database.
All database operations are centralized here to maintain separation of concerns
and make the code more maintainable and testable.
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models import Device, ActivityEvent
from datetime import datetime, UTC


def upsert_device(db: Session, device_id: str) -> Device:
    """
    Create a new device or get existing device by device_id (upsert operation).
    
    This function implements an "upsert" pattern: if the device exists, it
    returns the existing device; if not, it creates a new one. This ensures
    devices are automatically registered on their first inference request without
    creating duplicates.
    
    Args:
        db: Database session
        device_id: Unique identifier of the device (e.g., "pi-001")
    
    Returns:
        Device: The device object (either newly created or existing)
    """
    # Check if device already exists
    device = db.query(Device).filter(Device.device_id == device_id).first()
    
    # If device doesn't exist, create it
    if not device:
        device = Device(device_id=device_id, created_at=datetime.now(UTC))
        db.add(device)
        db.commit()
        db.refresh(device)
    
    return device


def create_activity_event(
    db: Session,
    device_id: str,
    camera_id: str,
    track_id: int,
    ts_start_ms: int,
    ts_end_ms: int,
    fps: int,
    window_size: int,
    activity: str,
    confidence: float
) -> ActivityEvent:
    """
    Create a new activity event record in the database.
    
    This function saves an inference result as an activity event. Each person
    detected in an inference request creates a separate event record.
    
    Args:
        db: Database session
        device_id: Unique identifier of the device
        camera_id: Identifier of the camera that captured the data
        track_id: Tracking ID of the person (for multi-person tracking)
        ts_start_ms: Start timestamp of the time window (milliseconds)
        ts_end_ms: End timestamp of the time window (milliseconds)
        fps: Frames per second of the video
        window_size: Number of frames in the time window
        activity: Predicted activity label (e.g., "standing", "unknown")
        confidence: Confidence score of the prediction (0.0 to 1.0)
    
    Returns:
        ActivityEvent: The created event object
    """
    event = ActivityEvent(
        device_id=device_id,
        camera_id=camera_id,
        track_id=track_id,
        ts_start_ms=ts_start_ms,
        ts_end_ms=ts_end_ms,
        fps=fps,
        window_size=window_size,
        activity=activity,
        confidence=confidence,
        created_at=datetime.now(UTC)
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_recent_events(db: Session, limit: int = 100):
    """
    Get recent activity events from all devices, ordered by creation time.
    
    Returns the most recent events across all devices, sorted by created_at
    in descending order (newest first). Used by the main dashboard and
    /v1/events endpoint.
    
    Args:
        db: Database session
        limit: Maximum number of events to return (default: 100)
    
    Returns:
        List[ActivityEvent]: List of event objects, newest first
    """
    return db.query(ActivityEvent).order_by(desc(ActivityEvent.created_at)).limit(limit).all()


def get_device_events(db: Session, device_id: str, limit: int = 200):
    """
    Get activity events for a specific device.
    
    Returns events filtered by device_id, ordered by creation time in
    descending order (newest first). Used by device-specific dashboard
    and /v1/devices/{device_id}/events endpoint.
    
    Args:
        db: Database session
        device_id: Unique identifier of the device
        limit: Maximum number of events to return (default: 200)
    
    Returns:
        List[ActivityEvent]: List of event objects for the device, newest first.
                            Returns empty list if device has no events.
    """
    return db.query(ActivityEvent).filter(
        ActivityEvent.device_id == device_id
    ).order_by(desc(ActivityEvent.created_at)).limit(limit).all()


def get_all_devices(db: Session):
    """
    Get all registered devices.
    
    Returns all devices that have sent at least one inference request.
    Used by the /v1/devices endpoint.
    
    Args:
        db: Database session
    
    Returns:
        List[Device]: List of all device objects
    """
    return db.query(Device).all()
