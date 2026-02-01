"""
Service layer for database operations.

This module contains business logic functions for interacting with the database.
All database operations are centralized here to maintain separation of concerns
and make the code more maintainable and testable.
"""
import math
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models import Device, ActivityEvent
from datetime import datetime, UTC


def compute_input_quality(
    keypoints: List[List[List[float]]], window_size: int
) -> Tuple[int, float, float]:
    """
    Compute input quality metrics from keypoints for one person.

    A frame is considered "ok" if every keypoint confidence is in [0, 1] and
    not NaN. frames_ok_ratio = (number of ok frames) / window_size.

    Args:
        keypoints: [T][K][3] — frames, keypoints per frame, (x, y, confidence)
        window_size: Expected number of frames (denominator for frames_ok_ratio)

    Returns:
        Tuple of (k_count, avg_pose_conf, frames_ok_ratio)
    """
    if not keypoints or window_size <= 0:
        return 0, 0.0, 0.0

    k_count = len(keypoints[0])
    conf_sum = 0.0
    conf_count = 0
    ok_frames = 0

    for frame in keypoints:
        frame_ok = True
        for kp in frame:
            if len(kp) < 3:
                frame_ok = False
                continue
            c = kp[2]
            if math.isnan(c) or c < 0.0 or c > 1.0:
                frame_ok = False
            conf_sum += c
            conf_count += 1
        if frame_ok:
            ok_frames += 1

    avg_pose_conf = conf_sum / conf_count if conf_count else 0.0
    frames_ok_ratio = ok_frames / window_size if window_size else 0.0
    # Clamp to [0, 1] in case ok_frames > window_size
    frames_ok_ratio = max(0.0, min(1.0, frames_ok_ratio))

    return k_count, avg_pose_conf, frames_ok_ratio


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
    confidence: float,
    k_count: Optional[int] = None,
    avg_pose_conf: Optional[float] = None,
    frames_ok_ratio: Optional[float] = None,
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
        k_count: Number of keypoints (17 or 25), optional
        avg_pose_conf: Average keypoint confidence for input quality, optional
        frames_ok_ratio: Ratio of valid frames (0..1), optional

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
        k_count=k_count,
        avg_pose_conf=avg_pose_conf,
        frames_ok_ratio=frames_ok_ratio,
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
