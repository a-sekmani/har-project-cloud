"""
Service layer for database operations and inference pipeline.

This module contains business logic functions for interacting with the database
and for the inference + persist pipeline (infer_and_persist). Inference uses
feature-based logic (extract_window_features + infer_activity) from Phase 4.
"""
import math
import time
import uuid
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models import Device, ActivityEvent, PoseWindow
from app.features import extract_window_features
from app.inference import infer_activity
from app.schemas import (
    DebugInfoSchema,
    DebugPerPersonSchema,
    InferenceRequestSchema,
    InferenceResponseSchema,
    PersonResultSchema,
)
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


def create_pose_window(
    db: Session,
    device_id: str,
    camera_id: str,
    session_id: Optional[str],
    track_id: int,
    ts_start_ms: int,
    ts_end_ms: int,
    fps: float,
    window_size: int,
    coord_space: str,
    keypoints: List[List[List[float]]],
    mean_pose_conf: Optional[float],
    missing_ratio: Optional[float] = None,
) -> PoseWindow:
    """
    Create one PoseWindow row. keypoints stored as JSON (list of lists).
    Returns the created PoseWindow with id.
    """
    row = PoseWindow(
        device_id=device_id,
        camera_id=camera_id,
        session_id=session_id,
        track_id=track_id,
        ts_start_ms=ts_start_ms,
        ts_end_ms=ts_end_ms,
        fps=float(fps),
        window_size=window_size,
        coord_space=coord_space,
        keypoints=keypoints,
        mean_pose_conf=mean_pose_conf,
        missing_ratio=missing_ratio,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def infer_and_persist(
    request: InferenceRequestSchema,
    db: Session,
    window_id: Optional[str] = None,
    window_ids: Optional[List[Optional[str]]] = None,
) -> InferenceResponseSchema:
    """
    Run feature-based inference and persist results to the database.
    Single source of truth for inference + storage; used by both
    POST /v1/activity/infer and by the edge window-complete hook.
    window_id: single id (edge path, one person). window_ids: one per person (infer path).
    """
    start_time = time.perf_counter()
    upsert_device(db, request.device_id)

    results: List[PersonResultSchema] = []
    per_person_debug: List[DebugPerPersonSchema] = []
    k_count_request: Optional[int] = None

    for i, person in enumerate(request.people):
        k_count, avg_pose_conf, frames_ok_ratio = compute_input_quality(
            person.keypoints, request.window.size
        )
        if k_count_request is None:
            k_count_request = k_count

        quality = (k_count, avg_pose_conf, frames_ok_ratio)
        features = extract_window_features(person.keypoints, request.window.fps)
        activity, confidence, top_k, infer_debug = infer_activity(features, quality)

        per_person_debug.append(
            DebugPerPersonSchema(
                avg_pose_conf=round(avg_pose_conf, 4),
                frames_ok_ratio=round(frames_ok_ratio, 4),
                features=infer_debug.get("features"),
                thresholds=infer_debug.get("thresholds"),
            )
        )

        results.append(
            PersonResultSchema(
                track_id=person.track_id,
                activity=activity,
                confidence=confidence,
                top_k=top_k,
            )
        )

        person_window_id = (window_ids[i] if window_ids and i < len(window_ids) else None) or window_id
        create_activity_event(
            db=db,
            device_id=request.device_id,
            camera_id=request.camera_id,
            track_id=person.track_id,
            ts_start_ms=request.window.ts_start_ms,
            ts_end_ms=request.window.ts_end_ms,
            fps=request.window.fps,
            window_size=request.window.size,
            activity=activity,
            confidence=confidence,
            k_count=k_count,
            avg_pose_conf=avg_pose_conf,
            frames_ok_ratio=frames_ok_ratio,
            window_id=person_window_id,
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    debug = DebugInfoSchema(
        frames_received=request.window.size,
        k_count=k_count_request or 0,
        latency_ms=latency_ms,
        per_person=per_person_debug,
    )

    return InferenceResponseSchema(
        schema_version=1,
        device_id=request.device_id,
        camera_id=request.camera_id,
        window=request.window,
        results=results,
        debug=debug,
    )


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
    window_id: Optional[str] = None,
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
        window_id: Optional UUID of the pose window that produced this event

    Returns:
        ActivityEvent: The created event object
    """
    event_kw: dict = dict(
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
        created_at=datetime.now(UTC),
    )
    if window_id is not None:
        event_kw["window_id"] = window_id if isinstance(window_id, uuid.UUID) else uuid.UUID(str(window_id))
    event = ActivityEvent(**event_kw)
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


def get_windows(
    db: Session,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    track_id: Optional[int] = None,
    ts_from_ms: Optional[int] = None,
    ts_to_ms: Optional[int] = None,
    label: Optional[str] = None,
    limit: int = 100,
) -> List[PoseWindow]:
    """
    Query pose windows with optional filters. Returns metadata (caller may omit keypoints).
    """
    q = db.query(PoseWindow)
    if device_id is not None:
        q = q.filter(PoseWindow.device_id == device_id)
    if session_id is not None:
        q = q.filter(PoseWindow.session_id == session_id)
    if camera_id is not None:
        q = q.filter(PoseWindow.camera_id == camera_id)
    if track_id is not None:
        q = q.filter(PoseWindow.track_id == track_id)
    if ts_from_ms is not None:
        q = q.filter(PoseWindow.ts_start_ms >= ts_from_ms)
    if ts_to_ms is not None:
        q = q.filter(PoseWindow.ts_end_ms <= ts_to_ms)
    if label is not None:
        q = q.filter(PoseWindow.label == label)
    q = q.order_by(desc(PoseWindow.created_at)).limit(limit)
    return q.all()


def get_window_by_id(db: Session, window_id: uuid.UUID) -> Optional[PoseWindow]:
    """Return a single PoseWindow by id, or None if not found."""
    return db.query(PoseWindow).filter(PoseWindow.id == window_id).first()


def update_window_label(
    db: Session,
    window_id: uuid.UUID,
    label: str,
    label_source: str,
) -> Optional[PoseWindow]:
    """
    Update PoseWindow label, label_source, labeled_at. Returns updated row or None if not found.
    """
    # #region agent log
    _debug_log = "/Users/ahmadsekmani/Desktop/Projects/har-project-cloud/.cursor/debug.log"
    def _dlog(hypothesis_id: str, message: str, **data):
        try:
            import json
            import time
            import logging
            with open(_debug_log, "a") as f:
                f.write(json.dumps({"hypothesisId": hypothesis_id, "message": message, "data": data, "timestamp": int(time.time() * 1000), "location": "services.update_window_label"}, default=str) + "\n")
        except Exception:
            pass
        logging.getLogger("cloud_har.services").error("DEBUG [%s] %s | %s", hypothesis_id, message, data)
    # #endregion
    _dlog("H2", "entry", window_id_str=str(window_id), label_repr=repr(label)[:50])
    row = db.query(PoseWindow).filter(PoseWindow.id == window_id).first()
    _dlog("H2", "row fetch", row_is_none=row is None)
    if row is None:
        return None
    _dlog("H5", "before assign labeled_at")
    row.label = label
    row.label_source = label_source
    row.labeled_at = datetime.now(UTC)
    _dlog("H2", "before commit")
    db.commit()
    _dlog("H2", "after commit")
    try:
        db.refresh(row)
        _dlog("H2", "after refresh")
    except Exception as refresh_err:
        _dlog("H2", "refresh failed", exc_type=type(refresh_err).__name__, exc_msg=str(refresh_err)[:200])
        # Refresh can fail with some drivers/relationships; row is already updated in memory
        pass
    return row
