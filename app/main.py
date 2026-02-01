"""
Main FastAPI application for Cloud HAR (Human Activity Recognition) API.

This module defines all API endpoints, dashboard routes, and core application logic.
It handles:
- Activity inference requests from edge devices
- Database operations for storing inference results
- API endpoints for retrieving events and devices
- Server-rendered dashboard pages

Endpoints:
- POST /v1/activity/infer: Main inference endpoint
- GET /v1/events: Get recent activity events
- GET /v1/devices: Get all registered devices
- GET /v1/devices/{device_id}/events: Get events for a specific device
- GET /dashboard: Main dashboard page
- GET /dashboard/devices/{device_id}: Device-specific dashboard
"""
from fastapi import FastAPI, HTTPException, Header, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import uuid

from app.config import API_KEY
from app.database import get_db
from app.schemas import InferenceRequestSchema, InferenceResponseSchema, WindowSchema
from app.logging import log_inference_request, get_logger
from app.health import router as health_router
from app.services import (
    get_all_devices,
    get_device_events,
    get_recent_events,
    infer_and_persist,
    upsert_device,
)
from app.config import EDGE_CAMERA_ID_DEFAULT
from app.aggregation import (
    get_buffer_details,
    get_buffer_sizes,
    get_frame_events_received_count,
    get_last_windows,
    ingest_internal_frame,
)
from app.edge_schemas import EdgeFrameEventSchema
from app.normalize import normalize_frame_event
from app.window_pipeline import get_windows_infer_failed_db_count

# Initialize FastAPI application
app = FastAPI(title="Cloud HAR API", version="1.0.0")

# Include health check router
app.include_router(health_router)


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request: Request, exc: OperationalError):
    """
    When PostgreSQL is not running (Connection refused), return a clear message
    instead of 500. Dashboard gets an HTML page; API routes get 503 JSON.
    """
    msg = (
        "Database unavailable. Start PostgreSQL (e.g. docker-compose up -d postgres) "
        "or check DATABASE_URL."
    )
    if request.url.path.startswith("/dashboard"):
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Database Unavailable</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:2em auto;padding:1em;">
  <h1>Database Unavailable</h1>
  <p>{msg}</p>
  <p><a href="/dashboard">Retry</a> &middot; <a href="/health">Health</a></p>
</body></html>"""
        return HTMLResponse(content=html, status_code=503)
    return JSONResponse(status_code=503, content={"detail": msg})


# Initialize Jinja2 template environment for dashboard rendering
jinja_env = Environment(loader=FileSystemLoader("app/templates"))

# Get logger instance for this module
logger = get_logger("main")


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency to verify API key authentication.
    
    This function is used as a dependency in protected endpoints to ensure
    requests include a valid API key in the X-API-Key header.
    
    Args:
        x_api_key: API key from X-API-Key header (optional at FastAPI level)
    
    Returns:
        str: The validated API key
    
    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


@app.post("/v1/activity/infer", response_model=InferenceResponseSchema)
async def infer_activity(
    request: InferenceRequestSchema,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
) -> InferenceResponseSchema:
    """
    Main inference endpoint. Validates API key and request, then delegates
    to infer_and_persist (single source of truth for inference + storage).
    """
    request_id = str(uuid.uuid4())
    with log_inference_request(
        device_id=request.device_id,
        camera_id=request.camera_id,
        num_people=len(request.people),
        request_id=request_id
    ):
        return infer_and_persist(request, db)


# ============================================================================
# Edge frame_event aggregation (no DB, no HAR)
# ============================================================================

def _resolve_camera_id(
    body: dict,
    query_camera_id: Optional[str] = None,
    x_camera_id: Optional[str] = None,
) -> str:
    """
    Resolve camera_id. Priority (literal):
    1. source.camera_id if present
    2. Query ?camera_id=...
    3. Header X-Camera-Id
    4. Default (e.g. "cam-1")
    """
    source = body.get("source") or {}
    if isinstance(source, dict) and source.get("camera_id"):
        return str(source["camera_id"])
    if query_camera_id:
        return query_camera_id
    if x_camera_id:
        return x_camera_id
    return EDGE_CAMERA_ID_DEFAULT


@app.post("/v1/edge/events", status_code=202)
async def edge_events(
    body: dict,
    api_key: str = Depends(verify_api_key),
    x_camera_id: Optional[str] = Header(None, alias="X-Camera-Id"),
    camera_id: Optional[str] = Query(None, alias="camera_id"),
):
    """
    Accept frame-level events from the edge (event_type "frame_event").
    Frames are normalized to internal format and aggregated per
    (device_id, camera_id, track_id). When a buffer reaches window.size, a Cloud
    window payload is built and logged (not stored, not sent to inference).
    camera_id priority: source.camera_id -> ?camera_id= -> X-Camera-Id -> default.
    """
    try:
        EdgeFrameEventSchema.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid frame_event: {e!s}")

    resolved_camera_id = _resolve_camera_id(body, query_camera_id=camera_id, x_camera_id=x_camera_id)
    internal_frames = normalize_frame_event(body, resolved_camera_id)
    completed: List[dict] = []
    for iframe in internal_frames:
        completed.extend(ingest_internal_frame(iframe))

    total_received = get_frame_events_received_count()
    buffer_sizes = get_buffer_sizes()
    logger.info(
        "edge_events | frame_events_received=%s | buffers=%s | windows_completed=%s",
        total_received, buffer_sizes, len(completed),
    )
    return {"status": "accepted"}


# ============================================================================
# Debug endpoints (buffers & windows)
# ============================================================================

@app.get("/debug/buffers")
async def debug_buffers(api_key: str = Depends(verify_api_key)):
    """
    Return current aggregation buffers: key (device_id|camera_id|track_id),
    frame_count, last_ts_ms per buffer; and windows_infer_failed_db count.
    """
    return {
        "buffers": get_buffer_details(),
        "frame_events_received": get_frame_events_received_count(),
        "windows_infer_failed_db": get_windows_infer_failed_db_count(),
    }


@app.get("/debug/windows")
async def debug_windows(
    n: int = Query(default=20, ge=1, le=100),
    api_key: str = Depends(verify_api_key),
):
    """
    Return last n completed windows (metadata only: device_id, camera_id, track_id,
    ts_start_ms, ts_end_ms, size, fps). No full keypoints.
    """
    return {"windows": get_last_windows(n)}


# ============================================================================
# API Endpoints for Data Retrieval
# ============================================================================

@app.get("/v1/events", response_model=List[dict])
async def get_events(
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get recent activity events from all devices.
    
    Returns the most recent activity events across all devices, ordered by
    creation time (newest first). Useful for monitoring overall system activity.
    
    Args:
        limit: Maximum number of events to return (1-1000, default: 100)
        db: Database session (from dependency)
    
    Returns:
        List[dict]: List of event dictionaries with all event fields
    
    Example:
        GET /v1/events?limit=50
    """
    # Get events from database (already sorted by created_at DESC)
    events = get_recent_events(db, limit=limit)
    
    # Convert SQLAlchemy models to dictionaries with ISO-formatted timestamps
    return [
        {
            "id": str(event.id),
            "device_id": event.device_id,
            "camera_id": event.camera_id,
            "track_id": event.track_id,
            "ts_start_ms": event.ts_start_ms,
            "ts_end_ms": event.ts_end_ms,
            "fps": event.fps,
            "window_size": event.window_size,
            "activity": event.activity,
            "confidence": event.confidence,
            "k_count": event.k_count,
            "avg_pose_conf": event.avg_pose_conf,
            "frames_ok_ratio": event.frames_ok_ratio,
            "created_at": event.created_at.isoformat()
        }
        for event in events
    ]


@app.get("/v1/devices", response_model=List[dict])
async def get_devices(db: Session = Depends(get_db)):
    """
    Get all registered devices.
    
    Returns a list of all devices that have sent at least one inference request.
    Devices are automatically registered on their first inference request.
    
    Args:
        db: Database session (from dependency)
    
    Returns:
        List[dict]: List of device dictionaries with id, device_id, and created_at
    
    Example:
        GET /v1/devices
    """
    devices = get_all_devices(db)
    return [
        {
            "id": str(device.id),
            "device_id": device.device_id,
            "created_at": device.created_at.isoformat()
        }
        for device in devices
    ]


@app.get("/v1/devices/{device_id}/events", response_model=List[dict])
async def get_device_events_endpoint(
    device_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get activity events for a specific device.
    
    Returns the most recent activity events for a given device, ordered by
    creation time (newest first). Useful for device-specific monitoring.
    
    Args:
        device_id: Unique identifier of the device
        limit: Maximum number of events to return (1-1000, default: 200)
        db: Database session (from dependency)
    
    Returns:
        List[dict]: List of event dictionaries for the specified device.
                   Returns empty list if device doesn't exist or has no events.
    
    Example:
        GET /v1/devices/pi-001/events?limit=100
    """
    events = get_device_events(db, device_id, limit=limit)
    return [
        {
            "id": str(event.id),
            "device_id": event.device_id,
            "camera_id": event.camera_id,
            "track_id": event.track_id,
            "ts_start_ms": event.ts_start_ms,
            "ts_end_ms": event.ts_end_ms,
            "fps": event.fps,
            "window_size": event.window_size,
            "activity": event.activity,
            "confidence": event.confidence,
            "k_count": event.k_count,
            "avg_pose_conf": event.avg_pose_conf,
            "frames_ok_ratio": event.frames_ok_ratio,
            "created_at": event.created_at.isoformat()
        }
        for event in events
    ]


# ============================================================================
# Dashboard Routes (Server-Rendered HTML)
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(limit: int = Query(default=100, ge=1, le=1000), db: Session = Depends(get_db)):
    """
    Main dashboard page showing recent activity events.
    
    Renders an HTML page with a table of recent events from all devices.
    The page auto-refreshes every 3 seconds to show new events.
    
    Args:
        limit: Maximum number of events to display (1-1000, default: 100)
        db: Database session (from dependency)
    
    Returns:
        HTMLResponse: Rendered HTML page with events table
    
    Example:
        GET /dashboard?limit=50
    """
    events = get_recent_events(db, limit=limit)
    template = jinja_env.get_template("dashboard.html")
    return HTMLResponse(content=template.render(
        events=events,
        limit=limit
    ))


@app.get("/dashboard/devices/{device_id}", response_class=HTMLResponse)
async def device_dashboard(
    device_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Device-specific dashboard page.
    
    Renders an HTML page with a table of events for a specific device.
    The page auto-refreshes every 3 seconds and includes a link back to
    the main dashboard.
    
    Args:
        device_id: Unique identifier of the device
        limit: Maximum number of events to display (1-1000, default: 200)
        db: Database session (from dependency)
    
    Returns:
        HTMLResponse: Rendered HTML page with device events table
    
    Example:
        GET /dashboard/devices/pi-001?limit=100
    """
    events = get_device_events(db, device_id, limit=limit)
    template = jinja_env.get_template("device_dashboard.html")
    return HTMLResponse(content=template.render(
        device_id=device_id,
        events=events,
        limit=limit
    ))
