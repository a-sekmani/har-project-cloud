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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import json
import time
import uuid

from app.config import API_KEY, LABELS_ALLOWED, STORE_INFER_WINDOWS
from app.database import get_db
from app.schemas import InferenceRequestSchema, InferenceResponseSchema, WindowSchema, WindowLabelSchema
from app.logging import log_inference_request, get_logger
from app.health import router as health_router
from app.services import (
    create_pose_window,
    get_all_devices,
    get_device_events,
    get_recent_events,
    get_window_by_id,
    get_windows,
    infer_and_persist,
    update_window_label,
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
from app.features import extract_window_features
from app.models import PoseWindow
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
    If STORE_INFER_WINDOWS, creates one PoseWindow per person (session_id null) then infers.
    """
    request_id = str(uuid.uuid4())
    with log_inference_request(
        device_id=request.device_id,
        camera_id=request.camera_id,
        num_people=len(request.people),
        request_id=request_id
    ):
        window_ids = None
        if STORE_INFER_WINDOWS and request.people:
            window_ids = []
            for person in request.people:
                features = extract_window_features(person.keypoints, request.window.fps)
                pw = create_pose_window(
                    db=db,
                    device_id=request.device_id,
                    camera_id=request.camera_id,
                    session_id=request.session_id,
                    track_id=person.track_id,
                    ts_start_ms=request.window.ts_start_ms,
                    ts_end_ms=request.window.ts_end_ms,
                    fps=request.window.fps,
                    window_size=request.window.size,
                    coord_space="norm",
                    keypoints=person.keypoints,
                    mean_pose_conf=person.pose_conf,
                    missing_ratio=features.missing_ratio,
                )
                window_ids.append(str(pw.id))
            return infer_and_persist(request, db, window_ids=window_ids)
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
# Windows API (Phase 5: store, list, label)
# ============================================================================

def _pose_window_to_item(w: PoseWindow, include_keypoints: bool = False) -> dict:
    """Build response dict for one PoseWindow. Ensures JSON-serializable types (e.g. Decimal -> float)."""
    def _dt_iso(v):
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)

    item = {
        "id": str(w.id),
        "device_id": w.device_id,
        "camera_id": w.camera_id,
        "session_id": w.session_id,
        "track_id": int(w.track_id),
        "ts_start_ms": int(w.ts_start_ms),
        "ts_end_ms": int(w.ts_end_ms),
        "fps": float(w.fps) if w.fps is not None else None,
        "window_size": int(w.window_size),
        "mean_pose_conf": float(w.mean_pose_conf) if w.mean_pose_conf is not None else None,
        "label": w.label,
        "label_source": w.label_source,
        "created_at": _dt_iso(w.created_at),
    }
    if w.labeled_at is not None:
        item["labeled_at"] = _dt_iso(w.labeled_at)
    if include_keypoints and w.keypoints is not None:
        item["keypoints"] = w.keypoints
    return item


@app.get("/v1/windows")
async def list_windows(
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    device_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    track_id: Optional[int] = Query(None),
    from_ts: Optional[int] = Query(None, alias="from"),
    to_ts: Optional[int] = Query(None, alias="to"),
    label: Optional[str] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    include_keypoints: int = Query(default=0, ge=0, le=1),
):
    """
    List pose windows with optional filters. By default keypoints are omitted.
    Use include_keypoints=1 to include keypoints in each item.
    """
    windows = get_windows(
        db,
        device_id=device_id,
        session_id=session_id,
        camera_id=camera_id,
        track_id=track_id,
        ts_from_ms=from_ts,
        ts_to_ms=to_ts,
        label=label,
        limit=limit,
    )
    return {
        "windows": [_pose_window_to_item(w, include_keypoints=(include_keypoints == 1)) for w in windows]
    }


@app.get("/v1/windows/{window_id}")
async def get_window(
    window_id: uuid.UUID,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    include_keypoints: int = Query(default=1, ge=0, le=1),
):
    """
    Return a single pose window by id. Keypoints included by default; use include_keypoints=0 to omit.
    """
    w = get_window_by_id(db, window_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Window not found")
    return _pose_window_to_item(w, include_keypoints=(include_keypoints == 1))


@app.post("/v1/windows/{window_id}/label")
async def label_window(
    window_id: uuid.UUID,
    body: WindowLabelSchema,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Set label and label_source for a pose window. Label must be in LABELS_ALLOWED.
    """
    if body.label not in LABELS_ALLOWED:
        raise HTTPException(
            status_code=422,
            detail=f"Label must be one of: {LABELS_ALLOWED}",
        )
    w = update_window_label(db, window_id, body.label, body.label_source)
    if w is None:
        raise HTTPException(status_code=404, detail="Window not found")
    return _pose_window_to_item(w, include_keypoints=False)


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


@app.get("/dashboard/label", response_class=HTMLResponse)
async def dashboard_label(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Labeling page: list last N pose windows with dropdown to set label."""
    # #region agent log
    _debug_log = "/Users/ahmadsekmani/Desktop/Projects/har-project-cloud/.cursor/debug.log"
    def _dlog(hypothesis_id: str, message: str, **data):
        try:
            with open(_debug_log, "a") as f:
                f.write(json.dumps({"hypothesisId": hypothesis_id, "message": message, "data": data, "timestamp": int(time.time() * 1000), "location": "main.dashboard_label_GET"}, default=str) + "\n")
        except Exception:
            pass
        logger.error("DEBUG [%s] %s | %s", hypothesis_id, message, data)
    # #endregion
    try:
        _dlog("H_GET", "entry")
        rows = get_windows(db, limit=limit)
        _dlog("H_GET", "get_windows done", rows_len=len(rows))
        windows = []
        for w in rows:
            windows.append({
                "id": str(w.id),
                "device_id": str(w.device_id) if w.device_id is not None else "",
                "camera_id": str(w.camera_id) if w.camera_id is not None else "",
                "track_id": int(w.track_id),
                "ts_start_ms": int(w.ts_start_ms),
                "ts_end_ms": int(w.ts_end_ms),
                "label": str(w.label) if w.label is not None else None,
            })
        _dlog("H_GET", "windows built", windows_len=len(windows))
        template = jinja_env.get_template("label.html")
        out = template.render(windows=windows, limit=limit, labels_allowed=LABELS_ALLOWED)
        _dlog("H_GET", "render done")
        return HTMLResponse(content=out)
    except Exception as e:
        _dlog("H_GET", "exception", exc_type=type(e).__name__, exc_msg=str(e)[:300])
        raise


@app.post("/dashboard/label", response_class=HTMLResponse)
async def dashboard_label_submit(
    request: Request,
    db: Session = Depends(get_db),
):
    """Form submit: set label for a window then redirect to label page."""
    # #region agent log
    _debug_log = "/Users/ahmadsekmani/Desktop/Projects/har-project-cloud/.cursor/debug.log"
    def _dlog(hypothesis_id: str, message: str, **data):
        try:
            with open(_debug_log, "a") as f:
                f.write(json.dumps({"hypothesisId": hypothesis_id, "message": message, "data": data, "timestamp": int(time.time() * 1000), "location": "main.dashboard_label_submit"}, default=str) + "\n")
        except Exception:
            pass
        logger.error("DEBUG [%s] %s | %s", hypothesis_id, message, data)
    # #endregion
    try:
        form = await request.form()
        window_id_s = form.get("window_id")
        label = form.get("label")
        _dlog("H1", "form raw", window_id_s_type=type(window_id_s).__name__, label_type=type(label).__name__, form_keys=list(form.keys()) if hasattr(form, "keys") else None)
        # Coerce to str (form can sometimes return list for multi-value keys)
        if isinstance(window_id_s, list):
            window_id_s = window_id_s[0] if window_id_s else ""
        window_id_s = (window_id_s or "").strip() if window_id_s is not None else ""
        if isinstance(label, list):
            label = label[0] if label else ""
        label = (label or "").strip() if label is not None else ""
        _dlog("H1", "after coercion", window_id_s_len=len(window_id_s), label_repr=repr(label)[:80])
        if not window_id_s:
            raise HTTPException(status_code=400, detail="window_id required")
        if label and label not in LABELS_ALLOWED:
            raise HTTPException(status_code=422, detail=f"Label must be one of: {LABELS_ALLOWED}")
        try:
            window_id = uuid.UUID(window_id_s)
        except (ValueError, TypeError) as ue:
            _dlog("H3", "UUID parse failed", error=type(ue).__name__, msg=str(ue)[:100])
            raise HTTPException(status_code=400, detail="Invalid window_id")
        _dlog("H2", "before update_window_label", window_id_str=str(window_id))
        w = update_window_label(db, window_id, label, "manual")
        _dlog("H2", "after update_window_label", w_is_none=w is None, w_id=str(w.id) if w else None)
        if w is None:
            raise HTTPException(status_code=404, detail="Window not found")
        return RedirectResponse(url="/dashboard/label", status_code=303)
    except HTTPException:
        raise
    except Exception as e:
        _dlog("H4", "exception caught", exc_type=type(e).__name__, exc_msg=str(e)[:200])
        logger.exception("dashboard_label_submit failed: %s", e)
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{type(e).__name__}: {e}</p><p><a href='/dashboard/label'>Back to label page</a></p></body></html>",
            status_code=500,
        )
