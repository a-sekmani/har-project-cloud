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
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from typing import Tuple, Optional, List
from sqlalchemy.orm import Session
import uuid

from app.config import API_KEY
from app.database import get_db
from app.schemas import (
    InferenceRequestSchema,
    InferenceResponseSchema,
    PersonResultSchema,
    TopKItemSchema,
    WindowSchema
)
from app.logging import log_inference_request, get_logger
from app.health import router as health_router
from app.services import (
    upsert_device,
    create_activity_event,
    get_recent_events,
    get_device_events,
    get_all_devices
)

# Initialize FastAPI application
app = FastAPI(title="Cloud HAR API", version="1.0.0")

# Include health check router
app.include_router(health_router)

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


def mock_inference_logic(pose_conf: float) -> Tuple[str, float, list[TopKItemSchema]]:
    """
    Mock inference logic for activity recognition.
    
    This is a placeholder implementation that simulates ML model inference.
    In production, this would be replaced with actual model inference.
    
    Logic:
    - If pose_conf < 0.4: Returns "unknown" activity with low confidence (0.2)
    - If pose_conf >= 0.4: Returns "standing" activity with medium confidence (0.6)
    
    Args:
        pose_conf: Pose confidence score (0.0 to 1.0)
    
    Returns:
        tuple: (activity, confidence, top_k)
            - activity: Predicted activity label
            - confidence: Confidence score for the prediction
            - top_k: List of top K predictions with scores
    """
    if pose_conf < 0.4:
        # Low pose confidence -> uncertain activity
        activity = "unknown"
        confidence = 0.2
        top_k = [
            TopKItemSchema(label="unknown", score=0.2),
            TopKItemSchema(label="standing", score=0.1),
            TopKItemSchema(label="walking", score=0.1),
        ]
    else:
        # High pose confidence -> standing activity
        activity = "standing"
        confidence = 0.6
        top_k = [
            TopKItemSchema(label="standing", score=0.6),
            TopKItemSchema(label="walking", score=0.2),
            TopKItemSchema(label="sitting", score=0.2),
        ]
    
    return activity, confidence, top_k


@app.post("/v1/activity/infer", response_model=InferenceResponseSchema)
async def infer_activity(
    request: InferenceRequestSchema,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
) -> InferenceResponseSchema:
    """
    Main inference endpoint for activity recognition.
    
    This endpoint receives skeleton window data from edge devices and returns
    activity predictions. Each person in the request is processed separately,
    and results are saved to the database.
    
    Process:
    1. Validates API key (via dependency)
    2. Generates unique request ID for logging
    3. Upserts device record (creates if new, gets if exists)
    4. Processes each person in the request:
       - Runs mock inference logic
       - Creates result object
       - Saves event to database
    5. Returns response with all predictions
    
    Args:
        request: Inference request containing skeleton data and metadata
        api_key: Validated API key (from dependency)
        db: Database session (from dependency)
    
    Returns:
        InferenceResponseSchema: Response containing predictions for all people
    
    Raises:
        HTTPException: 401 if API key is invalid
        HTTPException: 422 if request validation fails
    """
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Log request and track latency
    with log_inference_request(
        device_id=request.device_id,
        camera_id=request.camera_id,
        num_people=len(request.people),
        request_id=request_id
    ):
        # Ensure device exists in database (upsert: create if new, get if exists)
        upsert_device(db, request.device_id)
        
        # Process each person detected in the window
        results = []
        for person in request.people:
            # Run mock inference to get activity prediction
            activity, confidence, top_k = mock_inference_logic(person.pose_conf)
            
            # Create result object for this person
            result = PersonResultSchema(
                track_id=person.track_id,
                activity=activity,
                confidence=confidence,
                top_k=top_k
            )
            results.append(result)
            
            # Save inference result to database as an activity event
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
                confidence=confidence
            )
        
        # Build and return response with all predictions
        response = InferenceResponseSchema(
            schema_version=1,
            device_id=request.device_id,
            camera_id=request.camera_id,
            window=request.window,
            results=results
        )
        
        return response


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
