"""Main FastAPI application."""
from uuid import UUID
from fastapi import FastAPI, HTTPException, Header, Depends, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Tuple, Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.config import API_KEY, MODEL_KEY_DEFAULT
from app.database import get_db
from app.models_meta import list_available as list_models, get_labels_and_version
from app.schemas import (
    InferenceRequestSchema,
    InferenceResponseSchema,
    IngestWindowBody,
    PersonResultSchema,
    PredictWindowBody,
    SetLabelBody,
    TopKItemSchema,
    WindowSchema,
)
from app.logging import log_inference_request, get_logger
from app.health import router as health_router
from app.services import (
    create_pose_window_from_ingest,
    get_dashboard_windows,
    get_window_by_id,
    get_windows,
    get_recent_windows_with_predictions,
    run_predict_for_window,
    update_window_label,
)

app = FastAPI(title="Cloud HAR API", version="1.0.0")
templates = Jinja2Templates(directory="app/templates")

# Include health router
app.include_router(health_router)

logger = get_logger("main")


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """Dependency to verify API key."""
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
    Mock inference logic.
    
    Returns:
        tuple: (activity, confidence, top_k)
    """
    if pose_conf < 0.4:
        activity = "unknown"
        confidence = 0.2
        top_k = [
            TopKItemSchema(label="unknown", score=0.2),
            TopKItemSchema(label="standing", score=0.1),
            TopKItemSchema(label="walking", score=0.1),
        ]
    else:
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
    api_key: str = Depends(verify_api_key)
) -> InferenceResponseSchema:
    """
    Inference endpoint for activity recognition.
    
    Accepts skeleton window data and returns mock activity predictions.
    """
    request_id = str(uuid.uuid4())
    
    with log_inference_request(
        device_id=request.device_id,
        camera_id=request.camera_id,
        num_people=len(request.people),
        request_id=request_id
    ):
        # Process each person
        results = []
        for person in request.people:
            activity, confidence, top_k = mock_inference_logic(person.pose_conf)
            
            result = PersonResultSchema(
                track_id=person.track_id,
                activity=activity,
                confidence=confidence,
                top_k=top_k
            )
            results.append(result)
        
        # Build response
        response = InferenceResponseSchema(
            schema_version=1,
            device_id=request.device_id,
            camera_id=request.camera_id,
            window=request.window,
            results=results
        )
        
        return response


# --- Models API ---
@app.get("/v1/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    return {"models": list_models()}


@app.get("/v1/models/{model_key}")
async def get_model(model_key: str, api_key: str = Depends(verify_api_key)):
    if model_key not in list_models():
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        labels, model_version = get_labels_and_version(model_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"model_key": model_key, "model_version": model_version, "labels": labels}


# --- Windows API ---
@app.get("/v1/windows")
async def list_windows(limit: int = 100, api_key: str = Depends(verify_api_key), db: Session = Depends(get_db)):
    windows = get_windows(db, limit=limit)
    return {"windows": [{"id": str(w.id), "device_id": w.device_id, "camera_id": w.camera_id, "track_id": w.track_id, "ts_start_ms": w.ts_start_ms, "ts_end_ms": w.ts_end_ms, "fps": w.fps, "window_size": w.window_size, "label": w.label, "created_at": w.created_at.isoformat() if w.created_at else None} for w in windows]}


@app.get("/v1/dashboard/windows")
async def dashboard_windows(
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
    model_key: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    device_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    track_id: Optional[int] = None,
    only_with_predictions: bool = Query(False),
    pred_label: Optional[str] = None,
    max_pred_conf: Optional[float] = Query(None, ge=0, le=1),
    only_unlabeled: bool = Query(False),
    only_labeled: bool = Query(False),
    only_mismatches: bool = Query(False),
):
    """List windows for dashboard with optional filters; each item includes prediction with model_version."""
    rows = get_dashboard_windows(
        db,
        model_key=model_key,
        limit=limit,
        device_id=device_id,
        camera_id=camera_id,
        track_id=track_id,
        only_with_predictions=only_with_predictions,
        pred_label=pred_label,
        max_pred_conf=max_pred_conf,
        only_unlabeled=only_unlabeled,
        only_labeled=only_labeled,
        only_mismatches=only_mismatches,
    )
    return rows


@app.get("/v1/windows/{window_id}")
async def get_window(window_id: UUID, api_key: str = Depends(verify_api_key), db: Session = Depends(get_db)):
    w = get_window_by_id(db, window_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Window not found")
    return {"id": str(w.id), "device_id": w.device_id, "camera_id": w.camera_id, "track_id": w.track_id, "ts_start_ms": w.ts_start_ms, "ts_end_ms": w.ts_end_ms, "fps": w.fps, "window_size": w.window_size, "label": w.label, "created_at": w.created_at.isoformat() if w.created_at else None}


@app.post("/v1/windows/{window_id}/label")
async def set_window_label(window_id: UUID, body: SetLabelBody, api_key: str = Depends(verify_api_key), db: Session = Depends(get_db)):
    w = update_window_label(db, window_id, body.label)
    if w is None:
        raise HTTPException(status_code=404, detail="Window not found")
    return {"id": str(w.id), "label": w.label}


@app.post("/v1/windows/{window_id}/predict")
async def predict_window(
    window_id: UUID,
    body: PredictWindowBody,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Run ONNX model on window keypoints; optionally store result in window_predictions."""
    w = get_window_by_id(db, window_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Window not found")
    if body.model_key not in list_models():
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        return run_predict_for_window(
            db, window_id, body.model_key,
            store=body.store,
            return_probs=body.return_probs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/windows/ingest")
async def ingest_window(
    body: IngestWindowBody,
    predict: bool = True,
    model_key: Optional[str] = None,
    store_prediction: bool = True,
    return_probs: bool = False,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Ingest a full window from edge; by default run ONNX and store prediction."""
    try:
        w = create_pose_window_from_ingest(db, body)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Window with this id already exists. Omit 'id' in the body to create a new window.",
        )
    out = {
        "id": str(w.id),
        "device_id": w.device_id,
        "camera_id": w.camera_id,
        "track_id": w.track_id,
        "ts_start_ms": w.ts_start_ms,
        "ts_end_ms": w.ts_end_ms,
        "fps": w.fps,
        "window_size": w.window_size,
        "label": w.label,
        "created_at": w.created_at,
    }
    if predict:
        key = model_key or MODEL_KEY_DEFAULT
        if key not in list_models():
            raise HTTPException(status_code=503, detail="Model not found")
        try:
            pred_out = run_predict_for_window(
                db, w.id, key, store=store_prediction, return_probs=return_probs
            )
            out["pred_label"] = pred_out["pred_label"]
            out["pred_conf"] = pred_out["pred_conf"]
            out["model_key"] = pred_out["model_key"]
            out["prediction"] = {
                "model_key": pred_out["model_key"],
                "pred_label": pred_out["pred_label"],
                "pred_conf": pred_out["pred_conf"],
            }
            if return_probs and "probs" in pred_out:
                out["probs"] = pred_out["probs"]
                out["prediction"]["probs"] = pred_out["probs"]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return out


# --- Dashboard (HTML) ---
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, model_key: Optional[str] = None, db: Session = Depends(get_db)):
    key = model_key or MODEL_KEY_DEFAULT
    models = list_models()
    if key and key not in models:
        key = models[0] if models else None
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "model_key": key,
        "models": models,
        "api_key": API_KEY,
    })


@app.get("/dashboard/label", response_class=HTMLResponse)
async def label_page(request: Request, model_key: Optional[str] = None, db: Session = Depends(get_db)):
    key = model_key or MODEL_KEY_DEFAULT
    models = list_models()
    if key and key not in models:
        key = models[0] if models else None
    label_list = []
    if key:
        try:
            label_list, _ = get_labels_and_version(key)
        except FileNotFoundError:
            pass
    return templates.TemplateResponse("label.html", {
        "request": request,
        "labels": label_list,
        "model_key": key,
        "models": models,
        "api_key": API_KEY,
    })
