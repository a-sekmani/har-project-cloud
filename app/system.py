"""System status for Models / System page: activity model, face gallery, edge status, health."""
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.config import MODEL_KEY_DEFAULT
from app.models import Person, PersonFace, GalleryVersion, PoseWindow
from app.models_meta import list_available as list_models, get_labels_and_version, get_model_meta_extra
from app.face.routes import get_current_gallery_version
from app.utils import isoformat_utc


def get_system_status(db: Session) -> dict[str, Any]:
    """
    Gather current activity model, face gallery status, edge status, and health.
    Used by GET /v1/system/status and the Models/System page.
    """
    # --- Current Activity Model ---
    model_key = MODEL_KEY_DEFAULT
    models = list_models()
    if not models:
        current_model = {"model_key": None, "version": None, "input_shape": None, "feature_spec": None, "date_loaded": None}
    else:
        if model_key not in models:
            model_key = models[0]
        try:
            _, version = get_labels_and_version(model_key)
        except FileNotFoundError:
            version = None
        extra = get_model_meta_extra(model_key) if model_key else {}
        current_model = {
            "model_key": model_key,
            "version": extra.get("version") or version,
            "input_shape": extra.get("input_shape"),
            "feature_spec": extra.get("feature_spec"),
            "date_loaded": extra.get("date_loaded"),
        }

    # --- Face Gallery Status ---
    try:
        gallery_version = get_current_gallery_version(db)
    except Exception:
        gallery_version = None
    persons_count = db.execute(select(func.count()).select_from(Person)).scalar() or 0
    faces_count = db.execute(select(func.count()).select_from(PersonFace)).scalar() or 0
    last_sync = None
    if gallery_version:
        gv = db.execute(
            select(GalleryVersion).where(GalleryVersion.version == gallery_version).limit(1)
        ).scalar_one_or_none()
        if gv is not None:
            last_sync = isoformat_utc(gv.created_at)
    face_gallery = {
        "gallery_version": gallery_version,
        "persons_count": persons_count,
        "faces_count": faces_count,
        "last_sync": last_sync,
    }

    # --- Edge Status (from PoseWindow) ---
    last_window_row = db.execute(
        select(PoseWindow.created_at).order_by(desc(PoseWindow.created_at)).limit(1)
    ).scalar_one_or_none()
    last_window_at = isoformat_utc(last_window_row) if last_window_row else None

    start_today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    windows_today = db.execute(
        select(func.count()).select_from(PoseWindow).where(PoseWindow.created_at >= start_today)
    ).scalar() or 0

    last_camera_row = db.execute(
        select(PoseWindow.camera_id).order_by(desc(PoseWindow.created_at)).limit(1)
    ).scalar_one_or_none()
    last_active_camera = last_camera_row if last_camera_row else None

    windows_with_person_id = db.execute(
        select(func.count()).select_from(PoseWindow).where(PoseWindow.person_id.isnot(None))
    ).scalar() or 0

    edge_status = {
        "last_window_at": last_window_at,
        "windows_today": windows_today,
        "last_active_camera": last_active_camera,
        "windows_with_person_id": windows_with_person_id,
    }

    # --- Health ---
    db_ok = False
    try:
        db.execute(select(1))
        db_ok = True
    except Exception:
        pass

    gallery_ok = gallery_version is not None
    model_ok = len(models) > 0
    prediction_ok = False
    if model_key:
        try:
            get_labels_and_version(model_key)
            prediction_ok = True
        except Exception:
            pass

    health = {
        "database": "ok" if db_ok else "error",
        "face_gallery": "ok" if gallery_ok else "error",
        "model_availability": "ok" if model_ok else "error",
        "prediction_service": "ok" if prediction_ok else "error",
    }

    return {
        "current_activity_model": current_model,
        "face_gallery": face_gallery,
        "edge_status": edge_status,
        "health": health,
    }
