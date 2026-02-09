"""Business logic: windows, predictions, dashboard."""
import json
from datetime import datetime, UTC
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import PoseWindow, WindowPrediction
from app.schemas import IngestWindowBody


def get_windows(db: Session, limit: int = 100):
    return db.query(PoseWindow).order_by(desc(PoseWindow.created_at)).limit(limit).all()


def get_window_by_id(db: Session, window_id: UUID) -> PoseWindow | None:
    return db.query(PoseWindow).filter(PoseWindow.id == window_id).first()


def update_window_label(db: Session, window_id: UUID, label: str) -> PoseWindow | None:
    w = get_window_by_id(db, window_id)
    if w is None:
        return None
    w.label = label
    db.commit()
    db.refresh(w)
    return w


def get_recent_windows_with_predictions(db: Session, limit: int = 100, model_key: str | None = None) -> list[dict]:
    windows = db.query(PoseWindow).order_by(desc(PoseWindow.created_at)).limit(limit).all()
    out = []
    for w in windows:
        row = {
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
            "prediction": None,
        }
        if model_key:
            pred = (
                db.query(WindowPrediction)
                .filter(WindowPrediction.window_id == w.id, WindowPrediction.model_key == model_key)
                .order_by(desc(WindowPrediction.created_at))
                .limit(1)
                .first()
            )
            if pred:
                row["prediction"] = {"pred_label": pred.pred_label, "pred_conf": pred.pred_conf, "model_key": pred.model_key}
        out.append(row)
    return out


def get_dashboard_windows(
    db: Session,
    model_key: str | None,
    limit: int = 100,
    device_id: str | None = None,
    camera_id: str | None = None,
    track_id: int | None = None,
    only_with_predictions: bool = False,
    pred_label: str | None = None,
    max_pred_conf: float | None = None,
    only_unlabeled: bool = False,
    only_labeled: bool = False,
    only_mismatches: bool = False,
) -> list[dict]:
    """Windows for dashboard with optional filters; each row has latest prediction per model_key (LEFT JOIN–style)."""
    from app.models_meta import get_labels_and_version

    query = db.query(PoseWindow).order_by(desc(PoseWindow.created_at))
    if device_id is not None and device_id != "":
        query = query.filter(PoseWindow.device_id == device_id)
    if camera_id is not None and camera_id != "":
        query = query.filter(PoseWindow.camera_id == camera_id)
    if track_id is not None:
        query = query.filter(PoseWindow.track_id == track_id)
    fetch_limit = limit * 4 if (
        only_with_predictions or pred_label is not None or max_pred_conf is not None
        or only_unlabeled or only_labeled or only_mismatches
    ) else limit
    windows = query.limit(fetch_limit).all()

    model_version = None
    if model_key:
        try:
            _, model_version = get_labels_and_version(model_key)
        except FileNotFoundError:
            pass

    # Single bulk query: latest prediction per (window_id, model_key) — ORDER BY created_at DESC, then take first per window_id
    latest_pred_by_window: dict[UUID, WindowPrediction] = {}
    if model_key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == model_key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p

    out = []
    for w in windows:
        row = {
            "id": str(w.id),
            "created_at": w.created_at.isoformat() if w.created_at else None,
            "device_id": w.device_id,
            "camera_id": w.camera_id,
            "track_id": w.track_id,
            "ts_start_ms": w.ts_start_ms,
            "ts_end_ms": w.ts_end_ms,
            "fps": w.fps,
            "window_size": w.window_size,
            "label": w.label,
            "prediction": None,
        }
        pred = latest_pred_by_window.get(w.id) if model_key else None
        if pred:
            row["prediction"] = {
                "model_key": pred.model_key,
                "model_version": model_version,
                "features": None,
                "pred_label_id": None,
                "pred_label": pred.pred_label,
                "pred_conf": pred.pred_conf,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
            }
        if only_with_predictions and row["prediction"] is None:
            continue
        if pred_label is not None and (row["prediction"] is None or row["prediction"].get("pred_label") != pred_label):
            continue
        if max_pred_conf is not None and (row["prediction"] is None or row["prediction"].get("pred_conf", 1) > max_pred_conf):
            continue
        label_val = (row.get("label") or "").strip()
        if only_unlabeled and label_val != "":
            continue
        if only_labeled and label_val == "":
            continue
        if only_mismatches:
            pred_l = (row["prediction"] or {}).get("pred_label")
            if not label_val or not pred_l or label_val == pred_l:
                continue
        out.append(row)
        if len(out) >= limit:
            break
    return out


def create_window_prediction(
    db: Session,
    window_id: UUID,
    model_key: str,
    pred_label: str,
    pred_conf: float,
) -> WindowPrediction:
    """Insert a new window prediction row (keeps history)."""
    pred = WindowPrediction(
        window_id=window_id,
        model_key=model_key,
        pred_label=pred_label,
        pred_conf=pred_conf,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def create_pose_window_from_ingest(db: Session, body: IngestWindowBody) -> PoseWindow:
    """Build and persist a PoseWindow from ingest body; keypoints stored as JSON string."""
    window_id = body.id if body.id is not None else uuid4()
    created = body.created_at if body.created_at is not None else datetime.now(UTC)
    w = PoseWindow(
        id=window_id,
        device_id=body.device_id,
        camera_id=body.camera_id,
        track_id=body.track_id,
        ts_start_ms=body.ts_start_ms,
        ts_end_ms=body.ts_end_ms,
        fps=int(body.fps) if isinstance(body.fps, float) else body.fps,
        window_size=body.window_size,
        label=body.label,
        keypoints_json=json.dumps(body.keypoints),
        created_at=created,
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return w


def run_predict_for_window(
    db: Session,
    window_id: UUID,
    model_key: str,
    store: bool = True,
    return_probs: bool = False,
) -> dict:
    """
    Run ONNX prediction for a stored window. Returns dict with pred_label, pred_conf, model_key,
    and optionally probs. Raises FileNotFoundError if model missing (503), ValueError on feature/inference error (400).
    """
    from app.ml.features import keypoints_to_model_input
    from app.ml.onnx_runner import run_onnx_predict

    w = get_window_by_id(db, window_id)
    if w is None:
        raise ValueError("Window not found")
    if not w.keypoints_json:
        raise ValueError("Window has no keypoints")
    keypoints = json.loads(w.keypoints_json)
    inp = keypoints_to_model_input(keypoints, window_size=w.window_size or 30)
    pred_label, pred_conf, all_probs = run_onnx_predict(model_key, inp)
    if store:
        create_window_prediction(db, window_id, model_key, pred_label, pred_conf)
    out = {"pred_label": pred_label, "pred_conf": pred_conf, "model_key": model_key}
    if return_probs:
        out["probs"] = [{"label": l, "score": s} for l, s in all_probs]
    return out
