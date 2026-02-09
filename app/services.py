"""Business logic: windows, predictions, dashboard."""
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import PoseWindow, WindowPrediction


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
