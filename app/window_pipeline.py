"""
Window completion hook: when a window of EDGE_WINDOW_SIZE frames is built by aggregation,
optionally run infer_and_persist (EDGE_AUTO_INFER) and return status for debug.
"""
import logging
from typing import Any, Dict

from sqlalchemy.exc import OperationalError

from app.config import EDGE_AUTO_INFER
from app.database import SessionLocal
from app.schemas import InferenceRequestSchema
from app.services import create_pose_window, infer_and_persist
from app.features import extract_window_features

logger = logging.getLogger("cloud_har.window_pipeline")

# Counter for DB failures during auto-infer (for debug)
_windows_infer_failed_db = 0


def reset_windows_infer_failed_db_count() -> None:
    """Reset failed-DB counter (for tests)."""
    global _windows_infer_failed_db
    _windows_infer_failed_db = 0


def get_windows_infer_failed_db_count() -> int:
    """Return count of windows where auto-infer failed due to DB (for GET /debug/buffers)."""
    return _windows_infer_failed_db


def handle_completed_window(window_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    When a window is complete: if EDGE_AUTO_INFER is True, run infer_and_persist;
    otherwise just return disabled status. Never raises; on DB failure logs and
    returns failed_db so /v1/edge/events stays 202.
    """
    global _windows_infer_failed_db

    if not EDGE_AUTO_INFER:
        return {
            "auto_infer_attempted": False,
            "auto_infer_status": "disabled",
            "saved": False,
        }

    db = SessionLocal()
    try:
        request = InferenceRequestSchema.model_validate(window_payload)
        # One PoseWindow per completed window (one person from edge)
        person = request.people[0]
        window = request.window
        keypoints = person.keypoints
        mean_pose_conf = person.pose_conf
        features = extract_window_features(keypoints, window.fps)
        missing_ratio = features.missing_ratio

        pose_window = create_pose_window(
            db=db,
            device_id=request.device_id,
            camera_id=request.camera_id,
            session_id=request.session_id,
            track_id=person.track_id,
            ts_start_ms=window.ts_start_ms,
            ts_end_ms=window.ts_end_ms,
            fps=window.fps,
            window_size=window.size,
            coord_space="norm",
            keypoints=keypoints,
            mean_pose_conf=mean_pose_conf,
            missing_ratio=missing_ratio,
        )
        infer_and_persist(request, db, window_id=pose_window.id)
        logger.info(
            "window_pipeline | auto_infer ok | device_id=%s | track_id=%s | window_id=%s",
            request.device_id,
            person.track_id,
            pose_window.id,
        )
        return {
            "auto_infer_attempted": True,
            "auto_infer_status": "ok",
            "saved": True,
        }
    except OperationalError as e:
        _windows_infer_failed_db += 1
        logger.error(
            "window_pipeline | auto_infer failed_db | device_id=%s | error=%s",
            window_payload.get("device_id"),
            e,
            exc_info=True,
        )
        return {
            "auto_infer_attempted": True,
            "auto_infer_status": "failed_db",
            "saved": False,
        }
    except Exception as e:
        logger.warning(
            "window_pipeline | auto_infer failed_validation | device_id=%s | error=%s",
            window_payload.get("device_id"),
            e,
        )
        return {
            "auto_infer_attempted": True,
            "auto_infer_status": "failed_validation",
            "saved": False,
        }
    finally:
        db.close()
