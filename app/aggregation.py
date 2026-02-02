"""
Edge frame_event aggregation: in-memory buffers per (device_id, camera_id, track_id),
build Cloud window payload when buffer reaches window.size. No DB, no HAR call.

Accepts InternalFrame (normalized from edge payload). pose_conf = mean of c over
all frames and all 17 keypoints in the window.
"""
import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from app.config import EDGE_CAMERA_ID_DEFAULT, EDGE_WINDOW_SIZE
from app.normalize import InternalFrame
from app.schemas import InferenceRequestSchema
from app.window_pipeline import handle_completed_window, reset_windows_infer_failed_db_count

logger = logging.getLogger("cloud_har.aggregation")

# In-memory: key = (device_id, camera_id, track_id), value = list of frame entries
# Each frame entry: {"ts_unix_ms": int, "keypoints": [[x,y,c], ... 17], "session_id": str}
_buffers: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

# Count of frame_events received (for logging)
_frame_events_received = 0

# Last N completed windows (metadata only) for GET /debug/windows
_DEBUG_WINDOWS_MAX = 100
_completed_windows_meta: deque = deque(maxlen=_DEBUG_WINDOWS_MAX)

K_COCO = 17


def reset_aggregation_state() -> None:
    """Clear buffers, frame count, and completed windows. For tests only."""
    global _buffers, _frame_events_received, _completed_windows_meta
    _buffers = {}
    _frame_events_received = 0
    _completed_windows_meta = deque(maxlen=_DEBUG_WINDOWS_MAX)
    reset_windows_infer_failed_db_count()


def _normalize_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Normalize keypoints to [0, 1] for x, y. If max > 1, scale down. Clamp confidence to [0,1]."""
    if not keypoints:
        return keypoints
    flat = [v for kp in keypoints for i, v in enumerate(kp) if i < 2 and kp[0] >= 0 and kp[1] >= 0]
    max_xy = max(flat) if flat else 1.0
    scale = 1.0 if max_xy <= 1.0 else max_xy
    out = []
    for kp in keypoints:
        if len(kp) < 3:
            out.append([0.0, 0.0, 0.0])
            continue
        x, y = kp[0], kp[1]
        if x < 0 or y < 0:
            x, y = 0.0, 0.0
        else:
            x, y = x / scale, y / scale
        conf = max(0.0, min(1.0, float(kp[2])))
        out.append([x, y, conf])
    return out


def _validate_and_build_window(
    device_id: str,
    camera_id: str,
    session_id: str,
    track_id: int,
    frames: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build Cloud window payload from buffer. Validate before returning.
    pose_conf = mean of c over all frames and all 17 keypoints.
    """
    if len(frames) != EDGE_WINDOW_SIZE:
        logger.warning(
            "aggregation window rejected | reason=size_mismatch | "
            "device_id=%s | track_id=%s | got=%s | expected=%s",
            device_id, track_id, len(frames), EDGE_WINDOW_SIZE,
        )
        return None

    if not (10 <= EDGE_WINDOW_SIZE <= 120):
        logger.warning(
            "aggregation window rejected | reason=window_size_out_of_range | size=%s",
            EDGE_WINDOW_SIZE,
        )
        return None

    ts_start_ms = frames[0]["ts_unix_ms"]
    ts_end_ms = frames[-1]["ts_unix_ms"]
    if ts_end_ms <= ts_start_ms:
        logger.warning(
            "aggregation window rejected | reason=ts_end_not_after_ts_start | "
            "device_id=%s | track_id=%s | ts_start=%s | ts_end=%s",
            device_id, track_id, ts_start_ms, ts_end_ms,
        )
        return None

    keypoints_list: List[List[List[float]]] = []
    conf_sum = 0.0
    conf_count = 0

    for f in frames:
        kp = f.get("keypoints")
        if not isinstance(kp, list) or len(kp) != K_COCO:
            logger.warning(
                "aggregation window rejected | reason=keypoints_not_17 | "
                "device_id=%s | track_id=%s | frame_len=%s",
                device_id, track_id, len(kp) if isinstance(kp, list) else "not_list",
            )
            return None
        for row in kp:
            if len(row) >= 3:
                conf_sum += float(row[2])
                conf_count += 1
        normalized = _normalize_keypoints(kp)
        for n in normalized:
            if len(n) != 3 or not (0 <= n[2] <= 1):
                logger.warning(
                    "aggregation window rejected | reason=confidence_out_of_range | "
                    "device_id=%s | track_id=%s",
                    device_id, track_id,
                )
                return None
        keypoints_list.append(normalized)

    # pose_conf = mean of c over all frames and all 17 points
    pose_conf = conf_sum / conf_count if conf_count else 0.0
    pose_conf = max(0.0, min(1.0, pose_conf))

    duration_sec = (ts_end_ms - ts_start_ms) / 1000.0
    fps = int(round((len(frames) - 1) / duration_sec)) if duration_sec > 0 else 30
    fps = max(1, min(120, fps))

    window_payload = {
        "schema_version": 1,
        "device_id": device_id,
        "camera_id": camera_id,
        "session_id": session_id or None,
        "window": {
            "ts_start_ms": ts_start_ms,
            "ts_end_ms": ts_end_ms,
            "fps": fps,
            "size": EDGE_WINDOW_SIZE,
        },
        "people": [
            {
                "track_id": track_id,
                "keypoints": keypoints_list,
                "pose_conf": round(pose_conf, 4),
            }
        ],
    }

    try:
        InferenceRequestSchema.model_validate(window_payload)
    except Exception as e:
        logger.warning(
            "aggregation window rejected | reason=schema_validation_failed | "
            "device_id=%s | track_id=%s | error=%s",
            device_id, track_id, str(e),
        )
        return None

    return window_payload


def get_frame_events_received_count() -> int:
    """Return total number of frame_events received (for logging)."""
    return _frame_events_received


def get_buffer_sizes() -> Dict[str, int]:
    """Return current buffer sizes keyed by 'device_id|camera_id|track_id'."""
    return {
        f"{d}|{c}|{t}": len(buf)
        for (d, c, t), buf in _buffers.items()
    }


def get_buffer_details() -> List[Dict[str, Any]]:
    """Return buffer details for GET /debug/buffers: key, frame_count, last_ts_ms."""
    out = []
    for (d, c, t), buf in _buffers.items():
        key = f"{d}|{c}|{t}"
        last_ts = buf[-1]["ts_unix_ms"] if buf else None
        out.append({"key": key, "frame_count": len(buf), "last_ts_ms": last_ts})
    return out


def get_last_windows(n: int = 20) -> List[Dict[str, Any]]:
    """Return last n completed windows (metadata only) for GET /debug/windows."""
    items = list(_completed_windows_meta)
    return items[-n:] if n < len(items) else items


def ingest_internal_frame(iframe: InternalFrame) -> List[Dict[str, Any]]:
    """
    Ingest one InternalFrame. Append to per-person buffer.
    When buffer reaches window.size, build window payload, clear buffer, store metadata.
    Returns list of completed window payloads (0 or 1 per person that just completed).
    """
    global _frame_events_received
    _frame_events_received += 1

    device_id = iframe.device_id
    camera_id = iframe.camera_id
    track_id = iframe.track_id
    ts_ms = iframe.ts_ms
    keypoints_17x3 = iframe.keypoints_17x3

    if len(keypoints_17x3) != K_COCO:
        return []

    key = (device_id, camera_id, track_id)
    if key not in _buffers:
        _buffers[key] = []
    buf = _buffers[key]
    buf.append({
        "ts_unix_ms": ts_ms,
        "keypoints": [list(row) for row in keypoints_17x3],
        "session_id": getattr(iframe, "session_id", "") or "",
    })

    completed_windows: List[Dict[str, Any]] = []
    if len(buf) >= EDGE_WINDOW_SIZE:
        frames = buf[:EDGE_WINDOW_SIZE]
        _buffers[key] = buf[EDGE_WINDOW_SIZE:]
        session_id = frames[0].get("session_id", "") if frames else ""
        window_payload = _validate_and_build_window(
            device_id, camera_id, session_id, track_id, frames
        )
        if window_payload:
            completed_windows.append(window_payload)
            w = window_payload["window"]
            meta = {
                "device_id": device_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "ts_start_ms": w["ts_start_ms"],
                "ts_end_ms": w["ts_end_ms"],
                "size": w["size"],
                "fps": w["fps"],
            }
            status = handle_completed_window(window_payload)
            meta.update(status)
            _completed_windows_meta.append(meta)
            logger.info(
                "aggregation window complete | device_id=%s | track_id=%s | "
                "window_size=%s | ts_start_ms=%s | ts_end_ms=%s",
                device_id, track_id,
                w["size"], w["ts_start_ms"], w["ts_end_ms"],
            )
            logger.info("aggregation window payload (JSON): %s", json.dumps(window_payload))

    return completed_windows
