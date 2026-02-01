"""
Normalization layer: edge frame_event payload -> internal frame format.

Converts keypoints from 17 objects {name, x, y, c} to 17×3 [[x, y, c], ...]
in fixed COCO-17 order. Undetected points (x/y = -1 or c = 0) are stored as-is.
"""
from dataclasses import dataclass
from typing import Any, List

from app.edge_schemas import COCO_17_NAMES


@dataclass
class InternalFrame:
    """Single frame for one person after normalization (cloud-internal format)."""
    device_id: str
    camera_id: str
    track_id: int
    ts_ms: int
    keypoints_17x3: List[List[float]]  # 17 rows in COCO-17 order, each [x, y, c]


def _get_kp_value(kp: Any, key: str, default: float = 0.0) -> float:
    if isinstance(kp, dict):
        return float(kp.get(key, default))
    return float(getattr(kp, key, default))


def normalize_frame_event(body: dict, camera_id: str) -> List[InternalFrame]:
    """
    Convert edge frame_event body to a list of InternalFrame (one per person).

    Keypoints are converted from [{name, x, y, c}, ...] to [[x, y, c], ...]
    in fixed COCO-17 order (order in request is ignored). Undetected points
    (x=-1, y=-1, or c=0) are kept as-is.

    Args:
        body: Validated edge payload (event_type, source, frame, persons).
        camera_id: Resolved camera_id (priority: source -> query -> header -> default).

    Returns:
        List of InternalFrame; persons with != 17 keypoints or wrong names are skipped.
    """
    source = body.get("source", {})
    frame = body.get("frame", {})
    persons = body.get("persons", [])

    device_id = source.get("device_id", "")
    ts_unix_ms = frame.get("ts_unix_ms", 0)
    ts_ms = int(ts_unix_ms)

    result: List[InternalFrame] = []
    for person in persons:
        kps = person.get("keypoints", [])
        if len(kps) != 17:
            continue
        # Build name -> [x, y, c]; then output in COCO-17 order
        by_name: dict[str, List[float]] = {}
        for kp in kps:
            name = kp.get("name", "") if isinstance(kp, dict) else getattr(kp, "name", "")
            x = _get_kp_value(kp, "x")
            y = _get_kp_value(kp, "y")
            c = _get_kp_value(kp, "c")
            by_name[name] = [x, y, c]
        keypoints_17x3: List[List[float]] = []
        for name in COCO_17_NAMES:
            keypoints_17x3.append(by_name.get(name, [-1.0, -1.0, 0.0]))

        result.append(
            InternalFrame(
                device_id=device_id,
                camera_id=camera_id,
                track_id=int(person.get("track_id", 0)),
                ts_ms=ts_ms,
                keypoints_17x3=keypoints_17x3,
            )
        )
    return result
