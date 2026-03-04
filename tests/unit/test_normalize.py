"""
Unit tests for app.normalize: normalize_frame_event, InternalFrame.

Verifies: keypoints output in COCO-17 order, undetected points preserved,
ts_unix_ms float->int, two persons -> two InternalFrames, person with !=17 keypoints skipped.

Skipped when app.normalize module is not present (e.g. removed from the project).
"""
import pytest

pytest.importorskip("app.normalize")

from app.edge_schemas import COCO_17_NAMES
from app.normalize import InternalFrame, normalize_frame_event


def _make_keypoints_17_objects(order=None):
    """17 keypoints as {name, x, y, c}. order: list of 17 names (default COCO_17_NAMES)."""
    names = order or list(COCO_17_NAMES)
    return [
        {"name": names[i], "x": 10.0 + i, "y": 20.0 + i, "c": 0.9}
        for i in range(17)
    ]


def _make_body(device_id="dev-1", ts_unix_ms=1000000, persons=None):
    if persons is None:
        persons = [{"track_id": 0, "keypoints": _make_keypoints_17_objects()}]
    return {
        "event_type": "frame_event",
        "source": {"device_id": device_id, "session_id": "s1"},
        "frame": {"ts_unix_ms": ts_unix_ms},
        "persons": persons,
    }


def test_normalize_keypoints_output_coco17_order():
    """Keypoints sent in random order are output in canonical COCO-17 order."""
    # Reverse order in request
    reversed_names = list(reversed(COCO_17_NAMES))
    keypoints = _make_keypoints_17_objects(order=reversed_names)
    body = _make_body(persons=[{"track_id": 0, "keypoints": keypoints}])
    result = normalize_frame_event(body, "cam-1")
    assert len(result) == 1
    iframe = result[0]
    assert len(iframe.keypoints_17x3) == 17
    # First row should be "nose" (first in COCO_17_NAMES); in reversed input nose was last with x=10+16=26
    nose_idx_in_reversed = reversed_names.index("nose")
    expected_nose_xy = (10.0 + nose_idx_in_reversed, 20.0 + nose_idx_in_reversed)
    assert iframe.keypoints_17x3[0][0] == expected_nose_xy[0]
    assert iframe.keypoints_17x3[0][1] == expected_nose_xy[1]
    # Last row should be "right_ankle" (last in COCO_17_NAMES)
    right_ankle_idx = list(COCO_17_NAMES).index("right_ankle")
    assert iframe.keypoints_17x3[16][0] == 10.0 + reversed_names.index("right_ankle")
    assert iframe.keypoints_17x3[16][1] == 20.0 + reversed_names.index("right_ankle")


def test_normalize_undetected_points_preserved():
    """Undetected points (x=-1, y=-1, c=0) are stored as-is in keypoints_17x3."""
    keypoints = _make_keypoints_17_objects()
    keypoints[0] = {"name": "nose", "x": -1.0, "y": -1.0, "c": 0.0}
    keypoints[5] = {"name": "left_shoulder", "x": -1, "y": -1, "c": 0}
    body = _make_body(persons=[{"track_id": 0, "keypoints": keypoints}])
    result = normalize_frame_event(body, "cam-1")
    assert len(result) == 1
    kp = result[0].keypoints_17x3
    assert kp[0] == [-1.0, -1.0, 0.0]
    assert kp[5] == [-1.0, -1.0, 0.0]


def test_normalize_ts_unix_ms_float_converted_to_int():
    """ts_unix_ms as float is converted to int in InternalFrame.ts_ms."""
    body = _make_body(ts_unix_ms=1737970000000.7)
    result = normalize_frame_event(body, "cam-1")
    assert len(result) == 1
    assert result[0].ts_ms == 1737970000000


def test_normalize_two_persons_two_internal_frames():
    """Payload with two persons produces two InternalFrames with same device_id, camera_id, ts_ms."""
    persons = [
        {"track_id": 0, "keypoints": _make_keypoints_17_objects()},
        {"track_id": 1, "keypoints": _make_keypoints_17_objects()},
    ]
    body = _make_body(device_id="two-person", persons=persons)
    result = normalize_frame_event(body, "cam-2")
    assert len(result) == 2
    assert result[0].device_id == "two-person"
    assert result[1].device_id == "two-person"
    assert result[0].camera_id == result[1].camera_id == "cam-2"
    assert result[0].ts_ms == result[1].ts_ms
    assert result[0].track_id == 0
    assert result[1].track_id == 1


def test_normalize_person_with_16_keypoints_skipped():
    """Person with != 17 keypoints is skipped; only valid persons appear in result."""
    persons = [
        {"track_id": 0, "keypoints": _make_keypoints_17_objects()[:16]},
        {"track_id": 1, "keypoints": _make_keypoints_17_objects()},
    ]
    body = _make_body(persons=persons)
    result = normalize_frame_event(body, "cam-1")
    assert len(result) == 1
    assert result[0].track_id == 1


def test_normalize_empty_persons_returns_empty_list():
    """Payload with persons=[] returns empty list."""
    body = _make_body(persons=[])
    result = normalize_frame_event(body, "cam-1")
    assert result == []


def test_normalize_internal_frame_fields():
    """InternalFrame has device_id, camera_id, track_id, ts_ms, keypoints_17x3."""
    body = _make_body(device_id="d1", ts_unix_ms=999)
    result = normalize_frame_event(body, "camera-x")
    assert len(result) == 1
    iframe = result[0]
    assert iframe.device_id == "d1"
    assert iframe.camera_id == "camera-x"
    assert iframe.track_id == 0
    assert iframe.ts_ms == 999
    assert len(iframe.keypoints_17x3) == 17
    assert all(len(row) == 3 for row in iframe.keypoints_17x3)
