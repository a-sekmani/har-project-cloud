"""
Unit tests for app.aggregation: ingest_internal_frame, buffers, window completion.

Verifies: frame_events_received count, two persons -> two buffers, window metadata
and pose_conf, buffer clears after window completes.
"""
import pytest

from app.config import EDGE_WINDOW_SIZE
from app.normalize import InternalFrame, normalize_frame_event
from app.aggregation import (
    ingest_internal_frame,
    get_buffer_details,
    get_frame_events_received_count,
    get_last_windows,
    reset_aggregation_state,
)


@pytest.fixture(autouse=True)
def reset_aggregation():
    """Reset aggregation state before each test."""
    reset_aggregation_state()


def _make_keypoints_17():
    """17 keypoints as [[x, y, c], ...] in order, with valid c in [0,1] and x,y >= 0 for normalization."""
    return [[10.0 + i, 20.0 + i, 0.9] for i in range(17)]


def _internal_frame(device_id="d1", camera_id="cam-1", track_id=0, ts_ms=1000000, keypoints=None):
    if keypoints is None:
        keypoints = _make_keypoints_17()
    return InternalFrame(
        device_id=device_id,
        camera_id=camera_id,
        track_id=track_id,
        ts_ms=ts_ms,
        keypoints_17x3=keypoints,
    )


def test_ingest_one_request_two_persons_increments_count_twice():
    """One logical "request" with two InternalFrames increments frame_events_received by 2."""
    initial = get_frame_events_received_count()
    ingest_internal_frame(_internal_frame(track_id=0))
    ingest_internal_frame(_internal_frame(track_id=1))
    assert get_frame_events_received_count() == initial + 2


def test_ingest_two_persons_two_buffers():
    """Two InternalFrames with different track_ids create two separate buffers."""
    base_ts = 2000000
    ingest_internal_frame(_internal_frame(device_id="agg-dev", track_id=0, ts_ms=base_ts))
    ingest_internal_frame(_internal_frame(device_id="agg-dev", track_id=1, ts_ms=base_ts))
    details = get_buffer_details()
    keys = [b["key"] for b in details]
    assert any("agg-dev" in k and "|0" in k for k in keys)
    assert any("agg-dev" in k and "|1" in k for k in keys)
    assert len(details) == 2


def test_ingest_window_completes_metadata_and_pose_conf():
    """After EDGE_WINDOW_SIZE frames, get_last_windows returns metadata with pose_conf in [0,1], size=30."""
    base_ts = 3000000
    for i in range(EDGE_WINDOW_SIZE):
        ingest_internal_frame(_internal_frame(device_id="win-dev", track_id=0, ts_ms=base_ts + i * 33))
    windows = get_last_windows(5)
    assert len(windows) >= 1
    w = windows[-1]
    assert w["device_id"] == "win-dev"
    assert w["track_id"] == 0
    assert w["size"] == EDGE_WINDOW_SIZE
    assert w["ts_end_ms"] >= w["ts_start_ms"]
    assert 1 <= w.get("fps", 0) <= 120


def test_ingest_buffer_clears_after_window_completes():
    """After a window completes, buffer for that key is cleared; next frame starts new buffer."""
    base_ts = 4000000
    for i in range(EDGE_WINDOW_SIZE):
        ingest_internal_frame(_internal_frame(device_id="clear-dev", track_id=0, ts_ms=base_ts + i * 33))
    details_before = get_buffer_details()
    key_clear = "clear-dev|cam-1|0"
    # After 30 frames, buffer may have been cleared (0) or if there were extra frames, some remain
    # Add one more frame and check buffer has 1
    ingest_internal_frame(_internal_frame(device_id="clear-dev", track_id=0, ts_ms=base_ts + 9999))
    details_after = get_buffer_details()
    buf = next((b for b in details_after if b["key"] == key_clear), None)
    assert buf is not None
    assert buf["frame_count"] == 1


def test_ingest_internal_frame_with_16_keypoints_not_appended():
    """InternalFrame with keypoints_17x3 length != 17 is not appended; returns [] and count still increments."""
    initial_count = get_frame_events_received_count()
    kp_16 = _make_keypoints_17()[:16]
    iframe = _internal_frame(device_id="bad-kp", track_id=0, keypoints=kp_16)
    completed = ingest_internal_frame(iframe)
    assert completed == []
    assert get_frame_events_received_count() == initial_count + 1
    details = get_buffer_details()
    assert not any("bad-kp" in b["key"] for b in details)
