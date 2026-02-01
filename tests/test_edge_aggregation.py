"""
Tests for POST /v1/edge/events and frame_event aggregation.

Verifies: valid frame_event (event_type "frame_event", keypoints as 17 objects)
accepted (202), invalid rejected (422), and sending window.size frames completes
a window. Debug endpoints GET /debug/buffers and GET /debug/windows.
"""
from typing import Optional, Union

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import API_KEY, EDGE_WINDOW_SIZE
from app.aggregation import (
    get_buffer_details,
    get_frame_events_received_count,
    get_last_windows,
    reset_aggregation_state,
)
from app.edge_schemas import COCO_17_NAMES


@pytest.fixture
def client():
    """Test client; edge/events does not use DB."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_aggregation():
    """Reset aggregation state before each test so tests don't share buffers/windows."""
    reset_aggregation_state()


def _make_keypoints_17_objects(names=None):
    """17 keypoints as objects {name, x, y, c}. names: list of 17 names (default COCO_17_NAMES)."""
    names = names or list(COCO_17_NAMES)
    return [
        {"name": names[i], "x": 0.5 + i * 0.01, "y": 0.2 + i * 0.01, "c": 0.9}
        for i in range(17)
    ]


def _make_frame_event(
    device_id: str = "edge-1",
    camera_id: Optional[str] = None,
    session_id: str = "sess-1",
    track_id: int = 0,
    ts_ms: Optional[Union[int, float]] = 1000000,
    persons: Optional[list] = None,
):
    """One frame_event; persons default one person with 17 COCO keypoints. ts_ms can be int or float."""
    source = {"device_id": device_id, "session_id": session_id}
    if camera_id is not None:
        source["camera_id"] = camera_id
    if persons is None:
        persons = [{"track_id": track_id, "keypoints": _make_keypoints_17_objects()}]
    return {
        "event_type": "frame_event",
        "source": source,
        "frame": {"ts_unix_ms": ts_ms},
        "persons": persons,
    }


def test_edge_events_accepts_valid_frame_event(client):
    """POST valid frame_event (event_type frame_event, keypoints as objects) returns 202."""
    body = _make_frame_event()
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


def test_edge_events_rejects_wrong_event_type(client):
    """POST with event_type != frame_event returns 422."""
    body = _make_frame_event()
    body["event_type"] = "frame"
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_rejects_missing_event_type(client):
    """POST without event_type returns 422."""
    body = _make_frame_event()
    del body["event_type"]
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_rejects_missing_source_device_id(client):
    """POST without source.device_id returns 422."""
    body = _make_frame_event()
    body["source"] = {"session_id": "sess-1", "camera_id": "cam-1"}
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_rejects_missing_source_session_id(client):
    """POST without source.session_id returns 422."""
    body = _make_frame_event()
    body["source"] = {"device_id": "edge-1", "camera_id": "cam-1"}
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_rejects_missing_frame_ts(client):
    """POST without frame.ts_unix_ms returns 422."""
    body = _make_frame_event()
    body["frame"] = {}
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_accepts_float_ts_unix_ms(client):
    """POST with frame.ts_unix_ms as float (e.g. 1700000123456.0) returns 202, not 422."""
    body = _make_frame_event(ts_ms=1700000123456.0)
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


def test_edge_events_persons_required(client):
    """POST without 'persons' field returns 422 (persons is required)."""
    body = _make_frame_event()
    del body["persons"]
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_persons_can_be_empty(client):
    """POST with persons=[] (required field, empty list) returns 202."""
    body = _make_frame_event(persons=[])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


def test_edge_events_rejects_wrong_keypoint_names(client):
    """POST with keypoints names not equal to COCO-17 set returns 422."""
    wrong_names = list(COCO_17_NAMES)
    wrong_names[0] = "invalid_name"
    keypoints = _make_keypoints_17_objects(names=wrong_names)
    body = _make_frame_event(persons=[{"track_id": 0, "keypoints": keypoints}])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_requires_api_key(client):
    """POST /v1/edge/events without X-API-Key returns 401."""
    body = _make_frame_event()
    response = client.post("/v1/edge/events", json=body)
    assert response.status_code == 401


def test_edge_events_rejects_invalid_api_key(client):
    """POST /v1/edge/events with wrong X-API-Key returns 401."""
    body = _make_frame_event()
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_edge_events_accepts_extra_fields(client):
    """POST with extra fields (bbox, score) returns 202; payload not rejected."""
    body = _make_frame_event()
    body["bbox"] = [0, 0, 640, 480]
    body["score"] = 0.95
    body["persons"][0]["bbox"] = [100, 100, 200, 300]
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


def test_camera_id_from_source_priority(client):
    """camera_id priority 1: source.camera_id wins over query and header."""
    body = _make_frame_event(camera_id="from-source")
    response = client.post(
        "/v1/edge/events",
        json=body,
        params={"camera_id": "from-query"},
        headers={"X-API-Key": API_KEY, "X-Camera-Id": "from-header"},
    )
    assert response.status_code == 202
    details = get_buffer_details()
    assert any("from-source" in b["key"] for b in details)


def test_camera_id_from_header_when_no_source_no_query(client):
    """camera_id from X-Camera-Id when source and query do not provide it."""
    body = _make_frame_event(camera_id=None)
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY, "X-Camera-Id": "from-header-only"},
    )
    assert response.status_code == 202
    details = get_buffer_details()
    assert any("from-header-only" in b["key"] for b in details)


def test_camera_id_default_when_none_provided(client):
    """Default camera_id (e.g. cam-1) when no source, query, or header."""
    body = _make_frame_event(camera_id=None)
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    details = get_buffer_details()
    assert any("cam-1" in b["key"] for b in details)


def test_edge_events_rejects_keypoint_c_out_of_range(client):
    """POST with keypoint c > 1 or c < 0 returns 422."""
    keypoints_high = _make_keypoints_17_objects()
    keypoints_high[0]["c"] = 1.5
    body = _make_frame_event(persons=[{"track_id": 0, "keypoints": keypoints_high}])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422

    keypoints_low = _make_keypoints_17_objects()
    keypoints_low[0]["c"] = -0.1
    body = _make_frame_event(persons=[{"track_id": 0, "keypoints": keypoints_low}])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_edge_events_rejects_keypoints_count_not_17(client):
    """POST with 16 or 18 keypoints per person returns 422."""
    kp_16 = _make_keypoints_17_objects()[:16]
    body = _make_frame_event(persons=[{"track_id": 0, "keypoints": kp_16}])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422

    kp_18 = _make_keypoints_17_objects() + [{"name": "nose", "x": 0, "y": 0, "c": 0.5}]
    body = _make_frame_event(persons=[{"track_id": 0, "keypoints": kp_18}])
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_camera_id_priority_query_before_header(client):
    """camera_id priority: 1 source, 2 query, 3 header, 4 default. Query wins over header."""
    body = _make_frame_event(camera_id=None)  # no source.camera_id
    response = client.post(
        "/v1/edge/events",
        json=body,
        params={"camera_id": "from-query"},
        headers={"X-API-Key": API_KEY, "X-Camera-Id": "from-header"},
    )
    assert response.status_code == 202
    details = get_buffer_details()
    assert any("from-query" in b["key"] for b in details), "buffer key should contain camera_id from query"


def test_edge_events_aggregation_completes_window(client):
    """Send EDGE_WINDOW_SIZE frame_events for same person; all 202, window built and valid."""
    base_ts = 2000000
    for i in range(EDGE_WINDOW_SIZE):
        body = _make_frame_event(ts_ms=base_ts + i * 33)
        response = client.post(
            "/v1/edge/events",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 202
    assert get_frame_events_received_count() >= EDGE_WINDOW_SIZE
    windows = get_last_windows(10)
    assert len(windows) >= 1
    w = windows[-1]
    assert w["size"] == EDGE_WINDOW_SIZE
    assert w["ts_start_ms"] == base_ts
    assert w["ts_end_ms"] == base_ts + (EDGE_WINDOW_SIZE - 1) * 33


def test_one_frame_two_persons_two_buffers(client):
    """One request with persons=[track_id=0, track_id=1] creates two buffers."""
    persons = [
        {"track_id": 0, "keypoints": _make_keypoints_17_objects()},
        {"track_id": 1, "keypoints": _make_keypoints_17_objects()},
    ]
    body = _make_frame_event(device_id="multi-dev", persons=persons)
    response = client.post(
        "/v1/edge/events",
        json=body,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    details = get_buffer_details()
    keys = [b["key"] for b in details]
    assert any("multi-dev" in k and "|0" in k for k in keys)
    assert any("multi-dev" in k and "|1" in k for k in keys)
    # One HTTP request with 2 persons = 2 ingest_internal_frame calls = count 2
    assert get_frame_events_received_count() == 2


def test_thirty_frames_two_persons_two_windows(client):
    """30 frames for track_id=0 and 30 for track_id=1 completes two windows."""
    base_ts = 3000000
    for i in range(EDGE_WINDOW_SIZE):
        body = _make_frame_event(
            device_id="two-track",
            ts_ms=base_ts + i * 33,
            persons=[
                {"track_id": 0, "keypoints": _make_keypoints_17_objects()},
                {"track_id": 1, "keypoints": _make_keypoints_17_objects()},
            ],
        )
        response = client.post(
            "/v1/edge/events",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 202
    windows = get_last_windows(10)
    track_ids = {w["track_id"] for w in windows}
    assert 0 in track_ids and 1 in track_ids
    assert len([w for w in windows if w["device_id"] == "two-track"]) >= 2


def test_completed_window_has_expected_structure(client):
    """Completed window metadata has device_id, camera_id, track_id, ts_start_ms, ts_end_ms, size, fps."""
    base_ts = 4000000
    for i in range(EDGE_WINDOW_SIZE):
        body = _make_frame_event(device_id="struct-dev", ts_ms=base_ts + i * 33)
        response = client.post(
            "/v1/edge/events",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 202
    windows = get_last_windows(1)
    assert len(windows) == 1
    w = windows[0]
    for key in ("device_id", "camera_id", "track_id", "ts_start_ms", "ts_end_ms", "size", "fps"):
        assert key in w
    assert w["device_id"] == "struct-dev"
    assert w["size"] == EDGE_WINDOW_SIZE
    assert 0 <= w.get("fps", 0) <= 120
    assert w["ts_end_ms"] >= w["ts_start_ms"]


def test_completed_window_has_auto_infer_metadata(client):
    """Completed window metadata includes auto_infer_attempted, auto_infer_status, saved."""
    base_ts = 5000000
    for i in range(EDGE_WINDOW_SIZE):
        body = _make_frame_event(device_id="auto-meta", ts_ms=base_ts + i * 33)
        response = client.post(
            "/v1/edge/events",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 202
    windows = get_last_windows(1)
    assert len(windows) == 1
    w = windows[0]
    assert "auto_infer_attempted" in w
    assert "auto_infer_status" in w
    assert "saved" in w
    assert w["auto_infer_status"] in ("ok", "disabled", "failed_db", "failed_validation")


def test_auto_infer_disabled(client):
    """With EDGE_AUTO_INFER disabled (default), completed window has auto_infer_status=disabled."""
    import app.window_pipeline as wp
    original = wp.EDGE_AUTO_INFER
    wp.EDGE_AUTO_INFER = False
    try:
        base_ts = 6000000
        for i in range(EDGE_WINDOW_SIZE):
            body = _make_frame_event(device_id="no-infer", ts_ms=base_ts + i * 33)
            response = client.post(
                "/v1/edge/events",
                json=body,
                headers={"X-API-Key": API_KEY},
            )
            assert response.status_code == 202
        windows = get_last_windows(1)
        assert len(windows) == 1
        assert windows[0]["auto_infer_attempted"] is False
        assert windows[0]["auto_infer_status"] == "disabled"
        assert windows[0]["saved"] is False
    finally:
        wp.EDGE_AUTO_INFER = original


def test_auto_infer_enabled_event_saved(client):
    """With EDGE_AUTO_INFER=True, completing a window creates an ActivityEvent (standing)."""
    from app.database import SessionLocal, engine, Base
    from app.models import ActivityEvent
    import app.window_pipeline as wp

    Base.metadata.create_all(bind=engine)
    original = wp.EDGE_AUTO_INFER
    wp.EDGE_AUTO_INFER = True
    try:
        base_ts = 7000000
        for i in range(EDGE_WINDOW_SIZE):
            body = _make_frame_event(device_id="auto-infer-dev", ts_ms=base_ts + i * 33)
            response = client.post(
                "/v1/edge/events",
                json=body,
                headers={"X-API-Key": API_KEY},
            )
            assert response.status_code == 202
        windows = get_last_windows(1)
        assert len(windows) == 1
        assert windows[0]["auto_infer_status"] == "ok"
        assert windows[0]["saved"] is True
        db = SessionLocal()
        try:
            events = db.query(ActivityEvent).filter(
                ActivityEvent.device_id == "auto-infer-dev"
            ).all()
            assert len(events) >= 1
            assert events[0].activity == "standing"
            assert events[0].confidence == 0.6
        finally:
            db.close()
    finally:
        wp.EDGE_AUTO_INFER = original


def test_auto_infer_enabled_low_c_unknown(client):
    """With EDGE_AUTO_INFER=True and low keypoint confidence, activity is unknown."""
    def low_c_keypoints():
        names = list(COCO_17_NAMES)
        return [
            {"name": names[i], "x": 0.5 + i * 0.01, "y": 0.2 + i * 0.01, "c": 0.1}
            for i in range(17)
        ]

    from app.database import SessionLocal, engine, Base
    from app.models import ActivityEvent
    import app.window_pipeline as wp

    Base.metadata.create_all(bind=engine)
    original = wp.EDGE_AUTO_INFER
    wp.EDGE_AUTO_INFER = True
    try:
        base_ts = 8000000
        persons = [{"track_id": 0, "keypoints": low_c_keypoints()}]
        for i in range(EDGE_WINDOW_SIZE):
            body = _make_frame_event(
                device_id="low-c-dev", ts_ms=base_ts + i * 33, persons=persons
            )
            response = client.post(
                "/v1/edge/events",
                json=body,
                headers={"X-API-Key": API_KEY},
            )
            assert response.status_code == 202
        db = SessionLocal()
        try:
            events = db.query(ActivityEvent).filter(
                ActivityEvent.device_id == "low-c-dev"
            ).all()
            assert len(events) >= 1
            assert events[0].activity == "unknown"
            assert events[0].confidence == 0.2
        finally:
            db.close()
    finally:
        wp.EDGE_AUTO_INFER = original


def test_db_failure_does_not_break_ingestion(client):
    """When infer_and_persist raises OperationalError, edge/events still returns 202 and status is failed_db."""
    from sqlalchemy.exc import OperationalError
    import app.window_pipeline as wp

    def failing_infer(_request, _db):
        raise OperationalError("", "", "connection refused")

    original_infer = wp.infer_and_persist
    wp.infer_and_persist = failing_infer
    original_auto = wp.EDGE_AUTO_INFER
    wp.EDGE_AUTO_INFER = True
    try:
        base_ts = 9000000
        for i in range(EDGE_WINDOW_SIZE):
            body = _make_frame_event(device_id="fail-db-dev", ts_ms=base_ts + i * 33)
            response = client.post(
                "/v1/edge/events",
                json=body,
                headers={"X-API-Key": API_KEY},
            )
            assert response.status_code == 202
        windows = get_last_windows(1)
        assert len(windows) == 1
        assert windows[0]["auto_infer_status"] == "failed_db"
        assert windows[0]["saved"] is False
    finally:
        wp.infer_and_persist = original_infer
        wp.EDGE_AUTO_INFER = original_auto


def test_debug_windows_n_validation(client):
    """GET /debug/windows with n=0 or n=101 returns 422 (ge=1, le=100)."""
    r0 = client.get("/debug/windows", params={"n": 0}, headers={"X-API-Key": API_KEY})
    assert r0.status_code == 422
    r101 = client.get("/debug/windows", params={"n": 101}, headers={"X-API-Key": API_KEY})
    assert r101.status_code == 422


def test_debug_buffers_returns_structure(client):
    """GET /debug/buffers returns buffers, frame_events_received, windows_infer_failed_db."""
    response = client.get("/debug/buffers", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert "buffers" in data
    assert "frame_events_received" in data
    assert "windows_infer_failed_db" in data
    assert isinstance(data["buffers"], list)


def test_debug_windows_returns_structure(client):
    """GET /debug/windows returns windows list (metadata only)."""
    response = client.get("/debug/windows", params={"n": 5}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert "windows" in data
    assert isinstance(data["windows"], list)


def test_debug_buffers_requires_api_key(client):
    """GET /debug/buffers without API key returns 401."""
    response = client.get("/debug/buffers")
    assert response.status_code == 401


def test_debug_windows_requires_api_key(client):
    """GET /debug/windows without API key returns 401."""
    response = client.get("/debug/windows")
    assert response.status_code == 401
