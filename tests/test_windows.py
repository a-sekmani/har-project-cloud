"""
Tests for Phase 5: GET /v1/windows, GET /v1/windows/{id}, POST /v1/windows/{id}/label.

Verifies: windows list with filters and include_keypoints; get by id; label API
(valid label 200, invalid 422, not found 404); API key required.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import API_KEY
from app.database import Base, get_db
from app.models import PoseWindow
from app.services import create_pose_window, update_window_label

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def one_window(db_session):
    """Create one PoseWindow in DB for tests that need it."""
    keypoints = [[[0.5 + t * 0.01, 0.2, 0.9] for _ in range(17)] for t in range(30)]
    pw = create_pose_window(
        db=db_session,
        device_id="dev-1",
        camera_id="cam-1",
        session_id="sess-1",
        track_id=0,
        ts_start_ms=1000000,
        ts_end_ms=1001000,
        fps=30,
        window_size=30,
        coord_space="norm",
        keypoints=keypoints,
        mean_pose_conf=0.9,
        missing_ratio=0.0,
    )
    return pw


def test_get_windows_empty(client):
    """GET /v1/windows with no data returns empty list."""
    r = client.get("/v1/windows", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert "windows" in data
    assert data["windows"] == []


def test_get_windows_after_infer(client):
    """POST /v1/activity/infer creates PoseWindow; GET /v1/windows returns it without keypoints by default."""
    payload = {
        "schema_version": 1,
        "device_id": "pi-1",
        "camera_id": "cam-1",
        "window": {"ts_start_ms": 2000000, "ts_end_ms": 2001000, "fps": 30, "size": 30},
        "people": [{
            "track_id": 0,
            "keypoints": [[[0.5, 0.2, 0.9] for _ in range(17)] for _ in range(30)],
            "pose_conf": 0.9,
        }],
    }
    r_infer = client.post("/v1/activity/infer", json=payload, headers={"X-API-Key": API_KEY})
    assert r_infer.status_code == 200

    r = client.get("/v1/windows", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert len(data["windows"]) >= 1
    w = data["windows"][0]
    assert w["device_id"] == "pi-1"
    assert w["camera_id"] == "cam-1"
    assert "id" in w
    assert "keypoints" not in w


def test_get_windows_include_keypoints(client, one_window):
    """GET /v1/windows?include_keypoints=1 returns keypoints in each item."""
    r = client.get("/v1/windows", params={"include_keypoints": 1}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert len(data["windows"]) >= 1
    w = next(x for x in data["windows"] if x["id"] == str(one_window.id))
    assert "keypoints" in w
    assert len(w["keypoints"]) == 30
    assert len(w["keypoints"][0]) == 17


def test_get_window_by_id(client, one_window):
    """GET /v1/windows/{id} returns single window with keypoints."""
    r = client.get(f"/v1/windows/{one_window.id}", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == str(one_window.id)
    assert data["device_id"] == "dev-1"
    assert "keypoints" in data
    assert len(data["keypoints"]) == 30


def test_get_window_by_id_not_found(client):
    """GET /v1/windows/{id} with nonexistent id returns 404."""
    r = client.get("/v1/windows/00000000-0000-0000-0000-000000000000", headers={"X-API-Key": API_KEY})
    assert r.status_code == 404


def test_label_valid(client, one_window):
    """POST /v1/windows/{id}/label with valid label returns 200 and updates window."""
    r = client.post(
        f"/v1/windows/{one_window.id}/label",
        json={"label": "moving", "label_source": "manual"},
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "moving"
    assert data["label_source"] == "manual"
    assert "labeled_at" in data


def test_label_invalid(client, one_window):
    """POST /v1/windows/{id}/label with label not in LABELS_ALLOWED returns 422."""
    r = client.post(
        f"/v1/windows/{one_window.id}/label",
        json={"label": "invalid_activity", "label_source": "manual"},
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 422
    assert "Label must be one of" in r.json()["detail"]


def test_label_not_found(client):
    """POST /v1/windows/{id}/label with nonexistent window_id returns 404."""
    r = client.post(
        "/v1/windows/00000000-0000-0000-0000-000000000000/label",
        json={"label": "standing", "label_source": "manual"},
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 404


def test_windows_require_api_key(client, one_window):
    """GET /v1/windows and GET /v1/windows/{id} without X-API-Key return 401."""
    r1 = client.get("/v1/windows")
    assert r1.status_code == 401
    r2 = client.get(f"/v1/windows/{one_window.id}")
    assert r2.status_code == 401


def test_label_requires_api_key(client, one_window):
    """POST /v1/windows/{id}/label without X-API-Key returns 401."""
    r = client.post(
        f"/v1/windows/{one_window.id}/label",
        json={"label": "standing", "label_source": "manual"},
    )
    assert r.status_code == 401


def test_get_windows_filter_by_label(client, one_window, db_session):
    """GET /v1/windows?label=X returns only windows with that label."""
    update_window_label(db_session, one_window.id, "standing", "manual")

    r = client.get("/v1/windows", params={"label": "standing"}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert all(w["label"] == "standing" for w in data["windows"])
