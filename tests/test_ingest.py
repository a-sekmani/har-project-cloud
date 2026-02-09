"""Tests for POST /v1/windows/ingest (Phase 7.A)."""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import UUID

from app.main import app
from app.database import Base, get_db
from app.models import PoseWindow

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session with tables."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Test client with database override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def _minimal_ingest_payload():
    """Minimal valid ingest body: 30 frames x 17 keypoints x 3 values in [0,1]."""
    import math
    # One frame: 17 keypoints, each [x, y, conf]
    frame = [[0.5 + 0.01 * j, 0.5, 0.9] for j in range(17)]
    keypoints = [frame for _ in range(30)]
    return {
        "device_id": "test-device",
        "camera_id": "cam-1",
        "track_id": 1,
        "ts_start_ms": 1000000,
        "ts_end_ms": 1000999,
        "fps": 30,
        "window_size": 30,
        "keypoints": keypoints,
    }


def test_ingest_window_returns_200_and_id(client: TestClient):
    """POST /v1/windows/ingest with valid body returns 200 and window id."""
    payload = _minimal_ingest_payload()
    r = client.post(
        "/v1/windows/ingest",
        json=payload,
        headers={"X-API-Key": "dev-key"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["device_id"] == "test-device"
    assert data["camera_id"] == "cam-1"
    assert data["track_id"] == 1
    assert data["window_size"] == 30


def test_ingest_without_api_key_returns_401(client: TestClient):
    """POST /v1/windows/ingest without X-API-Key returns 401."""
    r = client.post("/v1/windows/ingest", json=_minimal_ingest_payload())
    assert r.status_code == 401


def test_ingest_invalid_keypoints_shape_returns_422(client: TestClient):
    """Ingest with keypoints length != window_size returns 422."""
    payload = _minimal_ingest_payload()
    payload["keypoints"] = payload["keypoints"][:10]  # only 10 frames
    r = client.post(
        "/v1/windows/ingest",
        json=payload,
        headers={"X-API-Key": "dev-key"},
    )
    assert r.status_code == 422


@pytest.mark.skipif(
    "sqlite" in os.environ.get("DATABASE_URL", "sqlite").lower(),
    reason="Duplicate-id 409 test requires PostgreSQL (SQLite UUID handling differs)",
)
def test_ingest_duplicate_id_returns_409(client: TestClient, db_session):
    """Ingest with an id that already exists returns 409 Conflict (PostgreSQL)."""
    fixed_id = UUID("11111111-2222-3333-4444-555555555555")
    w = PoseWindow(
        id=fixed_id,
        device_id="x",
        camera_id="c",
        track_id=0,
        ts_start_ms=0,
        ts_end_ms=1,
        fps=30,
        window_size=30,
    )
    db_session.add(w)
    db_session.commit()
    payload = _minimal_ingest_payload()
    payload["id"] = str(fixed_id)
    r = client.post(
        "/v1/windows/ingest",
        json=payload,
        headers={"X-API-Key": "dev-key"},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"].lower()
    assert "id" in r.json()["detail"].lower()
