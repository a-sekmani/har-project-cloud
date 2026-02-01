"""
Tests using the golden edge payload (K=17, 30 frames).

Verifies that POST /v1/activity/infer accepts the edge-style payload and
returns debug/diagnostic fields: frames_received, k_count, avg_pose_conf,
frames_ok_ratio, latency_ms.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import API_KEY
from app.database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create test client with database dependency override."""

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
def golden_payload():
    """Load golden edge payload (K=17, 30 frames)."""
    path = Path(__file__).parent / "fixtures" / "golden_edge_payload.json"
    with open(path) as f:
        return json.load(f)


def test_golden_edge_payload_accepted(client, golden_payload):
    """POST golden edge payload returns 200 and response includes debug fields."""
    response = client.post(
        "/v1/activity/infer",
        json=golden_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["device_id"] == golden_payload["device_id"]
    assert data["camera_id"] == golden_payload["camera_id"]
    assert len(data["results"]) == 1
    assert data["results"][0]["track_id"] == golden_payload["people"][0]["track_id"]

    # Debug/diagnostic fields
    assert "debug" in data
    debug = data["debug"]
    assert debug["frames_received"] == 30
    assert debug["k_count"] == 17
    assert debug["latency_ms"] >= 0
    assert len(debug["per_person"]) == 1
    assert "avg_pose_conf" in debug["per_person"][0]
    assert "frames_ok_ratio" in debug["per_person"][0]
    assert 0 <= debug["per_person"][0]["avg_pose_conf"] <= 1
    assert 0 <= debug["per_person"][0]["frames_ok_ratio"] <= 1


def test_golden_edge_event_has_quality_fields(client, golden_payload, db_session):
    """After infer with golden payload, GET /v1/events returns quality columns."""
    response = client.post(
        "/v1/activity/infer",
        json=golden_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200

    events_response = client.get("/v1/events?limit=5")
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) >= 1
    event = events[0]
    assert event.get("k_count") == 17
    assert "avg_pose_conf" in event
    assert "frames_ok_ratio" in event
