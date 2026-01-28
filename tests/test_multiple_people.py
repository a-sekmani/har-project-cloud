"""
Tests for multiple people in inference requests.

This module tests scenarios where a single inference request contains multiple
people (multiple track_ids). Each person should be processed separately and
saved as a separate event in the database.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.config import API_KEY
from app.database import Base, get_db
from app.models import ActivityEvent

# Create in-memory SQLite database for tests
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
    from app.database import get_db
    
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
def keypoint_data():
    """Fixture for keypoint data."""
    keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
    keypoints = [keypoint_template.copy() for _ in range(30)]
    return keypoints


def test_multiple_people_same_request(client, keypoint_data, db_session):
    """Test inference with 2 people in same request creates 2 separate events."""
    request_data = {
        "schema_version": 1,
        "device_id": "pi-001",
        "camera_id": "cam-1",
        "window": {
            "ts_start_ms": 1737970000000,
            "ts_end_ms": 1737970001000,
            "fps": 30,
            "size": 30
        },
        "people": [
            {
                "track_id": 7,
                "keypoints": keypoint_data,
                "pose_conf": 0.83
            },
            {
                "track_id": 8,
                "keypoints": keypoint_data,
                "pose_conf": 0.75
            }
        ]
    }
    
    response = client.post(
        "/v1/activity/infer",
        json=request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response has 2 results
    assert len(data["results"]) == 2
    assert data["results"][0]["track_id"] == 7
    assert data["results"][1]["track_id"] == 8
    
    # Verify both events saved to database
    events = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).all()
    assert len(events) == 2
    
    # Verify both events have same metadata
    assert all(e.device_id == "pi-001" for e in events)
    assert all(e.camera_id == "cam-1" for e in events)
    assert all(e.ts_start_ms == 1737970000000 for e in events)
    assert all(e.ts_end_ms == 1737970001000 for e in events)
    assert all(e.fps == 30 for e in events)
    assert all(e.window_size == 30 for e in events)
    
    # Verify different track_ids
    track_ids = [e.track_id for e in events]
    assert 7 in track_ids
    assert 8 in track_ids


def test_multiple_people_different_pose_conf(client, keypoint_data, db_session):
    """Test multiple people with different pose_conf values get correct activities."""
    request_data = {
        "schema_version": 1,
        "device_id": "pi-001",
        "camera_id": "cam-1",
        "window": {
            "ts_start_ms": 1737970000000,
            "ts_end_ms": 1737970001000,
            "fps": 30,
            "size": 30
        },
        "people": [
            {
                "track_id": 7,
                "keypoints": keypoint_data,
                "pose_conf": 0.3  # Should be "unknown" (0.2)
            },
            {
                "track_id": 8,
                "keypoints": keypoint_data,
                "pose_conf": 0.8  # Should be "standing" (0.6)
            }
        ]
    }
    
    response = client.post(
        "/v1/activity/infer",
        json=request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify results
    results = {r["track_id"]: r for r in data["results"]}
    
    # Person 1: pose_conf = 0.3 → unknown, confidence = 0.2
    assert results[7]["activity"] == "unknown"
    assert results[7]["confidence"] == 0.2
    
    # Person 2: pose_conf = 0.8 → standing, confidence = 0.6
    assert results[8]["activity"] == "standing"
    assert results[8]["confidence"] == 0.6
    
    # Verify both events saved correctly
    events = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).all()
    assert len(events) == 2
    
    # Verify activities in database
    event_by_track = {e.track_id: e for e in events}
    assert event_by_track[7].activity == "unknown"
    assert event_by_track[7].confidence == 0.2
    assert event_by_track[8].activity == "standing"
    assert event_by_track[8].confidence == 0.6
