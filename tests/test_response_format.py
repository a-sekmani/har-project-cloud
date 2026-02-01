"""
Tests for API response format validation.

This module verifies that API responses conform to the expected format:
- Correct JSON structure with all required fields
- ISO-formatted timestamps
- UUIDs as strings (not objects)
- Proper data types for all fields
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.config import API_KEY
from app.database import Base, get_db
import re
from datetime import datetime

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
def valid_request_data():
    """Fixture with valid request data."""
    keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
    keypoints = [keypoint_template.copy() for _ in range(30)]
    
    return {
        "schema_version": 1,
        "device_id": "pi-001",
        "camera_id": "cam-1",
        "window": {
            "ts_start_ms": 1737970000000,
            "ts_end_ms": 1737970001000,
            "fps": 30,
            "size": 30
        },
        "people": [{
            "track_id": 7,
            "keypoints": keypoints,
            "pose_conf": 0.83
        }]
    }


def test_events_response_structure(client, valid_request_data, db_session):
    """Test GET /v1/events returns correct JSON structure."""
    # Create an event
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    event = data[0]
    required_fields = [
        "id", "device_id", "camera_id", "track_id",
        "ts_start_ms", "ts_end_ms", "fps", "window_size",
        "activity", "confidence",
        "k_count", "avg_pose_conf", "frames_ok_ratio",
        "created_at"
    ]
    
    for field in required_fields:
        assert field in event, f"Missing field: {field}"


def test_devices_response_structure(client, valid_request_data, db_session):
    """Test GET /v1/devices returns correct JSON structure."""
    # Create a device
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    device = data[0]
    required_fields = ["id", "device_id", "created_at"]
    
    for field in required_fields:
        assert field in device, f"Missing field: {field}"


def test_device_events_response_structure(client, valid_request_data, db_session):
    """Test GET /v1/devices/{device_id}/events returns correct JSON structure."""
    # Create an event
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/devices/pi-001/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    event = data[0]
    required_fields = [
        "id", "device_id", "camera_id", "track_id",
        "ts_start_ms", "ts_end_ms", "fps", "window_size",
        "activity", "confidence",
        "k_count", "avg_pose_conf", "frames_ok_ratio",
        "created_at"
    ]
    
    for field in required_fields:
        assert field in event, f"Missing field: {field}"


def test_timestamps_iso_format(client, valid_request_data, db_session):
    """Test that all timestamps are in ISO format."""
    # Create an event
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Check events endpoint
    response = client.get("/v1/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    event = data[0]
    
    # Verify ISO format (YYYY-MM-DDTHH:MM:SS or with microseconds)
    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    assert re.match(iso_pattern, event["created_at"]), f"Invalid ISO format: {event['created_at']}"
    
    # Try to parse it
    try:
        datetime.fromisoformat(event["created_at"].replace('Z', '+00:00'))
    except ValueError:
        pytest.fail(f"Timestamp is not valid ISO format: {event['created_at']}")
    
    # Check devices endpoint
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    device = data[0]
    
    assert re.match(iso_pattern, device["created_at"]), f"Invalid ISO format: {device['created_at']}"


def test_uuids_are_strings(client, valid_request_data, db_session):
    """Test that UUIDs are returned as strings, not objects."""
    # Create an event
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Check events endpoint
    response = client.get("/v1/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    event = data[0]
    
    # Verify id is a string
    assert isinstance(event["id"], str), f"ID should be string, got {type(event['id'])}"
    
    # Verify it looks like a UUID (basic check)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    assert re.match(uuid_pattern, event["id"]), f"ID should be UUID format: {event['id']}"
    
    # Check devices endpoint
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    device = data[0]
    
    assert isinstance(device["id"], str), f"ID should be string, got {type(device['id'])}"
    assert re.match(uuid_pattern, device["id"]), f"ID should be UUID format: {device['id']}"
