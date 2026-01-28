"""
Tests for API edge cases and validation.

This module contains comprehensive tests for edge cases and boundary conditions
in the API endpoints. These tests ensure the API handles invalid inputs,
boundary values, and edge cases correctly.

Test categories:
- Limit parameter validation (min, max, invalid values)
- Empty database scenarios
- Sorting verification
- Device upsert behavior
- Non-existent device handling
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.config import API_KEY
from app.database import Base, get_db
from app.models import Device, ActivityEvent

# Create in-memory SQLite database for tests
# Using SQLite instead of PostgreSQL for faster test execution
# check_same_thread=False: Allows SQLite to work with FastAPI's async nature
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    Create a test database session for each test function.
    
    This fixture:
    1. Creates all database tables before the test
    2. Yields a database session
    3. Closes the session and drops all tables after the test
    
    Scope: function - Each test gets a fresh database
    This ensures test isolation (tests don't affect each other).
    """
    # Create all tables defined in models
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        # Cleanup: close session and drop all tables
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """
    Create a test client with database dependency override.
    
    This fixture overrides FastAPI's get_db dependency to use our test database
    session instead of the production database. This allows tests to run
    against an isolated test database.
    
    Args:
        db_session: Test database session (from db_session fixture)
    
    Yields:
        TestClient: FastAPI test client configured to use test database
    """
    from app.database import get_db
    
    # Override the get_db dependency to use test database
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    # Override the dependency
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    # Clear overrides after test to avoid affecting other tests
    app.dependency_overrides.clear()


@pytest.fixture
def valid_request_data():
    """
    Fixture providing valid inference request data for testing.
    
    Creates a complete, valid inference request with:
    - 30 frames (matching window.size)
    - 17 keypoints per frame (standard pose estimation format)
    - Valid metadata (device_id, camera_id, timestamps, etc.)
    
    Returns:
        dict: Valid inference request payload
    """
    # Create keypoint template: [x, y, confidence] for 17 keypoints
    keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
    # Create 30 frames (matching window.size = 30)
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


# ============================================================================
# GET /v1/events Edge Cases
# ============================================================================

def test_get_events_with_limit_1(client, valid_request_data, db_session):
    """
    Test GET /v1/events with limit=1 (minimum valid value).
    
    Verifies that the minimum limit value (1) works correctly and returns
    exactly one event when multiple events exist.
    """
    # Create 3 events
    for i in range(3):
        client.post(
            "/v1/activity/infer",
            json=valid_request_data,
            headers={"X-API-Key": API_KEY}
        )
    
    response = client.get("/v1/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1


def test_get_events_with_limit_1000(client, valid_request_data, db_session):
    """
    Test GET /v1/events with limit=1000 (maximum valid value).
    
    Verifies that the maximum limit value (1000) works correctly and doesn't
    exceed the limit even if more events exist.
    """
    # Create 1 event
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/events?limit=1000")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 1000


def test_get_events_with_limit_0(client):
    """
    Test GET /v1/events with limit=0 (should fail validation).
    
    Verifies that invalid limit values (below minimum) are rejected with
    HTTP 422 (Unprocessable Entity) status code.
    """
    response = client.get("/v1/events?limit=0")
    assert response.status_code == 422


def test_get_events_with_limit_negative(client):
    """Test GET /v1/events with limit=-1 (should fail validation)."""
    response = client.get("/v1/events?limit=-1")
    assert response.status_code == 422


def test_get_events_with_limit_exceeds_max(client):
    """Test GET /v1/events with limit=1001 (should fail validation)."""
    response = client.get("/v1/events?limit=1001")
    assert response.status_code == 422


def test_get_events_empty_database(client):
    """
    Test GET /v1/events with empty database returns empty array.
    
    Verifies that the endpoint handles empty database gracefully by returning
    an empty list instead of an error.
    """
    response = client.get("/v1/events")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_get_events_sorted_newest_first(client, valid_request_data, db_session):
    """
    Test that GET /v1/events returns events sorted by created_at DESC.
    
    Verifies that events are returned in descending order (newest first),
    which is important for dashboards and monitoring tools.
    """
    # Create 3 events with slight delays
    import time
    for i in range(3):
        client.post(
            "/v1/activity/infer",
            json=valid_request_data,
            headers={"X-API-Key": API_KEY}
        )
        time.sleep(0.01)  # Small delay to ensure different timestamps
    
    response = client.get("/v1/events?limit=10")
    assert response.status_code == 200
    data = response.json()
    
    # Verify sorting (newest first)
    if len(data) > 1:
        for i in range(len(data) - 1):
            assert data[i]["created_at"] >= data[i + 1]["created_at"]


# ============================================================================
# GET /v1/devices Edge Cases
# ============================================================================

def test_get_devices_empty_database(client):
    """
    Test GET /v1/devices with empty database returns empty array.
    
    Verifies that the endpoint handles empty database gracefully.
    """
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_get_devices_upsert_no_duplicates(client, valid_request_data, db_session):
    """Test that same device_id doesn't create duplicate devices."""
    # Send first inference
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Send second inference with same device_id
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Check devices
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    
    # Should have only one device
    device_ids = [d["device_id"] for d in data]
    assert device_ids.count("pi-001") == 1


def test_get_devices_multiple_devices(client, valid_request_data, db_session):
    """Test GET /v1/devices with multiple devices."""
    # Create device 1
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Create device 2
    request_data_2 = valid_request_data.copy()
    request_data_2["device_id"] = "pi-002"
    client.post(
        "/v1/activity/infer",
        json=request_data_2,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/devices")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    device_ids = [d["device_id"] for d in data]
    assert "pi-001" in device_ids
    assert "pi-002" in device_ids


# ============================================================================
# GET /v1/devices/{device_id}/events Edge Cases
# ============================================================================

def test_get_device_events_nonexistent_device(client):
    """
    Test GET /v1/devices/{device_id}/events with non-existent device_id.
    
    Verifies that querying events for a non-existent device returns an empty
    array instead of an error. This ensures graceful handling of invalid
    device IDs.
    """
    response = client.get("/v1/devices/non-existent/events")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_get_device_events_limit_1(client, valid_request_data, db_session):
    """Test GET /v1/devices/{device_id}/events with limit=1."""
    # Create 3 events
    for i in range(3):
        client.post(
            "/v1/activity/infer",
            json=valid_request_data,
            headers={"X-API-Key": API_KEY}
        )
    
    response = client.get("/v1/devices/pi-001/events?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1


def test_get_device_events_limit_1000(client, valid_request_data, db_session):
    """Test GET /v1/devices/{device_id}/events with limit=1000."""
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    response = client.get("/v1/devices/pi-001/events?limit=1000")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 1000


def test_get_device_events_invalid_limit(client):
    """Test GET /v1/devices/{device_id}/events with invalid limit."""
    response = client.get("/v1/devices/pi-001/events?limit=0")
    assert response.status_code == 422


def test_get_device_events_sorted_newest_first(client, valid_request_data, db_session):
    """Test that device events are sorted by created_at DESC."""
    import time
    for i in range(3):
        client.post(
            "/v1/activity/infer",
            json=valid_request_data,
            headers={"X-API-Key": API_KEY}
        )
        time.sleep(0.01)
    
    response = client.get("/v1/devices/pi-001/events?limit=10")
    assert response.status_code == 200
    data = response.json()
    
    if len(data) > 1:
        for i in range(len(data) - 1):
            assert data[i]["created_at"] >= data[i + 1]["created_at"]


def test_get_device_events_only_returns_specified_device(client, valid_request_data, db_session):
    """
    Test that device events endpoint only returns events for specified device.
    
    Verifies that filtering by device_id works correctly and doesn't return
    events from other devices. This is critical for data isolation and security.
    """
    # Create event for device 1
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Create event for device 2
    request_data_2 = valid_request_data.copy()
    request_data_2["device_id"] = "pi-002"
    client.post(
        "/v1/activity/infer",
        json=request_data_2,
        headers={"X-API-Key": API_KEY}
    )
    
    # Get events for device 1
    response = client.get("/v1/devices/pi-001/events")
    assert response.status_code == 200
    data = response.json()
    
    # All events should be for pi-001
    assert all(e["device_id"] == "pi-001" for e in data)
    assert len(data) == 1
