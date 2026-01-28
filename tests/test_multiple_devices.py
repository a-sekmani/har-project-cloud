"""
Tests for multiple devices and device upsert functionality.

This module tests:
- Multiple devices sending inference requests
- Device upsert behavior (no duplicate devices)
- Same device with different cameras
- Same device/camera with different track IDs
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


def test_multiple_devices(client, keypoint_data, db_session):
    """Test inference from multiple devices creates separate device records."""
    # Device 1
    request_data_1 = {
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
            "keypoints": keypoint_data,
            "pose_conf": 0.83
        }]
    }
    
    # Device 2
    request_data_2 = {
        "schema_version": 1,
        "device_id": "pi-002",
        "camera_id": "cam-1",
        "window": {
            "ts_start_ms": 1737970002000,
            "ts_end_ms": 1737970003000,
            "fps": 30,
            "size": 30
        },
        "people": [{
            "track_id": 1,
            "keypoints": keypoint_data,
            "pose_conf": 0.85
        }]
    }
    
    # Send inference from device 1
    response1 = client.post(
        "/v1/activity/infer",
        json=request_data_1,
        headers={"X-API-Key": API_KEY}
    )
    assert response1.status_code == 200
    
    # Send inference from device 2
    response2 = client.post(
        "/v1/activity/infer",
        json=request_data_2,
        headers={"X-API-Key": API_KEY}
    )
    assert response2.status_code == 200
    
    # Verify both devices exist
    devices = db_session.query(Device).all()
    assert len(devices) == 2
    device_ids = [d.device_id for d in devices]
    assert "pi-001" in device_ids
    assert "pi-002" in device_ids
    
    # Verify device-specific events
    events_1 = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).all()
    events_2 = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-002"
    ).all()
    
    assert len(events_1) == 1
    assert len(events_2) == 1
    assert all(e.device_id == "pi-001" for e in events_1)
    assert all(e.device_id == "pi-002" for e in events_2)
    
    # Verify API endpoints return correct data
    response = client.get("/v1/devices")
    assert response.status_code == 200
    devices_data = response.json()
    assert len(devices_data) == 2
    
    # Verify device events endpoints
    response = client.get("/v1/devices/pi-001/events")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) == 1
    assert all(e["device_id"] == "pi-001" for e in events_data)
    
    response = client.get("/v1/devices/pi-002/events")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) == 1
    assert all(e["device_id"] == "pi-002" for e in events_data)


def test_device_upsert_no_duplicates(client, keypoint_data, db_session):
    """Test that same device_id doesn't create duplicate devices (upsert)."""
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
        "people": [{
            "track_id": 7,
            "keypoints": keypoint_data,
            "pose_conf": 0.83
        }]
    }
    
    # First inference
    response1 = client.post(
        "/v1/activity/infer",
        json=request_data,
        headers={"X-API-Key": API_KEY}
    )
    assert response1.status_code == 200
    
    # Get first device creation time
    device1 = db_session.query(Device).filter(
        Device.device_id == "pi-001"
    ).first()
    first_created_at = device1.created_at
    
    # Second inference with same device_id
    request_data["window"]["ts_start_ms"] = 1737970001000
    request_data["window"]["ts_end_ms"] = 1737970002000
    response2 = client.post(
        "/v1/activity/infer",
        json=request_data,
        headers={"X-API-Key": API_KEY}
    )
    assert response2.status_code == 200
    
    # Verify only one device exists
    devices = db_session.query(Device).filter(
        Device.device_id == "pi-001"
    ).all()
    assert len(devices) == 1
    
    # Verify device created_at is from first inference (not updated)
    device = devices[0]
    assert device.created_at == first_created_at


def test_same_device_different_cameras(client, keypoint_data, db_session):
    """Test same device with different cameras creates separate events."""
    # Camera 1
    request_data_1 = {
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
            "keypoints": keypoint_data,
            "pose_conf": 0.83
        }]
    }
    
    # Camera 2
    request_data_2 = {
        "schema_version": 1,
        "device_id": "pi-001",
        "camera_id": "cam-2",
        "window": {
            "ts_start_ms": 1737970002000,
            "ts_end_ms": 1737970003000,
            "fps": 30,
            "size": 30
        },
        "people": [{
            "track_id": 9,
            "keypoints": keypoint_data,
            "pose_conf": 0.75
        }]
    }
    
    # Send from cam-1
    response1 = client.post(
        "/v1/activity/infer",
        json=request_data_1,
        headers={"X-API-Key": API_KEY}
    )
    assert response1.status_code == 200
    
    # Send from cam-2
    response2 = client.post(
        "/v1/activity/infer",
        json=request_data_2,
        headers={"X-API-Key": API_KEY}
    )
    assert response2.status_code == 200
    
    # Verify only one device exists
    devices = db_session.query(Device).filter(
        Device.device_id == "pi-001"
    ).all()
    assert len(devices) == 1
    
    # Verify both events exist
    events = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).all()
    assert len(events) == 2
    
    # Verify camera_ids
    camera_ids = [e.camera_id for e in events]
    assert "cam-1" in camera_ids
    assert "cam-2" in camera_ids
    
    # Verify both appear in device events endpoint
    response = client.get("/v1/devices/pi-001/events")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) == 2
    assert all(e["device_id"] == "pi-001" for e in events_data)
    camera_ids_api = [e["camera_id"] for e in events_data]
    assert "cam-1" in camera_ids_api
    assert "cam-2" in camera_ids_api


def test_same_device_camera_different_track_ids(client, keypoint_data, db_session):
    """Test same device and camera with different track_ids creates separate events."""
    # Track 7
    request_data_1 = {
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
            "keypoints": keypoint_data,
            "pose_conf": 0.83
        }]
    }
    
    # Track 8
    request_data_2 = {
        "schema_version": 1,
        "device_id": "pi-001",
        "camera_id": "cam-1",
        "window": {
            "ts_start_ms": 1737970001000,
            "ts_end_ms": 1737970002000,
            "fps": 30,
            "size": 30
        },
        "people": [{
            "track_id": 8,
            "keypoints": keypoint_data,
            "pose_conf": 0.75
        }]
    }
    
    # Send track 7
    response1 = client.post(
        "/v1/activity/infer",
        json=request_data_1,
        headers={"X-API-Key": API_KEY}
    )
    assert response1.status_code == 200
    
    # Send track 8
    response2 = client.post(
        "/v1/activity/infer",
        json=request_data_2,
        headers={"X-API-Key": API_KEY}
    )
    assert response2.status_code == 200
    
    # Verify both events saved separately
    events = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).all()
    assert len(events) == 2
    
    # Verify track_ids
    track_ids = [e.track_id for e in events]
    assert 7 in track_ids
    assert 8 in track_ids
    
    # Verify both have same device_id and camera_id
    assert all(e.device_id == "pi-001" for e in events)
    assert all(e.camera_id == "cam-1" for e in events)
