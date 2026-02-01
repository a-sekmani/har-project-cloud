"""
Integration tests for end-to-end workflows.

This module contains integration tests that verify complete workflows from
start to finish, including:
- End-to-end inference flow (request → save → retrieve)
- Multiple requests in sequence
- Database consistency across operations

These tests ensure all components work together correctly.
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


def test_end_to_end_flow(client, keypoint_data, db_session):
    """Test complete end-to-end flow: inference → query → verify."""
    # Step 1: Send inference request
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
    
    response = client.post(
        "/v1/activity/infer",
        json=request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Step 2: Verify response is 200 with correct data
    assert response.status_code == 200
    data = response.json()
    assert data["device_id"] == "pi-001"
    assert data["camera_id"] == "cam-1"
    assert len(data["results"]) == 1
    assert data["results"][0]["track_id"] == 7
    assert data["results"][0]["activity"] == "standing"
    assert 0.3 <= data["results"][0]["confidence"] <= 0.9  # Phase 4: confidence from motion_energy

    # Step 3: Query /v1/events → verify event appears
    response = client.get("/v1/events?limit=10")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) > 0
    
    # Find our event
    our_event = next((e for e in events_data if e["device_id"] == "pi-001" and e["track_id"] == 7), None)
    assert our_event is not None
    assert our_event["activity"] == "standing"
    assert 0.3 <= our_event["confidence"] <= 0.9  # Phase 4: confidence from motion_energy

    # Step 4: Query /v1/devices → verify device appears
    response = client.get("/v1/devices")
    assert response.status_code == 200
    devices_data = response.json()
    assert len(devices_data) > 0
    
    # Find our device
    our_device = next((d for d in devices_data if d["device_id"] == "pi-001"), None)
    assert our_device is not None
    
    # Step 5: Query /v1/devices/{device_id}/events → verify event appears
    response = client.get("/v1/devices/pi-001/events?limit=10")
    assert response.status_code == 200
    device_events_data = response.json()
    assert len(device_events_data) > 0
    
    # Verify our event is there
    our_device_event = next((e for e in device_events_data if e["track_id"] == 7), None)
    assert our_device_event is not None
    assert our_device_event["device_id"] == "pi-001"
    assert our_device_event["activity"] == "standing"
    assert 0.3 <= our_device_event["confidence"] <= 0.9  # Phase 4


def test_multiple_requests_sequence(client, keypoint_data, db_session):
    """Test multiple inference requests in sequence."""
    # Send 5 inference requests
    for i in range(5):
        request_data = {
            "schema_version": 1,
            "device_id": "pi-001",
            "camera_id": "cam-1",
            "window": {
                "ts_start_ms": 1737970000000 + (i * 1000),
                "ts_end_ms": 1737970001000 + (i * 1000),
                "fps": 30,
                "size": 30
            },
            "people": [{
                "track_id": 7 + i,
                "keypoints": keypoint_data,
                "pose_conf": 0.83
            }]
        }
        
        response = client.post(
            "/v1/activity/infer",
            json=request_data,
            headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 200
    
    # Verify all 5 events appear in /v1/events?limit=10
    response = client.get("/v1/events?limit=10")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) >= 5
    
    # Verify events are in correct order (newest first)
    if len(events_data) > 1:
        for i in range(len(events_data) - 1):
            assert events_data[i]["created_at"] >= events_data[i + 1]["created_at"]
    
    # Verify all events are for pi-001
    pi001_events = [e for e in events_data if e["device_id"] == "pi-001"]
    assert len(pi001_events) >= 5
    
    # Verify track_ids
    track_ids = [e["track_id"] for e in pi001_events]
    for i in range(5):
        assert (7 + i) in track_ids or (7 + i) in [e["track_id"] for e in events_data]


def test_database_consistency_after_multiple_requests(client, keypoint_data, db_session):
    """Test database consistency after multiple requests."""
    # Send multiple requests with different devices and cameras
    requests = [
        {"device_id": "pi-001", "camera_id": "cam-1", "track_id": 7},
        {"device_id": "pi-001", "camera_id": "cam-2", "track_id": 8},
        {"device_id": "pi-002", "camera_id": "cam-1", "track_id": 1},
    ]
    
    for i, req_config in enumerate(requests):
        request_data = {
            "schema_version": 1,
            "device_id": req_config["device_id"],
            "camera_id": req_config["camera_id"],
            "window": {
                "ts_start_ms": 1737970000000 + (i * 1000),
                "ts_end_ms": 1737970001000 + (i * 1000),
                "fps": 30,
                "size": 30
            },
            "people": [{
                "track_id": req_config["track_id"],
                "keypoints": keypoint_data,
                "pose_conf": 0.83
            }]
        }
        
        response = client.post(
            "/v1/activity/infer",
            json=request_data,
            headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 200
    
    # Verify database state
    devices = db_session.query(Device).all()
    assert len(devices) == 2  # pi-001 and pi-002
    
    events = db_session.query(ActivityEvent).all()
    assert len(events) == 3
    
    # Verify device 1 has 2 events (cam-1 and cam-2)
    events_pi001 = [e for e in events if e.device_id == "pi-001"]
    assert len(events_pi001) == 2
    
    # Verify device 2 has 1 event
    events_pi002 = [e for e in events if e.device_id == "pi-002"]
    assert len(events_pi002) == 1
    
    # Verify API endpoints reflect database state
    response = client.get("/v1/devices")
    assert response.status_code == 200
    devices_data = response.json()
    assert len(devices_data) == 2
    
    response = client.get("/v1/events?limit=10")
    assert response.status_code == 200
    events_data = response.json()
    assert len(events_data) == 3
