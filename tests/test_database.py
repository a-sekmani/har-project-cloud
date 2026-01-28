"""
Tests for database operations and new endpoints.

This module tests:
- Database save operations (inference results)
- Data retrieval endpoints (/v1/events, /v1/devices, /v1/devices/{id}/events)
- Database model relationships
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
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop tables after test
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
        "people": [
            {
                "track_id": 7,
                "keypoints": keypoints,
                "pose_conf": 0.83
            }
        ]
    }


def test_infer_saves_to_database(client, valid_request_data, db_session):
    """Test that inference request saves results to database."""
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 200
    
    # Check device was created
    device = db_session.query(Device).filter(Device.device_id == "pi-001").first()
    assert device is not None
    assert device.device_id == "pi-001"
    
    # Check event was created
    event = db_session.query(ActivityEvent).filter(
        ActivityEvent.device_id == "pi-001"
    ).first()
    assert event is not None
    assert event.device_id == "pi-001"
    assert event.camera_id == "cam-1"
    assert event.track_id == 7
    assert event.activity == "standing"
    assert event.confidence == 0.6


def test_get_events_endpoint(client, valid_request_data, db_session):
    """Test GET /v1/events returns saved events."""
    # Create an event first
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Get events
    response = client.get("/v1/events?limit=10")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["device_id"] == "pi-001"
    assert data[0]["activity"] == "standing"


def test_get_devices_endpoint(client, valid_request_data, db_session):
    """Test GET /v1/devices returns device after first inference."""
    # Create an event first
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Get devices
    response = client.get("/v1/devices")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert any(d["device_id"] == "pi-001" for d in data)


def test_get_device_events_endpoint(client, valid_request_data, db_session):
    """Test GET /v1/devices/{device_id}/events returns device events."""
    # Create an event first
    client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    # Get device events
    response = client.get("/v1/devices/pi-001/events?limit=10")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert all(e["device_id"] == "pi-001" for e in data)
