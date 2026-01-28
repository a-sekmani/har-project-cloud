"""Tests for inference endpoint schema validation."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import API_KEY

client = TestClient(app)


@pytest.fixture
def valid_request_data():
    """Fixture with valid request data."""
    # Create 17 keypoints per frame (standard pose estimation format)
    keypoint_template = [[0.52, 0.18, 0.91] for _ in range(17)]
    
    # Create 30 frames to match window.size
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


def test_infer_with_valid_request(valid_request_data):
    """Test inference with valid request returns 200."""
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert data["schema_version"] == 1
    assert data["device_id"] == "pi-001"
    assert data["camera_id"] == "cam-1"
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["track_id"] == 7
    assert data["results"][0]["activity"] == "standing"
    assert data["results"][0]["confidence"] == 0.6
    assert len(data["results"][0]["top_k"]) >= 3


def test_infer_without_api_key(valid_request_data):
    """Test that request without API key returns 401."""
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data
    )
    
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_infer_with_invalid_api_key(valid_request_data):
    """Test that request with invalid API key returns 401."""
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": "wrong-key"}
    )
    
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_infer_window_size_mismatch(valid_request_data):
    """Test that window.size mismatch with keypoints length returns 422."""
    # Modify keypoints to have 29 frames instead of 30
    valid_request_data["people"][0]["keypoints"] = valid_request_data["people"][0]["keypoints"][:29]
    
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 422
    # Pydantic validation error should mention the mismatch
    detail = response.json()["detail"]
    assert any("window.size" in str(err) or "frames" in str(err).lower() for err in detail)


def test_infer_with_low_pose_conf(valid_request_data):
    """Test that low pose_conf returns 'unknown' activity."""
    valid_request_data["people"][0]["pose_conf"] = 0.3
    
    response = client.post(
        "/v1/activity/infer",
        json=valid_request_data,
        headers={"X-API-Key": API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["activity"] == "unknown"
    assert data["results"][0]["confidence"] == 0.2
