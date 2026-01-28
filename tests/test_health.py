"""Tests for health check endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test that /health returns 200 and status=ok."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
