"""
Tests for health check endpoint.

This module contains tests for the /health endpoint, which is used to verify
that the API is running and responding to requests. Health checks are essential
for monitoring and load balancer integration.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a test client for making HTTP requests to the FastAPI app
# TestClient is FastAPI's built-in testing tool that simulates HTTP requests
client = TestClient(app)


def test_health_check():
    """
    Test that /health endpoint returns 200 status and correct response.
    
    This test verifies:
    1. The endpoint is accessible
    2. Returns HTTP 200 status code
    3. Response body contains {"status": "ok"}
    
    This is a basic smoke test to ensure the API is running correctly.
    """
    response = client.get("/health")
    
    # Verify HTTP status code is 200 (OK)
    assert response.status_code == 200
    
    # Verify response body matches expected format
    data = response.json()
    assert data == {"status": "ok"}
