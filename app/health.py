"""
Health check endpoint.

This module provides a simple health check endpoint that can be used by
monitoring systems, load balancers, and orchestration tools to verify
that the API is running and responding to requests.

The endpoint returns a simple JSON response indicating the service status.
"""
from fastapi import APIRouter
from typing import Dict

# Create router for health check endpoint
# This router is included in the main app in app/main.py
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns a simple status response indicating the API is operational.
    This endpoint does not perform any database checks or complex validation
    - it's designed to be fast and always available.
    
    Returns:
        Dict[str, str]: JSON response with status field set to "ok"
    
    Example:
        GET /health
        Response: {"status": "ok"}
    
    Use cases:
        - Kubernetes liveness/readiness probes
        - Load balancer health checks
        - Monitoring system status checks
    """
    return {"status": "ok"}
