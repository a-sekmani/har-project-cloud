"""
Face recognition module for Cloud HAR.

This module provides face detection, alignment, and embedding generation
using InsightFace. It manages person registration and face gallery for
edge device synchronization.

Components:
- processor: Face detection and embedding generation
- storage: Face image storage utilities
- schemas: Pydantic models for face API
- routes: FastAPI routes for face management
"""

from app.face.processor import FaceProcessor

__all__ = ["FaceProcessor"]
