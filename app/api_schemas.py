"""Pydantic schemas for API responses."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class EventResponseSchema(BaseModel):
    """Activity event response schema."""
    id: str
    device_id: str
    camera_id: str
    track_id: int
    ts_start_ms: int
    ts_end_ms: int
    fps: int
    window_size: int
    activity: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


class DeviceResponseSchema(BaseModel):
    """Device response schema."""
    id: str
    device_id: str
    created_at: datetime

    class Config:
        from_attributes = True
