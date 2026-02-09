"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class WindowSchema(BaseModel):
    """Window schema for time window information."""
    ts_start_ms: int
    ts_end_ms: int
    fps: int = Field(ge=1, le=120)
    size: int = Field(ge=10, le=120)
    
    @model_validator(mode='after')
    def validate_timestamps(self):
        """Validate that end time is after start time."""
        if self.ts_end_ms <= self.ts_start_ms:
            raise ValueError("ts_end_ms must be greater than ts_start_ms")
        return self


class PersonSchema(BaseModel):
    """Person schema with track_id, keypoints, and pose confidence."""
    track_id: int = Field(ge=0)
    keypoints: List[List[List[float]]]  # [T][K][3]
    pose_conf: float = Field(ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_keypoints_shape(self):
        """Validate keypoints shape: [T][K][3] where K is 17 or 25."""
        if not self.keypoints:
            raise ValueError("keypoints cannot be empty")
        
        # Check that all frames have the same number of keypoints
        num_keypoints = len(self.keypoints[0])
        if num_keypoints not in [17, 25]:
            raise ValueError(f"Number of keypoints per frame must be 17 or 25, got {num_keypoints}")
        
        # Validate all frames have same number of keypoints
        for i, frame in enumerate(self.keypoints):
            if len(frame) != num_keypoints:
                raise ValueError(
                    f"Frame {i} has {len(frame)} keypoints, expected {num_keypoints}"
                )
            
            # Validate each keypoint has 3 values (x, y, confidence)
            for j, kp in enumerate(frame):
                if not isinstance(kp, list) or len(kp) != 3:
                    raise ValueError(
                        f"Keypoint [{i}][{j}] must have 3 values [x, y, confidence], got {len(kp) if isinstance(kp, list) else type(kp)}"
                    )
                # Validate confidence (third value) is between 0 and 1
                if not (0 <= kp[2] <= 1):
                    raise ValueError(
                        f"Keypoint [{i}][{j}] confidence must be between 0 and 1, got {kp[2]}"
                    )
        
        return self


class InferenceRequestSchema(BaseModel):
    """Request schema for activity inference."""
    schema_version: int = Field(...)
    device_id: str
    camera_id: str
    window: WindowSchema
    people: List[PersonSchema] = Field(..., min_length=1)
    
    @model_validator(mode='after')
    def validate_schema_version(self):
        """Validate that schema_version equals 1."""
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        return self
    
    @model_validator(mode='after')
    def validate_keypoints_match_window_size(self):
        """Validate that keypoints length matches window.size."""
        for person_idx, person in enumerate(self.people):
            keypoints_frames = len(person.keypoints)
            if keypoints_frames != self.window.size:
                raise ValueError(
                    f"Person {person_idx} (track_id={person.track_id}): "
                    f"keypoints has {keypoints_frames} frames, "
                    f"but window.size is {self.window.size}"
                )
        return self


class TopKItemSchema(BaseModel):
    """Top-K activity prediction item."""
    label: str
    score: float = Field(ge=0.0, le=1.0)


class PersonResultSchema(BaseModel):
    """Result schema for a single person's activity prediction."""
    track_id: int
    activity: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: List[TopKItemSchema] = Field(..., min_length=3)


class InferenceResponseSchema(BaseModel):
    """Response schema for activity inference."""
    schema_version: int = Field(default=1)
    device_id: str
    camera_id: str
    window: WindowSchema
    results: List[PersonResultSchema]


class SetLabelBody(BaseModel):
    """Body for POST /v1/windows/{id}/label. Empty label clears the label."""
    label: str = Field(default="")
    label_source: str = Field(default="manual")


class PredictWindowBody(BaseModel):
    """Body for POST /v1/windows/{id}/predict."""
    model_key: str = Field(..., min_length=1)
    store: bool = Field(default=True, description="Store prediction in window_predictions")
    return_probs: bool = Field(default=False, description="Include full class probabilities in response")


class IngestWindowBody(BaseModel):
    """Body for POST /v1/windows/ingest — full window from edge (HAR-WindowNet contract)."""
    device_id: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)
    track_id: int = Field(..., ge=0)
    ts_start_ms: int = Field(...)
    ts_end_ms: int = Field(...)
    fps: Union[int, float] = Field(..., ge=1, le=120)
    window_size: int = Field(..., ge=10, le=120)
    keypoints: List[List[List[float]]] = Field(...)

    id: Optional[UUID] = None
    session_id: Optional[str] = None
    mean_pose_conf: Optional[float] = None
    label: Optional[str] = None
    label_source: Optional[str] = None
    created_at: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_ts(self):
        if self.ts_end_ms <= self.ts_start_ms:
            raise ValueError("ts_end_ms must be greater than ts_start_ms")
        return self

    @model_validator(mode='after')
    def validate_keypoints_ingest(self):
        if not self.keypoints:
            raise ValueError("keypoints cannot be empty")
        if len(self.keypoints) != self.window_size:
            raise ValueError(
                f"keypoints must have {self.window_size} frames, got {len(self.keypoints)}"
            )
        for i, frame in enumerate(self.keypoints):
            if len(frame) != 17:
                raise ValueError(
                    f"Frame {i} must have 17 keypoints, got {len(frame)}"
                )
            for j, kp in enumerate(frame):
                if not isinstance(kp, list) or len(kp) != 3:
                    raise ValueError(
                        f"Keypoint [{i}][{j}] must have 3 values [x, y, conf]"
                    )
                for idx, v in enumerate(kp):
                    if not (0 <= v <= 1):
                        raise ValueError(
                            f"Keypoint [{i}][{j}] value [{idx}] must be in [0, 1], got {v}"
                        )
        return self
