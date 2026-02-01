"""
Pydantic schemas for edge frame_event (input contract for POST /v1/edge/events).

EdgeFrameEvent v1: event_type "frame_event", source.device_id, source.session_id,
frame.ts_unix_ms (int or float; converted to int in cloud), persons (required, can be []),
persons[].keypoints as 17 objects {name, x, y, c} with names = COCO-17 set.
Extra fields allowed (bbox, coords, score, etc.) and ignored.
"""
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Canonical COCO-17 keypoint names. Order used for normalization output; request order is free.
COCO_17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_17_SET = frozenset(COCO_17_NAMES)


class EdgeKeypointSchema(BaseModel):
    """Single keypoint from edge: name, x, y, c (confidence)."""
    name: str
    x: float
    y: float
    c: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")


class EdgeFrameEventSourceSchema(BaseModel):
    """Source of the frame event (device, session; camera optional)."""
    device_id: str
    session_id: str
    camera_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class EdgeFrameEventFrameSchema(BaseModel):
    """Frame metadata (timestamp). Accept int or float; cloud converts to int."""
    ts_unix_ms: Union[int, float]

    model_config = ConfigDict(extra="allow")


class EdgeFrameEventPersonSchema(BaseModel):
    """One person in a frame: track_id, keypoints as 17 objects {name, x, y, c} (COCO-17 names)."""
    track_id: int = Field(ge=0)
    keypoints: List[EdgeKeypointSchema]  # exactly 17, names = COCO_17_SET

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def check_keypoints_17_coco(self):
        if len(self.keypoints) != 17:
            raise ValueError("keypoints must have exactly 17 elements")
        names = {k.name for k in self.keypoints}
        if names != COCO_17_SET:
            raise ValueError(
                f"keypoints names must be exactly the COCO-17 set; got {sorted(names)}"
            )
        return self


class EdgeFrameEventSchema(BaseModel):
    """
    Edge frame_event as sent by the edge app (v1).
    event_type must be "frame_event". Required: source.device_id, source.session_id,
    frame.ts_unix_ms, persons (field required; can be empty list []).
    """
    event_type: str
    source: EdgeFrameEventSourceSchema
    frame: EdgeFrameEventFrameSchema
    persons: List[EdgeFrameEventPersonSchema]  # required field; can be []

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def check_event_type(self):
        if self.event_type != "frame_event":
            raise ValueError('event_type must be "frame_event"')
        return self


# Aliases for backward compatibility
FrameEventSchema = EdgeFrameEventSchema
FrameEventSourceSchema = EdgeFrameEventSourceSchema
FrameEventFrameSchema = EdgeFrameEventFrameSchema
FrameEventPersonSchema = EdgeFrameEventPersonSchema
