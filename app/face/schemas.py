"""
Pydantic schemas for face recognition API.

Defines request/response models for person and face management endpoints.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Person Schemas
# ============================================================================

class PersonCreate(BaseModel):
    """Request body for creating a person."""
    name: str = Field(..., min_length=1, max_length=255, description="Person's name")
    external_ref: Optional[str] = Field(None, max_length=255, description="Optional external reference")
    is_active: bool = Field(True, description="Whether person is active in gallery")


class PersonUpdate(BaseModel):
    """Request body for updating a person."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="New name")
    external_ref: Optional[str] = Field(None, max_length=255, description="Optional external reference")
    is_active: Optional[bool] = Field(None, description="Active status")


class PersonResponse(BaseModel):
    """Response for a single person."""
    id: UUID
    name: str
    external_ref: Optional[str] = None
    is_active: bool
    created_at: datetime
    face_count: int = Field(0, description="Number of face images")
    last_seen: Optional[str] = Field(None, description="ISO datetime of last window")
    total_windows: Optional[int] = Field(None, description="Number of windows linked to person")
    main_activity: Optional[str] = Field(None, description="Most common activity from predictions")

    model_config = {"from_attributes": True}


class PersonListResponse(BaseModel):
    """Response for list of persons."""
    persons: list[PersonResponse]
    total: int


# ============================================================================
# Face Schemas
# ============================================================================

class FaceResponse(BaseModel):
    """Response for a single face record."""
    id: UUID
    person_id: UUID
    image_path: str
    det_score: Optional[float] = None
    quality_score: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FaceUploadResponse(BaseModel):
    """Response for face upload operation."""
    person_id: UUID
    added: int = Field(..., description="Number of faces successfully added")
    skipped: int = Field(0, description="Number of files skipped (duplicate filename)")
    failed: int = Field(0, description="Number of faces that failed processing")
    errors: list[str] = Field(default_factory=list, description="Error messages for failed faces")
    gallery_version: str = Field(..., description="Updated gallery version")


class FaceListResponse(BaseModel):
    """Response for list of faces."""
    person_id: UUID
    faces: list[FaceResponse]
    total: int


# ============================================================================
# Face Gallery Schemas
# ============================================================================

class GalleryPerson(BaseModel):
    """Person entry in face gallery."""
    person_id: UUID
    name: str
    embeddings: list[list[float]] = Field(..., description="List of 512-dim embeddings")


class FaceGalleryResponse(BaseModel):
    """Response for face gallery endpoint (for edge devices)."""
    gallery_version: str = Field(..., description="Version identifier for cache invalidation")
    embedding_dim: int = Field(512, description="Embedding vector dimension")
    threshold: float = Field(0.45, description="Recommended cosine similarity threshold")
    people: list[GalleryPerson] = Field(default_factory=list)


# ============================================================================
# Person Ingest Schema (for windows/ingest)
# ============================================================================

class PersonIngest(BaseModel):
    """Person identification data from edge device."""
    person_id: Optional[UUID] = Field(None, description="Identified person ID (null if unknown)")
    person_name: Optional[str] = Field(None, description="Person name at time of identification")
    person_conf: float = Field(..., ge=0.0, le=1.0, description="Face recognition confidence")
    gallery_version: Optional[str] = Field(None, description="Gallery version used for recognition")

    @field_validator("person_conf")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("person_conf must be between 0 and 1")
        return v


# ============================================================================
# Gallery Version Schema
# ============================================================================

class GalleryVersionResponse(BaseModel):
    """Response for gallery version check."""
    version: str
    created_at: datetime

    model_config = {"from_attributes": True}
