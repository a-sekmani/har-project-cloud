"""
FastAPI routes for face recognition management.

Provides endpoints for:
- Person CRUD operations
- Face image upload and management
- Face gallery for edge devices
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import API_KEY, MODEL_KEY_DEFAULT
from app.models import Person, PersonFace, GalleryVersion
from app.services import get_person_window_stats, get_person_detail
from app.face.schemas import (
    PersonCreate, PersonUpdate, PersonResponse, PersonListResponse,
    FaceResponse, FaceUploadResponse, FaceListResponse,
    FaceGalleryResponse, GalleryPerson, GalleryVersionResponse
)
from app.face.processor import (
    get_face_processor, FaceProcessor,
    NoFaceDetectedError, MultipleFacesError, FaceProcessorError
)
from app.face.storage import get_face_storage, FaceStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["face"])


# ============================================================================
# Dependencies
# ============================================================================

def get_processor() -> FaceProcessor:
    """Dependency to get face processor."""
    return get_face_processor()


def get_storage() -> FaceStorage:
    """Dependency to get face storage."""
    return get_face_storage()


# ============================================================================
# Gallery Version Management
# ============================================================================

def increment_gallery_version(db: Session) -> str:
    """
    Create new gallery version and return version string.
    Called after any change that affects face gallery.
    """
    # Get current max version number
    result = db.execute(
        select(func.count()).select_from(GalleryVersion)
    )
    count = result.scalar() or 0
    
    # Create new version
    new_version = f"v{count + 1}"
    version_record = GalleryVersion(version=new_version)
    db.add(version_record)
    db.commit()
    
    logger.info(f"Gallery version incremented to {new_version}")
    return new_version


def get_current_gallery_version(db: Session) -> str:
    """Get current gallery version string."""
    result = db.execute(
        select(GalleryVersion)
        .order_by(GalleryVersion.created_at.desc())
        .limit(1)
    )
    version = result.scalar_one_or_none()
    
    if version is None:
        # Initialize first version
        return increment_gallery_version(db)
    
    return version.version


# ============================================================================
# Person Endpoints
# ============================================================================

@router.post("/persons", response_model=PersonResponse)
async def create_person(
    body: PersonCreate,
    db: Session = Depends(get_db)
):
    """Create a new person."""
    person = Person(
        name=body.name,
        external_ref=body.external_ref,
        is_active=body.is_active,
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    
    logger.info(f"Created person: {person.id} ({person.name})")
    
    return PersonResponse(
        id=person.id,
        name=person.name,
        external_ref=person.external_ref,
        is_active=person.is_active,
        created_at=person.created_at,
        face_count=0
    )


@router.get("/persons", response_model=PersonListResponse)
async def list_persons(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_stats: bool = Query(False, description="Include last_seen, total_windows, main_activity"),
    model_key: Optional[str] = Query(None, description="Model for main_activity when include_stats=True"),
    db: Session = Depends(get_db)
):
    """List all persons with optional filtering and optional window stats."""
    query = select(Person)
    
    if is_active is not None:
        query = query.where(Person.is_active == is_active)
    
    query = query.order_by(Person.created_at.desc()).offset(offset).limit(limit)
    
    result = db.execute(query)
    persons = result.scalars().all()
    
    model = model_key or MODEL_KEY_DEFAULT
    person_responses = []
    for person in persons:
        face_count = db.execute(
            select(func.count()).select_from(PersonFace).where(PersonFace.person_id == person.id)
        ).scalar() or 0
        
        extra = {}
        if include_stats:
            stats = get_person_window_stats(db, person.id, model_key=model)
            extra["last_seen"] = stats["last_seen"]
            extra["total_windows"] = stats["total_windows"]
            extra["main_activity"] = stats["main_activity"]
        
        person_responses.append(PersonResponse(
            id=person.id,
            name=person.name,
            external_ref=getattr(person, "external_ref", None),
            is_active=person.is_active,
            created_at=person.created_at,
            face_count=face_count,
            **extra
        ))
    
    # Get total count
    count_query = select(func.count()).select_from(Person)
    if is_active is not None:
        count_query = count_query.where(Person.is_active == is_active)
    total = db.execute(count_query).scalar() or 0
    
    return PersonListResponse(persons=person_responses, total=total)


@router.get("/persons/{person_id}", response_model=PersonResponse)
async def get_person(
    person_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a single person by ID."""
    person = db.get(Person, person_id)
    
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    
    face_count = db.execute(
        select(func.count()).select_from(PersonFace).where(PersonFace.person_id == person.id)
    ).scalar() or 0
    
    return PersonResponse(
        id=person.id,
        name=person.name,
        external_ref=getattr(person, "external_ref", None),
        is_active=person.is_active,
        created_at=person.created_at,
        face_count=face_count
    )


@router.get("/persons/{person_id}/detail")
async def get_person_detail_endpoint(
    person_id: UUID,
    model_key: Optional[str] = Query(None, description="Model for activity stats"),
    since: Optional[str] = Query(None, description="ISO datetime (inclusive) filter for activity data"),
    until: Optional[str] = Query(None, description="ISO datetime (inclusive) filter for activity data"),
    db: Session = Depends(get_db)
):
    """Get full person detail: stats, activity_distribution, activity_timeline, recent_windows.
    Optional since/until filter activity data by window created_at."""
    from datetime import datetime, timezone
    since_dt = None
    until_dt = None
    if since and since.strip():
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    if until and until.strip():
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    detail = get_person_detail(
        db, person_id,
        model_key=model_key or MODEL_KEY_DEFAULT,
        recent_windows_limit=50,
        since=since_dt,
        until=until_dt,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return detail


@router.patch("/persons/{person_id}", response_model=PersonResponse)
async def update_person(
    person_id: UUID,
    body: PersonUpdate,
    db: Session = Depends(get_db)
):
    """Update a person's details."""
    person = db.get(Person, person_id)
    
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Track if we need to update gallery version
    gallery_changed = False
    
    if body.name is not None:
        person.name = body.name
    
    if body.external_ref is not None:
        person.external_ref = body.external_ref
    
    if body.is_active is not None and body.is_active != person.is_active:
        person.is_active = body.is_active
        gallery_changed = True
    
    db.commit()
    db.refresh(person)
    
    # Update gallery version if active status changed
    if gallery_changed:
        increment_gallery_version(db)
    
    logger.info(f"Updated person: {person.id}")
    
    face_count = db.execute(
        select(func.count()).select_from(PersonFace).where(PersonFace.person_id == person.id)
    ).scalar() or 0
    
    return PersonResponse(
        id=person.id,
        name=person.name,
        external_ref=person.external_ref,
        is_active=person.is_active,
        created_at=person.created_at,
        face_count=face_count
    )


@router.delete("/persons/{person_id}")
async def delete_person(
    person_id: UUID,
    db: Session = Depends(get_db),
    storage: FaceStorage = Depends(get_storage)
):
    """Delete a person and all their face images."""
    person = db.get(Person, person_id)
    
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Delete face images from storage
    deleted_count = storage.delete_person_faces(person_id)
    
    # Delete from database (cascades to person_faces)
    db.delete(person)
    db.commit()
    
    # Update gallery version
    increment_gallery_version(db)
    
    logger.info(f"Deleted person: {person_id} ({deleted_count} face images)")
    
    return {"deleted": True, "face_images_deleted": deleted_count}


# ============================================================================
# Face Endpoints
# ============================================================================

@router.post("/persons/{person_id}/faces", response_model=FaceUploadResponse)
async def upload_faces(
    person_id: UUID,
    files: list[UploadFile] = File(..., description="Face image files"),
    db: Session = Depends(get_db),
    processor: FaceProcessor = Depends(get_processor),
    storage: FaceStorage = Depends(get_storage)
):
    """
    Upload face images for a person.
    
    Each image is processed to detect face and extract embedding.
    Images without a clear face are rejected.
    """
    person = db.get(Person, person_id)
    
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Existing filenames for this person (normalized) to skip duplicates
    existing = db.execute(
        select(PersonFace.original_filename).where(PersonFace.person_id == person_id)
    )
    existing_filenames = { (f or "").strip().lower() for f in existing.scalars() if f }
    
    added = 0
    skipped = 0
    failed = 0
    errors = []
    
    for file in files:
        try:
            name = (file.filename or "").strip()
            if not name:
                errors.append("Unnamed file: Empty filename")
                failed += 1
                continue
            name_lower = name.lower()
            if name_lower in existing_filenames:
                skipped += 1
                continue
            
            # Read image data
            image_data = await file.read()
            
            if len(image_data) == 0:
                errors.append(f"{file.filename}: Empty file")
                failed += 1
                continue
            
            # Process face
            result = processor.process_image(image_data)
            
            # Create face record
            face = PersonFace(
                person_id=person_id,
                image_path="",  # Will be updated after save
                original_filename=name,
                embedding=result.embedding.tolist(),
                det_score=result.det_score
            )
            db.add(face)
            db.flush()  # Get the ID
            
            # Save image to storage
            image_path = storage.save_face(
                person_id=person_id,
                face_id=face.id,
                image_data=image_data,
                original_filename=file.filename
            )
            face.image_path = image_path
            existing_filenames.add(name_lower)
            
            db.commit()
            added += 1
            
            logger.info(f"Added face for person {person_id}: {face.id} (det_score={result.det_score:.3f})")
            
        except NoFaceDetectedError:
            errors.append(f"{file.filename}: No face detected")
            failed += 1
            db.rollback()
            
        except MultipleFacesError as e:
            errors.append(f"{file.filename}: Multiple faces detected ({e.count})")
            failed += 1
            db.rollback()
            
        except FaceProcessorError as e:
            errors.append(f"{file.filename}: Processing error - {str(e)}")
            failed += 1
            db.rollback()
            
        except Exception as e:
            errors.append(f"{file.filename}: Unexpected error - {str(e)}")
            failed += 1
            db.rollback()
            logger.exception(f"Unexpected error processing face upload")
    
    # Update gallery version if any faces were added
    gallery_version = get_current_gallery_version(db)
    if added > 0:
        gallery_version = increment_gallery_version(db)
    
    return FaceUploadResponse(
        person_id=person_id,
        added=added,
        skipped=skipped,
        failed=failed,
        errors=errors,
        gallery_version=gallery_version
    )


@router.get("/persons/{person_id}/faces", response_model=FaceListResponse)
async def list_person_faces(
    person_id: UUID,
    db: Session = Depends(get_db)
):
    """List all face images for a person."""
    person = db.get(Person, person_id)
    
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    
    result = db.execute(
        select(PersonFace)
        .where(PersonFace.person_id == person_id)
        .order_by(PersonFace.created_at.desc())
    )
    faces = result.scalars().all()
    
    return FaceListResponse(
        person_id=person_id,
        faces=[FaceResponse.model_validate(f) for f in faces],
        total=len(faces)
    )


@router.delete("/persons/{person_id}/faces/{face_id}")
async def delete_face(
    person_id: UUID,
    face_id: UUID,
    db: Session = Depends(get_db),
    storage: FaceStorage = Depends(get_storage)
):
    """Delete a specific face image."""
    face = db.execute(
        select(PersonFace)
        .where(PersonFace.id == face_id)
        .where(PersonFace.person_id == person_id)
    ).scalar_one_or_none()
    
    if face is None:
        raise HTTPException(status_code=404, detail="Face not found")
    
    # Delete from storage
    storage.delete_face(face.image_path)
    
    # Delete from database
    db.delete(face)
    db.commit()
    
    # Update gallery version
    increment_gallery_version(db)
    
    logger.info(f"Deleted face: {face_id} (person: {person_id})")
    
    return {"deleted": True}


# ============================================================================
# Face Gallery Endpoint
# ============================================================================

@router.get("/face-gallery", response_model=FaceGalleryResponse)
async def get_face_gallery(
    db: Session = Depends(get_db)
):
    """
    Get face gallery for edge devices.
    
    Returns all active persons with their face embeddings.
    Edge devices should cache this and refresh when gallery_version changes.
    """
    gallery_version = get_current_gallery_version(db)
    
    # Get all active persons with their faces
    result = db.execute(
        select(Person)
        .where(Person.is_active == True)
        .order_by(Person.name)
    )
    persons = result.scalars().all()
    
    people = []
    for person in persons:
        # Get all embeddings for this person
        faces_result = db.execute(
            select(PersonFace.embedding)
            .where(PersonFace.person_id == person.id)
            .order_by(PersonFace.det_score.desc())
        )
        embeddings = [row[0] for row in faces_result.fetchall()]
        
        if embeddings:  # Only include persons with at least one face
            people.append(GalleryPerson(
                person_id=person.id,
                name=person.name,
                embeddings=embeddings
            ))
    
    return FaceGalleryResponse(
        gallery_version=gallery_version,
        embedding_dim=FaceProcessor.EMBEDDING_DIM,
        threshold=FaceProcessor.DEFAULT_THRESHOLD,
        people=people
    )


@router.get("/face-gallery/version", response_model=GalleryVersionResponse)
async def get_gallery_version(
    db: Session = Depends(get_db)
):
    """
    Get current gallery version.
    
    Edge devices can poll this to check if they need to refresh their gallery.
    """
    version_str = get_current_gallery_version(db)
    
    result = db.execute(
        select(GalleryVersion)
        .where(GalleryVersion.version == version_str)
    )
    version = result.scalar_one()
    
    return GalleryVersionResponse(
        version=version.version,
        created_at=version.created_at
    )
