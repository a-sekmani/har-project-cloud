"""
Face image storage utilities.

Handles saving and loading face images to/from local filesystem.
Images are stored in ./data/person_faces/{person_id}/{face_id}.jpg
"""

import os
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Default storage directory
DEFAULT_STORAGE_DIR = Path("./data/person_faces")


class FaceStorageError(Exception):
    """Base exception for face storage errors."""
    pass


class FaceStorage:
    """
    Face image storage manager.
    
    Stores face images in a hierarchical directory structure:
    {storage_dir}/{person_id}/{face_id}.jpg
    """
    
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    DEFAULT_FORMAT = ".jpg"
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize face storage.
        
        Args:
            storage_dir: Base directory for storing face images.
                        Defaults to ./data/person_faces
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Face storage directory: {self.storage_dir.absolute()}")
    
    def _get_person_dir(self, person_id: UUID) -> Path:
        """Get directory for a person's face images."""
        return self.storage_dir / str(person_id)
    
    def _get_face_path(self, person_id: UUID, face_id: UUID, ext: str = ".jpg") -> Path:
        """Get full path for a face image."""
        return self._get_person_dir(person_id) / f"{face_id}{ext}"
    
    def save_face(
        self,
        person_id: UUID,
        face_id: UUID,
        image_data: bytes,
        original_filename: Optional[str] = None
    ) -> str:
        """
        Save face image to storage.
        
        Args:
            person_id: Person's UUID
            face_id: Face record's UUID
            image_data: Raw image bytes
            original_filename: Original filename (used to determine extension)
            
        Returns:
            Relative path to saved image (relative to storage_dir)
            
        Raises:
            FaceStorageError: If save fails
        """
        # Determine extension
        ext = self.DEFAULT_FORMAT
        if original_filename:
            orig_ext = Path(original_filename).suffix.lower()
            if orig_ext in self.ALLOWED_EXTENSIONS:
                ext = orig_ext
        
        # Create person directory
        person_dir = self._get_person_dir(person_id)
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        face_path = self._get_face_path(person_id, face_id, ext)
        try:
            with open(face_path, "wb") as f:
                f.write(image_data)
            
            # Return relative path
            relative_path = str(face_path.relative_to(self.storage_dir))
            logger.debug(f"Saved face image: {relative_path}")
            return relative_path
            
        except IOError as e:
            raise FaceStorageError(f"Failed to save face image: {e}") from e
    
    def load_face(self, relative_path: str) -> bytes:
        """
        Load face image from storage.
        
        Args:
            relative_path: Relative path (as returned by save_face)
            
        Returns:
            Raw image bytes
            
        Raises:
            FaceStorageError: If load fails or file not found
        """
        full_path = self.storage_dir / relative_path
        
        if not full_path.exists():
            raise FaceStorageError(f"Face image not found: {relative_path}")
        
        try:
            with open(full_path, "rb") as f:
                return f.read()
        except IOError as e:
            raise FaceStorageError(f"Failed to load face image: {e}") from e
    
    def delete_face(self, relative_path: str) -> bool:
        """
        Delete face image from storage.
        
        Args:
            relative_path: Relative path (as returned by save_face)
            
        Returns:
            True if deleted, False if not found
        """
        full_path = self.storage_dir / relative_path
        
        if not full_path.exists():
            logger.warning(f"Face image not found for deletion: {relative_path}")
            return False
        
        try:
            full_path.unlink()
            logger.debug(f"Deleted face image: {relative_path}")
            
            # Try to remove empty person directory
            person_dir = full_path.parent
            if person_dir.exists() and not any(person_dir.iterdir()):
                person_dir.rmdir()
                logger.debug(f"Removed empty person directory: {person_dir.name}")
            
            return True
        except IOError as e:
            logger.error(f"Failed to delete face image: {e}")
            return False
    
    def delete_person_faces(self, person_id: UUID) -> int:
        """
        Delete all face images for a person.
        
        Args:
            person_id: Person's UUID
            
        Returns:
            Number of images deleted
        """
        person_dir = self._get_person_dir(person_id)
        
        if not person_dir.exists():
            return 0
        
        count = 0
        for file_path in person_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    count += 1
                except IOError:
                    pass
        
        # Remove directory
        try:
            person_dir.rmdir()
        except IOError:
            pass
        
        logger.info(f"Deleted {count} face images for person {person_id}")
        return count
    
    def get_face_url(self, relative_path: str, base_url: str = "/data/person_faces") -> str:
        """
        Get URL for accessing face image.
        
        Args:
            relative_path: Relative path (as returned by save_face)
            base_url: Base URL for face images
            
        Returns:
            Full URL to access the image
        """
        return f"{base_url}/{relative_path}"
    
    def exists(self, relative_path: str) -> bool:
        """Check if face image exists."""
        full_path = self.storage_dir / relative_path
        return full_path.exists()


# Singleton instance
_storage: Optional[FaceStorage] = None


def get_face_storage() -> FaceStorage:
    """Get or create singleton FaceStorage instance."""
    global _storage
    if _storage is None:
        _storage = FaceStorage()
    return _storage
