"""
Face processing module using InsightFace.

Provides face detection, alignment, and embedding generation.
Embeddings are 512-dimensional vectors, L2 normalized for cosine similarity.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# InsightFace imports with lazy loading to handle missing dependencies gracefully
_insightface_app = None


def _get_face_app():
    """Lazy load InsightFace FaceAnalysis app."""
    global _insightface_app
    if _insightface_app is None:
        try:
            from insightface.app import FaceAnalysis
            _insightface_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
            _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace FaceAnalysis initialized successfully")
        except ImportError as e:
            logger.error(f"InsightFace not installed: {e}")
            raise ImportError(
                "InsightFace is required for face recognition. "
                "Install with: pip install insightface"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise
    return _insightface_app


@dataclass
class FaceResult:
    """Result of face detection and embedding extraction."""
    embedding: NDArray[np.float32]  # 512-dim, L2 normalized
    det_score: float  # Detection confidence (0-1)
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: Optional[NDArray[np.float32]] = None  # 5-point facial landmarks


class FaceProcessorError(Exception):
    """Base exception for face processing errors."""
    pass


class NoFaceDetectedError(FaceProcessorError):
    """Raised when no face is detected in the image."""
    pass


class MultipleFacesError(FaceProcessorError):
    """Raised when multiple faces are detected (if strict mode)."""
    def __init__(self, count: int):
        self.count = count
        super().__init__(f"Multiple faces detected: {count}")


class FaceProcessor:
    """
    Face processor using InsightFace for detection and embedding.
    
    Usage:
        processor = FaceProcessor()
        result = processor.process_image(image_bytes)
        embedding = result.embedding  # 512-dim, L2 normalized
    """
    
    EMBEDDING_DIM = 512
    DEFAULT_THRESHOLD = 0.45  # Cosine similarity threshold for matching
    
    def __init__(self, select_best_on_multiple: bool = True):
        """
        Initialize face processor.
        
        Args:
            select_best_on_multiple: If True, select face with highest det_score
                                    when multiple faces detected. If False, raise error.
        """
        self.select_best_on_multiple = select_best_on_multiple
        self._app = None
    
    @property
    def app(self):
        """Lazy load face analysis app."""
        if self._app is None:
            self._app = _get_face_app()
        return self._app
    
    def process_image(self, image_data: bytes) -> FaceResult:
        """
        Process image bytes and extract face embedding.
        
        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.)
            
        Returns:
            FaceResult with embedding, det_score, bbox, and landmarks
            
        Raises:
            NoFaceDetectedError: If no face is detected
            MultipleFacesError: If multiple faces detected and select_best_on_multiple=False
            FaceProcessorError: If image cannot be processed
        """
        import cv2
        
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise FaceProcessorError("Failed to decode image")
        
        return self.process_array(img)
    
    def process_array(self, img: NDArray[np.uint8]) -> FaceResult:
        """
        Process numpy array image and extract face embedding.
        
        Args:
            img: Image as numpy array (BGR format, HWC)
            
        Returns:
            FaceResult with embedding, det_score, bbox, and landmarks
            
        Raises:
            NoFaceDetectedError: If no face is detected
            MultipleFacesError: If multiple faces detected and select_best_on_multiple=False
        """
        # Detect faces
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise NoFaceDetectedError("No face detected in image")
        
        if len(faces) > 1:
            if self.select_best_on_multiple:
                # Select face with highest detection score
                faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
                logger.info(f"Multiple faces ({len(faces)}) detected, selecting best (score={faces[0].det_score:.3f})")
            else:
                raise MultipleFacesError(len(faces))
        
        face = faces[0]
        
        # Extract embedding and normalize
        embedding = face.embedding
        embedding = self._l2_normalize(embedding)
        
        # Extract bounding box
        bbox = tuple(int(x) for x in face.bbox)
        
        return FaceResult(
            embedding=embedding,
            det_score=float(face.det_score),
            bbox=bbox,
            landmarks=face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
        )
    
    def _l2_normalize(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2 normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    
    @staticmethod
    def cosine_similarity(emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Assumes embeddings are L2 normalized, so dot product equals cosine similarity.
        
        Args:
            emb1: First embedding (512-dim)
            emb2: Second embedding (512-dim)
            
        Returns:
            Cosine similarity score (-1 to 1, higher is more similar)
        """
        return float(np.dot(emb1, emb2))
    
    @staticmethod
    def validate_embedding(embedding: list[float]) -> bool:
        """
        Validate an embedding list.
        
        Args:
            embedding: List of 512 floats
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embedding, (list, np.ndarray)):
            return False
        
        arr = np.array(embedding, dtype=np.float32)
        
        # Check dimension
        if arr.shape != (512,):
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False
        
        return True


# Singleton instance for reuse
_processor: Optional[FaceProcessor] = None


def get_face_processor() -> FaceProcessor:
    """Get or create singleton FaceProcessor instance."""
    global _processor
    if _processor is None:
        _processor = FaceProcessor(select_best_on_multiple=True)
    return _processor
