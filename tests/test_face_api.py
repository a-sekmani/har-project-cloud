"""Tests for face recognition API endpoints."""

import pytest
from uuid import uuid4
from unittest.mock import patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient

from app.database import Base, get_db, engine, SessionLocal
import app.models  # noqa: F401 — register all ORM tables (persons, person_faces, etc.) with Base.metadata


@pytest.fixture(scope="function")
def db_session():
    """Create test DB session with all tables (persons, person_faces, gallery_versions, pose_windows, etc.)."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_face_processor():
    """Mock face processor to avoid loading InsightFace."""
    with patch("app.face.routes.get_face_processor") as mock:
        processor = MagicMock()
        processor.EMBEDDING_DIM = 512
        processor.DEFAULT_THRESHOLD = 0.45
        
        # Create a mock FaceResult
        mock_result = MagicMock()
        mock_result.embedding = np.random.randn(512).astype(np.float32)
        mock_result.embedding = mock_result.embedding / np.linalg.norm(mock_result.embedding)
        mock_result.det_score = 0.95
        mock_result.bbox = (10, 10, 100, 100)
        
        processor.process_image.return_value = mock_result
        mock.return_value = processor
        yield processor


@pytest.fixture
def client(mock_face_processor, db_session):
    """Test client with mocked face processor and DB override so all tables exist."""
    from app.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def api_headers():
    """API headers with authentication."""
    return {"X-API-Key": "dev-key"}


class TestPersonEndpoints:
    """Tests for person CRUD endpoints."""
    
    def test_create_person(self, client, api_headers):
        """Test creating a new person."""
        response = client.post(
            "/v1/persons",
            json={"name": "Test Person"},
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Person"
        assert data["is_active"] is True
        assert data["face_count"] == 0
    
    def test_create_person_empty_name(self, client, api_headers):
        """Test creating person with empty name fails."""
        response = client.post(
            "/v1/persons",
            json={"name": ""},
            headers=api_headers
        )
        assert response.status_code == 422
    
    def test_list_persons(self, client, api_headers):
        """Test listing persons."""
        # Create a person first
        client.post("/v1/persons", json={"name": "Person 1"}, headers=api_headers)
        client.post("/v1/persons", json={"name": "Person 2"}, headers=api_headers)
        
        response = client.get("/v1/persons", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert "persons" in data
        assert "total" in data
        assert data["total"] >= 2
    
    def test_list_persons_filter_active(self, client, api_headers):
        """Test listing active persons only."""
        response = client.get("/v1/persons?is_active=true", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        for person in data["persons"]:
            assert person["is_active"] is True
    
    def test_get_person(self, client, api_headers):
        """Test getting a single person."""
        # Create a person
        create_resp = client.post(
            "/v1/persons",
            json={"name": "Get Test"},
            headers=api_headers
        )
        person_id = create_resp.json()["id"]
        
        response = client.get(f"/v1/persons/{person_id}", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == person_id
        assert data["name"] == "Get Test"
    
    def test_get_person_not_found(self, client, api_headers):
        """Test getting non-existent person returns 404."""
        fake_id = str(uuid4())
        response = client.get(f"/v1/persons/{fake_id}", headers=api_headers)
        assert response.status_code == 404
    
    def test_update_person_name(self, client, api_headers):
        """Test updating person's name."""
        # Create a person
        create_resp = client.post(
            "/v1/persons",
            json={"name": "Original Name"},
            headers=api_headers
        )
        person_id = create_resp.json()["id"]
        
        response = client.patch(
            f"/v1/persons/{person_id}",
            json={"name": "Updated Name"},
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
    
    def test_update_person_deactivate(self, client, api_headers):
        """Test deactivating a person."""
        # Create a person
        create_resp = client.post(
            "/v1/persons",
            json={"name": "To Deactivate"},
            headers=api_headers
        )
        person_id = create_resp.json()["id"]
        
        response = client.patch(
            f"/v1/persons/{person_id}",
            json={"is_active": False},
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False
    
    def test_delete_person(self, client, api_headers):
        """Test deleting a person."""
        # Create a person
        create_resp = client.post(
            "/v1/persons",
            json={"name": "To Delete"},
            headers=api_headers
        )
        person_id = create_resp.json()["id"]
        
        response = client.delete(f"/v1/persons/{person_id}", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        
        # Verify person is gone
        get_resp = client.get(f"/v1/persons/{person_id}", headers=api_headers)
        assert get_resp.status_code == 404


class TestFaceUploadEndpoint:
    """Tests for face upload endpoint and duplicate-filename skip behavior."""

    def test_upload_face_success(self, client, api_headers, tmp_path):
        """Test uploading a face image returns added=1 and stores original_filename."""
        from app.face.storage import FaceStorage
        with patch("app.face.routes.get_face_storage") as mock_storage:
            mock_storage.return_value = FaceStorage(tmp_path)
            create_resp = client.post(
                "/v1/persons",
                json={"name": "Upload Test"},
                headers=api_headers
            )
            person_id = create_resp.json()["id"]
            image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # minimal JPEG-like header
            response = client.post(
                f"/v1/persons/{person_id}/faces",
                files=[("files", ("photo.jpg", image_bytes, "image/jpeg"))],
                headers=api_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["added"] == 1
            assert data["skipped"] == 0
            assert "gallery_version" in data
            assert data["person_id"] == person_id

    def test_upload_face_duplicate_filename_skipped(self, client, api_headers, tmp_path):
        """Test uploading the same filename again is skipped (not re-added)."""
        from app.face.storage import FaceStorage
        with patch("app.face.routes.get_face_storage") as mock_storage:
            mock_storage.return_value = FaceStorage(tmp_path)
            create_resp = client.post(
                "/v1/persons",
                json={"name": "Duplicate Test"},
                headers=api_headers
            )
            person_id = create_resp.json()["id"]
            image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
            first = client.post(
                f"/v1/persons/{person_id}/faces",
                files=[("files", ("same_name.jpg", image_bytes, "image/jpeg"))],
                headers=api_headers
            )
            assert first.status_code == 200
            assert first.json()["added"] == 1
            assert first.json()["skipped"] == 0
            second = client.post(
                f"/v1/persons/{person_id}/faces",
                files=[("files", ("same_name.jpg", image_bytes + b"extra", "image/jpeg"))],
                headers=api_headers
            )
            assert second.status_code == 200
            data = second.json()
            assert data["added"] == 0
            assert data["skipped"] == 1

    def test_upload_response_includes_skipped(self, client, api_headers, tmp_path):
        """Test upload response always includes skipped field."""
        from app.face.storage import FaceStorage
        with patch("app.face.routes.get_face_storage") as mock_storage:
            mock_storage.return_value = FaceStorage(tmp_path)
            create_resp = client.post(
                "/v1/persons",
                json={"name": "Skip Field Test"},
                headers=api_headers
            )
            person_id = create_resp.json()["id"]
            image_bytes = b"\xff\xd8\xff"
            response = client.post(
                f"/v1/persons/{person_id}/faces",
                files=[("files", ("one.jpg", image_bytes, "image/jpeg"))],
                headers=api_headers
            )
            assert response.status_code == 200
            assert "skipped" in response.json()


class TestFaceGalleryEndpoint:
    """Tests for face gallery endpoint."""
    
    def test_get_face_gallery_empty(self, client, api_headers):
        """Test getting empty face gallery."""
        response = client.get("/v1/face-gallery", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert "gallery_version" in data
        assert data["embedding_dim"] == 512
        assert data["threshold"] == 0.45
        assert "people" in data
    
    def test_get_gallery_version(self, client, api_headers):
        """Test getting gallery version."""
        response = client.get("/v1/face-gallery/version", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "created_at" in data


class TestIngestWithPerson:
    """Tests for window ingest with person data."""
    
    def test_ingest_without_person(self, client, api_headers):
        """Test ingesting window without person data."""
        window_data = {
            "device_id": "test-device",
            "camera_id": "cam-1",
            "track_id": 1,
            "ts_start_ms": 1000000,
            "ts_end_ms": 1001000,
            "fps": 30,
            "window_size": 30,
            "keypoints": [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)]
        }
        
        response = client.post(
            "/v1/windows/ingest?predict=false",
            json=window_data,
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["person_id"] is None
        assert data["person_name"] is None
        assert data["person_conf"] is None
    
    def test_ingest_with_unknown_person(self, client, api_headers):
        """Test ingesting window with unknown person (null person_id)."""
        window_data = {
            "device_id": "test-device",
            "camera_id": "cam-1",
            "track_id": 1,
            "ts_start_ms": 1000000,
            "ts_end_ms": 1001000,
            "fps": 30,
            "window_size": 30,
            "keypoints": [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)],
            "person": {
                "person_id": None,
                "person_name": None,
                "person_conf": 0.0,
                "gallery_version": "v1"
            }
        }
        
        response = client.post(
            "/v1/windows/ingest?predict=false",
            json=window_data,
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["person_id"] is None
        assert data["person_conf"] == 0.0
        assert data["gallery_version"] == "v1"
    
    def test_ingest_with_known_person(self, client, api_headers):
        """Test ingesting window with known person."""
        # Create a person first
        create_resp = client.post(
            "/v1/persons",
            json={"name": "Known Person"},
            headers=api_headers
        )
        person_id = create_resp.json()["id"]
        
        window_data = {
            "device_id": "test-device",
            "camera_id": "cam-1",
            "track_id": 1,
            "ts_start_ms": 1000000,
            "ts_end_ms": 1001000,
            "fps": 30,
            "window_size": 30,
            "keypoints": [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)],
            "person": {
                "person_id": person_id,
                "person_name": "Known Person",
                "person_conf": 0.85,
                "gallery_version": "v1"
            }
        }
        
        response = client.post(
            "/v1/windows/ingest?predict=false",
            json=window_data,
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["person_id"] == person_id
        assert data["person_name"] == "Known Person"
        assert data["person_conf"] == 0.85
    
    def test_ingest_with_invalid_person_id(self, client, api_headers):
        """Test ingesting window with non-existent person_id returns 400."""
        fake_person_id = str(uuid4())
        window_data = {
            "device_id": "test-device",
            "camera_id": "cam-1",
            "track_id": 1,
            "ts_start_ms": 1000000,
            "ts_end_ms": 1001000,
            "fps": 30,
            "window_size": 30,
            "keypoints": [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)],
            "person": {
                "person_id": fake_person_id,
                "person_name": "Unknown",
                "person_conf": 0.5,
                "gallery_version": "v1"
            }
        }
        
        response = client.post(
            "/v1/windows/ingest?predict=false",
            json=window_data,
            headers=api_headers
        )
        assert response.status_code == 400
        assert "Person not found" in response.json()["detail"]
    
    def test_ingest_with_invalid_person_conf(self, client, api_headers):
        """Test ingesting window with invalid person_conf returns 422."""
        window_data = {
            "device_id": "test-device",
            "camera_id": "cam-1",
            "track_id": 1,
            "ts_start_ms": 1000000,
            "ts_end_ms": 1001000,
            "fps": 30,
            "window_size": 30,
            "keypoints": [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)],
            "person": {
                "person_id": None,
                "person_name": None,
                "person_conf": 1.5,  # Invalid: > 1.0
                "gallery_version": "v1"
            }
        }
        
        response = client.post(
            "/v1/windows/ingest?predict=false",
            json=window_data,
            headers=api_headers
        )
        assert response.status_code == 422


class TestFaceProcessorUnit:
    """Unit tests for FaceProcessor class."""
    
    def test_validate_embedding_valid(self):
        """Test validating a valid embedding."""
        from app.face.processor import FaceProcessor
        
        embedding = list(np.random.randn(512).astype(np.float32))
        assert FaceProcessor.validate_embedding(embedding) is True
    
    def test_validate_embedding_wrong_length(self):
        """Test validating embedding with wrong length."""
        from app.face.processor import FaceProcessor
        
        embedding = list(np.random.randn(256).astype(np.float32))
        assert FaceProcessor.validate_embedding(embedding) is False
    
    def test_validate_embedding_with_nan(self):
        """Test validating embedding with NaN values."""
        from app.face.processor import FaceProcessor
        
        embedding = list(np.random.randn(512).astype(np.float32))
        embedding[100] = float('nan')
        assert FaceProcessor.validate_embedding(embedding) is False
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from app.face.processor import FaceProcessor
        
        # Identical normalized vectors should have similarity 1.0
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        sim = FaceProcessor.cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5
        
        # Orthogonal vectors should have similarity ~0
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[0] = 1.0
        emb3 = np.zeros(512, dtype=np.float32)
        emb3[1] = 1.0
        
        sim = FaceProcessor.cosine_similarity(emb2, emb3)
        assert abs(sim) < 1e-5


class TestFaceStorageUnit:
    """Unit tests for FaceStorage class."""
    
    def test_storage_init(self, tmp_path):
        """Test storage initialization creates directory."""
        from app.face.storage import FaceStorage
        
        storage_dir = tmp_path / "faces"
        storage = FaceStorage(storage_dir)
        
        assert storage_dir.exists()
    
    def test_save_and_load_face(self, tmp_path):
        """Test saving and loading face image."""
        from app.face.storage import FaceStorage
        
        storage = FaceStorage(tmp_path / "faces")
        person_id = uuid4()
        face_id = uuid4()
        image_data = b"fake image data"
        
        path = storage.save_face(person_id, face_id, image_data, "test.jpg")
        assert path is not None
        
        loaded = storage.load_face(path)
        assert loaded == image_data
    
    def test_delete_face(self, tmp_path):
        """Test deleting face image."""
        from app.face.storage import FaceStorage
        
        storage = FaceStorage(tmp_path / "faces")
        person_id = uuid4()
        face_id = uuid4()
        image_data = b"fake image data"
        
        path = storage.save_face(person_id, face_id, image_data)
        assert storage.exists(path)
        
        deleted = storage.delete_face(path)
        assert deleted is True
        assert not storage.exists(path)
    
    def test_delete_person_faces(self, tmp_path):
        """Test deleting all faces for a person."""
        from app.face.storage import FaceStorage
        
        storage = FaceStorage(tmp_path / "faces")
        person_id = uuid4()
        
        # Save multiple faces
        for i in range(3):
            storage.save_face(person_id, uuid4(), f"image {i}".encode())
        
        count = storage.delete_person_faces(person_id)
        assert count == 3
