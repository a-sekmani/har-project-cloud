# Development

## Project Structure

```
har-project-cloud/
  app/
    __init__.py
    main.py              # FastAPI application and routes
    schemas.py           # Pydantic request/response (inference, label, predict, ingest)
    edge_schemas.py      # Pydantic schemas for edge frame_event (for future use)
    config.py            # Configuration (API_KEY, DATABASE_URL, MODELS_DIR, etc.)
    logging.py           # Logging setup
    health.py            # Health check (GET /health)
    database.py          # DB connection and session
    models.py            # SQLAlchemy ORM (Device, ActivityEvent, Person, PersonFace,
                         # GalleryVersion, PoseWindow, WindowPrediction)
    models_meta.py       # List models and load label_map / version from models/
    services.py          # Business logic (windows, predictions, dashboard data)
    api_schemas.py       # Pydantic API response schemas
    utils.py             # Shared utilities (e.g. isoformat_utc for datetime serialization)
    face/                # Face recognition
      routes.py          # /v1/persons, /v1/persons/{person_id}/faces, /v1/face-gallery, /v1/face-gallery/version
      processor.py       # Face detection and embedding (InsightFace)
      storage.py         # Face image storage
      schemas.py         # Pydantic schemas for face API
    ml/                  # Feature extraction and ONNX inference
      features.py        # keypoints → model input (1, window_size, 85)
      onnx_runner.py     # ONNX load and inference
    templates/           # HTML pages
      dashboard.html
      windows.html       # Recent windows (Date, Time, ID, Device, FPS, Track, Person, etc.)
      label.html
      persons.html
      person_detail.html
      device_dashboard.html
  data/
    person_faces/        # Stored face images (served at /face-images)
  scripts/
    seed_windows.py      # Insert pose_windows from JSON/JSONL
  tests/
    test_health.py, test_infer_schema.py, test_api_edge_cases.py, test_golden_edge.py
    test_normalize.py, test_multiple_devices.py, test_response_format.py
    test_face_api.py, test_ingest.py
    fixtures/
  models/                # ONNX models: <model_key>/model.onnx, label_map.json, model_meta.json
  docs/                  # Documentation
  alembic/               # Database migrations
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
```

## Scripts

### seed_windows.py

Inserts pose windows from a JSON or JSONL file (e.g. HAR-WindowNet export or sample data). If each record includes a `keypoints` field, it is stored in `pose_windows.keypoints_json` so that `POST /v1/windows/{id}/predict` can run ONNX on it.

**Usage:**

```bash
python scripts/seed_windows.py --from labelled.jsonl [--limit 50]
```

**Example:** Seed 20 windows, then open `http://localhost:8000/windows` and run predict on the new windows.

## Testing

Run tests:

```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pytest
pytest -v
pytest tests/test_health.py tests/test_face_api.py -v
pytest --cov=app --cov-report=term-missing
```

Test modules: `test_health.py`, `test_infer_schema.py`, `test_api_edge_cases.py`, `test_golden_edge.py`, `test_normalize.py`, `test_multiple_devices.py`, `test_response_format.py`, `test_face_api.py`, `test_ingest.py`. Many tests use in-memory SQLite; some may expect PostgreSQL (e.g. `DATABASE_URL`).

## Logging

- **Inference requests:** Logged with request_id, device_id, camera_id, num_people, latency_ms. Full keypoints are not logged.
- **Ingest:** See [Edge Integration – Logging](edge-integration.md#logging).

Log level is controlled by `LOG_LEVEL` (default `INFO`). Use `DEBUG` for more detail during development or troubleshooting.
