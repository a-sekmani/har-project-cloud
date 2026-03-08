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
    constants.py         # Application constants (e.g. ALERT_ACTIVITIES for dashboard)
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
  samples/               # Sample request payloads (inference, manual testing)
    test_request.json, test_request_valid.json, test_request_low_conf.json
  tests/
    conftest.py           # Pytest config (e.g. DATABASE_URL=SQLite)
    fixtures/             # Shared test data (e.g. golden_edge_payload.json)
    unit/                 # Unit tests
      test_health.py, test_normalize.py, test_infer_schema.py, test_database.py
    api/                  # API / endpoint tests
      test_ingest.py, test_face_api.py, test_golden_edge.py, test_multiple_people.py
    integration/          # Integration tests
      test_integration.py
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

### test_scenarios.sh

Runs manual scenarios for the inference API (e.g. multiple people, multiple devices). Run from the project root. Uses sample payloads from `samples/`:

- `samples/test_request.json` — used in the "Multiple Devices" scenario (`POST /v1/activity/infer`).

Other samples in `samples/` (`test_request_valid.json`, `test_request_low_conf.json`) are available for ad‑hoc or manual testing.

```bash
./test_scenarios.sh
```

## Testing

Run tests:

```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pytest
pytest -v
pytest tests/unit/test_health.py tests/api/test_face_api.py -v
pytest tests/unit/ tests/api/ -v
pytest --cov=app --cov-report=term-missing
```

Test layout:
- **unit/** — `test_health.py`, `test_normalize.py`, `test_infer_schema.py`, `test_database.py`
- **api/** — `test_ingest.py`, `test_face_api.py`, `test_golden_edge.py`, `test_multiple_people.py`
- **integration/** — `test_integration.py`

Many tests use in-memory SQLite; some may expect PostgreSQL (e.g. `DATABASE_URL`).

## Web UI

All pages use a consistent layout (max-width 1600px) and a shared **site header** with the app name **HAR Cloud App** and navigation: **Dashboard**, **Recent Windows**, **Person Management**. (Label Windows is not in the main nav; it is reached from within Recent Windows.)

- **Overview Dashboard** (`/dashboard`): Uses `GET /v1/dashboard/overview` to show: **Stats bar** (five cards: Total activities, Well-known person, Detected Activities, Fall Alerts, Last Update with date and time); **Quick filters** (time range: Last 24 hours / Last week / Last month, Model, Person, Camera, Device, Activity, Show only alerts, Only unknown persons, Only known persons); **Activity Distribution** chart (doughnut) and **Activity Timeline** (by hour for 24h, by day for week/month); **Person Presence** table and **Recent Important Events** (alerts) with “View Details” links. Auto-refresh every 5 seconds. See [API – GET /v1/dashboard/overview](API.md#get-v1dashboardoverview).

- **Recent Windows** (`/windows`): Lists pose windows with predictions. Filters: model, device, camera, track, pred label, only with predictions, low confidence. **Pagination**: page numbers (Prev, 1, 2, 3…, Next) at the bottom centre; each page shows up to the selected limit (e.g. 100). Uses `GET /v1/dashboard/windows` with `offset` and `limit`; response is `{ data, has_more }`. "Open Label Windows" button links to Label Windows. See [API – GET /v1/dashboard/windows](API.md#get-v1dashboardwindows).

- **Label Windows** (`/windows/label`): Reached from Recent Windows. Table: Time, Window ID, Device, Camera, Track, Person, Face Conf, Activity, Set label, Predicted, Pred Conf, Match?, Actions. Bulk label and bulk person assignment. **Pagination** as in Recent Windows (centred at bottom). Same `GET /v1/dashboard/windows` API with `offset`.

- **Person Management** (`/persons`) and **Person detail** (`/persons/{id}`): show **Gallery last updated** (date/time) instead of a version number, for when the face gallery was last changed. Person detail uses `GET /v1/persons/{id}/detail` for stats, activity distribution, timeline, and recent windows.

## Logging

- **Inference requests:** Logged with request_id, device_id, camera_id, num_people, latency_ms. Full keypoints are not logged.
- **Ingest:** See [Edge Integration – Logging](edge-integration.md#logging).

Log level is controlled by `LOG_LEVEL` (default `INFO`). Use `DEBUG` for more detail during development or troubleshooting.
