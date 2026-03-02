# Cloud HAR - Human Activity Recognition API

Cloud service for human activity recognition from skeleton (pose) data. It provides:

- **Window-level inference** (`POST /v1/activity/infer`): accepts a full pose window and returns activity predictions (currently mock).
- **Pose windows store** (`pose_windows` table): store windows with optional keypoints for later ONNX prediction. Windows can be seeded from JSON/JSONL via `scripts/seed_windows.py`.
- **ONNX prediction** (Phase 6): run an ONNX model on a stored window that has keypoints (`POST /v1/windows/{id}/predict`); results are saved in `window_predictions` and shown on the dashboard.
- **Dashboard**: view recent windows with labels and predicted activity/confidence; label windows using labels from the selected model.

**Window ingest (Phase 7.A):** The edge can send a full window via `POST /v1/windows/ingest` (same contract as HAR-WindowNet samples); the cloud stores it and optionally runs ONNX. There is no frame-level ingest (`POST /v1/edge/events` or `/v1/frames`); the schema for frame events is in `app/edge_schemas.py` for future use.

## Requirements

- Python 3.11+
- Docker and Docker Compose (required)
- PostgreSQL (managed via Docker Compose)

## Project Structure

```
har-project-cloud/
  app/
    __init__.py
    main.py              # FastAPI application and routes
    schemas.py           # Pydantic request/response (inference, label, predict)
    edge_schemas.py      # Pydantic schemas for edge frame_event (for future ingest)
    config.py            # Configuration (API_KEY, DATABASE_URL, MODELS_DIR, MODEL_KEY_DEFAULT)
    logging.py           # Logging setup
    health.py            # Health check (GET /health)
    database.py          # DB connection and session
    models.py            # SQLAlchemy ORM (Device, ActivityEvent, PoseWindow, WindowPrediction)
    models_meta.py       # List models and load label_map / version from models/
    services.py          # Business logic (windows, predictions, dashboard data)
    api_schemas.py       # Pydantic API response schemas
    ml/                  # Phase 6: feature extraction and ONNX inference
      features.py        # keypoints → model input (1, 30, 85) with normalization and velocity
      onnx_runner.py     # Load ONNX, run inference, return pred_label / pred_conf / probs
    templates/
      dashboard.html     # Home page (placeholder)
      windows.html       # Recent windows + predicted activity / confidence
      label.html         # Label pose windows (labels from model label_map)
      device_dashboard.html
  scripts/
    seed_windows.py      # Insert pose_windows from JSON/JSONL (with keypoints if present)
  tests/
    test_health.py, test_database.py, test_infer_schema.py, test_api_edge_cases.py
    test_golden_edge.py, test_normalize.py, test_multiple_people.py, test_multiple_devices.py
    test_response_format.py, test_integration.py
    fixtures/
      golden_edge_payload.json
  models/               # ONNX models: models/<model_key>/model.onnx, label_map.json, model_meta.json
  docs/
    API.md              # Complete API documentation
  alembic/              # Database migrations
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
```

## Installation and Running

### Using Docker

```bash
# Build the image
docker build -t cloud-har .

# Run the container (requires PostgreSQL; use docker-compose for full stack)
docker run -p 8000:8000 -e DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db cloud-har
```

### Using docker-compose (Recommended)

```bash
# Start all services (API + PostgreSQL)
docker-compose up -d

# Rebuild after code changes
docker-compose up -d --build cloud-har

# View logs
docker-compose logs -f cloud-har

# Stop services
docker-compose down

# Stop and remove volumes (deletes database data)
docker-compose down -v
```

The docker-compose setup includes:
- **cloud-har**: API service (port 8000)
- **postgres**: PostgreSQL database (exposed on host port 5433)
- Automatic database migrations on startup (`alembic upgrade head`)

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up database: start Postgres (e.g. docker-compose up -d postgres)
# Default DATABASE_URL uses localhost:5433 (docker-compose maps postgres to 5433).
# If your Postgres runs on port 5432, set: export DATABASE_URL="postgresql+psycopg://cloudhar:cloudhar@localhost:5432/cloudhar"

# Run database migrations
alembic upgrade head

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Phase 6: Running with ONNX

Place an exported HAR-WindowNet-style model under `models/`. Default path is `./models` (override with `MODELS_DIR`). Each model lives in a subdirectory named by `model_key` (e.g. `models/c_norm_vel/`) with:

- `model.onnx` — exported ONNX model (input shape e.g. `[1, 30, 85]`)
- `label_map.json` — mapping from class index to label name (e.g. `id_to_name`)
- `model_meta.json` — e.g. `input_shape`, `window_size`, `input_features`, `feature_spec`

Set `MODEL_KEY_DEFAULT` (e.g. `edge17_v6_lowlr`) for the dashboard default.

**Dependencies:** `numpy`, `onnxruntime` (in `pyproject.toml`). Install with `pip install -e .` or `pip install numpy onnxruntime`.

**Feature extraction pipeline** (must match training):
1. **Center on hips**: Subtract hips midpoint (keypoints 11, 12) from all keypoints
2. **Scale by shoulders**: Divide by shoulder width (distance between keypoints 5, 6)
3. **Clamp**: Clip x, y values to [-3, 3]
4. **Velocity**: Compute frame-to-frame difference (first frame = 0)
5. **Feature layout**: 85 features per frame = 51 pose [x,y,conf × 17] + 34 velocity [dx,dy × 17]

## API Documentation

For complete API documentation with all endpoints, request/response formats, and examples, see **[docs/API.md](docs/API.md)**.

### API Overview

All API endpoints (except `/health`) require the `X-API-Key` header (default: `dev-key`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_key}` | GET | Get model metadata |
| `/v1/windows` | GET | List pose windows |
| `/v1/windows/{id}` | GET | Get single window |
| `/v1/windows/{id}/label` | POST | Set window label |
| `/v1/windows/{id}/predict` | POST | Run ONNX prediction |
| `/v1/windows/ingest` | POST | Ingest window from edge |
| `/v1/dashboard/windows` | GET | Dashboard data with filters |
| `/v1/activity/infer` | POST | Mock inference (legacy) |

### Quick Examples

```bash
# Health check
curl http://localhost:8000/health

# List windows
curl -s "http://localhost:8000/v1/windows?limit=10" -H "X-API-Key: dev-key"

# Ingest window from edge (auto-prediction enabled)
curl -s -X POST "http://localhost:8000/v1/windows/ingest" \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  --data-binary @window_sample.json

# Run prediction on existing window
curl -s -X POST "http://localhost:8000/v1/windows/WINDOW_ID/predict" \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"model_key":"edge17_v6_lowlr","store":true,"return_probs":true}'
```

## Web Pages

| URL | Description |
|-----|-------------|
| `/dashboard` | Home page (placeholder for future use) |
| `/windows` | Recent windows with predicted activity/confidence |
| `/windows/label` | Label pose windows using model labels |

## Scripts

- **scripts/seed_windows.py** — Insert pose windows from a JSON or JSONL file (e.g. HAR-WindowNet export or samples). If each record has a `keypoints` field, it is stored in `pose_windows.keypoints_json` so that `POST /v1/windows/{id}/predict` can run ONNX on it.
  - Usage: `python scripts/seed_windows.py --from labelled.jsonl [--limit 50]`
  - Example: `python scripts/seed_windows.py --from labelled.jsonl --limit 20` then open `http://localhost:8000/windows` and run predict on the new windows.

## Testing

Run tests with:

```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pytest
pytest -v                  # verbose
pytest tests/test_health.py tests/test_database.py -v
pytest --cov=app --cov-report=term-missing
```

Test modules include: `test_health.py`, `test_database.py`, `test_infer_schema.py`, `test_api_edge_cases.py`, `test_golden_edge.py`, `test_normalize.py`, `test_multiple_people.py`, `test_multiple_devices.py`, `test_response_format.py`, `test_integration.py`. Many tests use in-memory SQLite; some expect PostgreSQL (e.g. `DATABASE_URL`).

## Environment Variables

- `API_KEY`: API key for all protected endpoints (default: `dev-key`)
- `HOST`, `PORT`: Server bind (default: `0.0.0.0`, `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar`)
- `MODELS_DIR`: Directory for ONNX models (default: `models`)
- `MODEL_KEY_DEFAULT`: Default model key for dashboard (default: `edge17_v6_lowlr`)

With Docker Compose, `DATABASE_URL` is typically set to the postgres service.

## Logging

Each inference request is logged with:
- `request_id` (UUID)
- `device_id`
- `camera_id`
- `num_people`
- `latency_ms`

**Note:** Full keypoints are not logged to avoid log bloat.

## Database

PostgreSQL stores:
- **devices**: Registered devices (e.g. on first inference)
- **activity_events**: Activity predictions (device, camera, track, window metadata, activity, confidence, optional quality fields)
- **pose_windows**: Pose windows (device_id, camera_id, track_id, ts_*, fps, window_size, label, **keypoints_json**, created_at). Keypoints are stored when provided (e.g. by `seed_windows.py` from JSONL with a `keypoints` field).
- **window_predictions** (Phase 6): ONNX results per window (window_id, model_key, pred_label, pred_conf, created_at).

Migrations: Alembic; run `alembic upgrade head` locally or rely on Docker Compose startup. If the DB is unavailable, the API may return 503.

## Quick Start

1. **Start services:** `docker-compose up -d` (or start Postgres only and run the API locally).
2. **Migrations:** `alembic upgrade head`
3. **Run API:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. **Health:** `curl http://localhost:8000/health`
5. **Seed windows (with keypoints):** `python scripts/seed_windows.py --from labelled.jsonl --limit 20` — or **ingest one window from file:** `curl -s -X POST "http://localhost:8000/v1/windows/ingest" -H "X-API-Key: dev-key" -H "Content-Type: application/json" --data-binary @window_sample.json`
6. **List windows:** `curl -s "http://localhost:8000/v1/windows?limit=5" -H "X-API-Key: dev-key"`
7. **Run ONNX predict** (replace `WINDOW_ID`):  
   `curl -s -X POST "http://localhost:8000/v1/windows/WINDOW_ID/predict" -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"model_key":"edge17_v6_lowlr","store":true}'`
8. **Recent Windows:** Open `http://localhost:8000/windows` to see windows and their predicted activity/confidence.
9. **Label page:** `http://localhost:8000/windows/label` to set or change labels on windows.
