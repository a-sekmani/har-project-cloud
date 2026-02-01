# Cloud HAR - Human Activity Recognition API

Cloud service for receiving skeleton data from edge devices and returning mock activity recognition results. It accepts **frame-level events** from the edge (`POST /v1/edge/events`), normalizes and aggregates them into windows in memory; it also accepts **window-level inference requests** (`POST /v1/activity/infer`) and returns mock activity predictions. Results are stored in PostgreSQL and can be viewed through a web dashboard.

## Requirements

- Python 3.11+
- Docker and Docker Compose (required)
- PostgreSQL (managed via Docker Compose)

## Project Structure

```
har-project-cloud/
  app/
    __init__.py
    main.py              # FastAPI application and endpoints
    schemas.py           # Pydantic models for inference request/response
    edge_schemas.py      # Pydantic schemas for edge frame_event (POST /v1/edge/events)
    config.py            # Configuration settings
    logging.py           # Logging setup
    health.py            # Health check endpoint
    database.py          # Database connection and session management
    models.py            # SQLAlchemy ORM models
    services.py          # Business logic and database operations
    api_schemas.py       # Pydantic schemas for API responses
    normalize.py         # Edge payload → InternalFrame normalization (COCO-17)
    aggregation.py      # In-memory buffers, window build (frame_event → window payload)
    templates/           # Jinja2 templates for dashboard
      dashboard.html
      device_dashboard.html
  docs/
    EDGE_DATA_SHAPE.md   # Edge data contract and examples
  tests/                 # Test suite (82 automated tests)
    test_health.py
    test_infer_schema.py
    test_database.py
    test_api_edge_cases.py
    test_edge_aggregation.py  # POST /v1/edge/events, aggregation, debug endpoints
    test_normalize.py        # normalize_frame_event unit tests
    test_aggregation.py      # ingest_internal_frame unit tests
    test_golden_edge.py
    test_multiple_people.py
    test_multiple_devices.py
    test_response_format.py
    test_integration.py
    fixtures/
      golden_edge_payload.json
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

## API Endpoints

### GET /health

Health check endpoint for monitoring and load balancers.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /v1/activity/infer

Main inference endpoint. Accepts skeleton window data and returns activity predictions. Results are automatically saved to the database.

**Headers:**
- `X-API-Key`: Required (default value: `dev-key`)

**Request Body:**
```json
{
  "schema_version": 1,
  "device_id": "pi-001",
  "camera_id": "cam-1",
  "window": {
    "ts_start_ms": 1737970000000,
    "ts_end_ms": 1737970001000,
    "fps": 30,
    "size": 30
  },
  "people": [
    {
      "track_id": 7,
      "keypoints": [
        [[0.52,0.18,0.91],[0.50,0.22,0.88], ...],  // 17 or 25 keypoints
        ...
      ],  // window.size frames
      "pose_conf": 0.83
    }
  ]
}
```

**Response:**
```json
{
  "schema_version": 1,
  "device_id": "pi-001",
  "camera_id": "cam-1",
  "window": {
    "ts_start_ms": 1737970000000,
    "ts_end_ms": 1737970001000,
    "fps": 30,
    "size": 30
  },
  "results": [
    {
      "track_id": 7,
      "activity": "standing",
      "confidence": 0.6,
      "top_k": [
        {"label": "standing", "score": 0.6},
        {"label": "walking", "score": 0.2},
        {"label": "sitting", "score": 0.2}
      ]
    }
  ]
}
```

## Validation Rules

- `schema_version` must equal 1
- `window.size` between 10 and 120
- `window.fps` between 1 and 120
- `people` must not be empty
- For each person:
  - `track_id >= 0`
  - `pose_conf` between 0 and 1
  - `keypoints` shape must be `[T][K][3]` where:
    - `T == window.size`
    - `K = 17` or `25`
    - Last dimension = 3 (x, y, confidence)

## Mock Inference Logic

- If `pose_conf < 0.4` → `activity = "unknown"`, `confidence = 0.2`
- Otherwise → `activity = "standing"`, `confidence = 0.6`

## Edge frame events (POST /v1/edge/events)

Accepts frame-level events from the edge (`event_type: "frame_event"`). Frames are normalized and aggregated in memory per `(device_id, camera_id, track_id)`; when a buffer reaches the window size (default 30 frames), a Cloud window payload is built and logged (no DB, no inference call in this phase).

**Headers:**
- `X-API-Key`: Required (same as inference; 401 if missing or invalid)

**Contract:**
- **camera_id priority:** 1. `source.camera_id` → 2. Query `?camera_id=` → 3. Header `X-Camera-Id` → 4. Default (e.g. `cam-1`)
- **ts_unix_ms:** int or float accepted; converted to int in cloud
- **persons:** required field; value can be `[]`
- **keypoints:** 17 objects `{ name, x, y, c }` with names = COCO-17 set; order in request is free; cloud sorts by COCO-17 in normalization
- **x, y:** currently treated as **pixel** coordinates in ingestion
- Extra fields (e.g. `bbox`, `score`) allowed (`extra="allow"`)

Full contract and examples: **[docs/EDGE_DATA_SHAPE.md](docs/EDGE_DATA_SHAPE.md)**

**Debug endpoints** (require `X-API-Key`):
- `GET /debug/buffers` — current aggregation buffers (key, frame_count, last_ts_ms) and `frame_events_received`
- `GET /debug/windows?n=20` — last n completed windows (metadata only; `n` between 1 and 100)

**Flow:** Edge sends `frame_event` → validated by `EdgeFrameEventSchema` → normalized to `InternalFrame` (COCO-17 order) → buffered per `(device_id, camera_id, track_id)` → when buffer reaches 30 frames, a window payload is built and logged (not stored, not sent to inference in this phase).

## Data Retrieval Endpoints

### GET /v1/events

Get recent activity events from all devices.

**Query Parameters:**
- `limit` (optional): Number of events to return (1-1000, default: 100)

**Example:**
```bash
curl "http://localhost:8000/v1/events?limit=50"
```

### GET /v1/devices

Get all registered devices.

**Example:**
```bash
curl "http://localhost:8000/v1/devices"
```

### GET /v1/devices/{device_id}/events

Get activity events for a specific device.

**Query Parameters:**
- `limit` (optional): Number of events to return (1-1000, default: 200)

**Example:**
```bash
curl "http://localhost:8000/v1/devices/pi-001/events?limit=100"
```

## Dashboard

The application includes a web-based dashboard for viewing activity events.

### Main Dashboard

View recent events from all devices:
```
http://localhost:8000/dashboard
```

**Features:**
- Table of recent events with device, camera, track ID, activity, and confidence
- Auto-refresh every 3 seconds
- Clickable device IDs to view device-specific events
- Color-coded confidence levels

### Device Dashboard

View events for a specific device:
```
http://localhost:8000/dashboard/devices/{device_id}
```

**Example:**
```
http://localhost:8000/dashboard/devices/pi-001
```

## Testing

The project includes **82 automated tests** covering:
- Health check endpoint
- Request validation and mock inference schema
- Database operations
- API edge cases (limits, empty DB, sorting)
- **Edge ingestion:** POST /v1/edge/events (validation, 401, camera_id priority, extra fields, keypoints)
- **Aggregation:** window completion, multi-person buffers, debug endpoints
- **Normalization:** `normalize_frame_event` (COCO-17 order, undetected points, ts conversion, multi-person)
- **Aggregation unit:** `ingest_internal_frame` (buffers, window metadata, buffer clear)
- Multiple people in inference requests
- Multiple devices and upsert behavior
- Response format validation
- End-to-end integration workflows

### Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_health.py -v
pytest tests/test_database.py -v
pytest tests/test_edge_aggregation.py -v
pytest tests/test_normalize.py -v
pytest tests/test_aggregation.py -v
pytest tests/test_api_edge_cases.py -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing
```

### Test Files

- `test_health.py`: Health check endpoint
- `test_infer_schema.py`: Request validation and mock inference
- `test_database.py`: Database operations and endpoints
- `test_edge_aggregation.py`: POST /v1/edge/events, aggregation, debug buffers/windows, API key, camera_id
- `test_normalize.py`: `normalize_frame_event` unit tests (COCO-17 order, undetected points, multi-person)
- `test_aggregation.py`: `ingest_internal_frame` unit tests (buffers, window completion, buffer clear)
- `test_golden_edge.py`: Golden edge payload and inference
- `test_api_edge_cases.py`: Edge cases and validation
- `test_multiple_people.py`: Multiple people scenarios
- `test_multiple_devices.py`: Multiple devices and upsert
- `test_response_format.py`: Response format validation
- `test_integration.py`: End-to-end workflows

**Note:** Most tests use in-memory SQLite and don't require Docker or PostgreSQL. Edge/aggregation tests don't use the database.

## Environment Variables

- `API_KEY`: API key for authentication (inference and edge/debug endpoints; default: `dev-key`)
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar`)
- `EDGE_WINDOW_SIZE`: Frames per aggregation window (default: `30`)
- `EDGE_CAMERA_ID_DEFAULT`: Default camera_id when not provided by edge (default: `cam-1`)

**Note:** When using Docker Compose, `DATABASE_URL` is set to the postgres service; other defaults apply.

## Logging

Each inference request is logged with:
- `request_id` (UUID)
- `device_id`
- `camera_id`
- `num_people`
- `latency_ms`

**Note:** Full keypoints are not logged to avoid log bloat.

## Database

The application uses PostgreSQL to store:
- **devices**: Registered edge devices (auto-created on first inference)
- **activity_events**: Inference results with activity predictions

Database migrations are managed with Alembic and run automatically on Docker Compose startup. If the database is unavailable, the API returns **503** (and the dashboard shows a friendly error page) instead of 500.

## Quick Start

1. **Start services:**
   ```bash
   docker-compose up -d
   ```

2. **Check health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Send inference request:**
   ```bash
   curl -X POST http://localhost:8000/v1/activity/infer \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev-key" \
     -d @test_request.json
   ```

4. **View dashboard:**
   Open `http://localhost:8000/dashboard` in your browser

5. **Query events:**
   ```bash
   curl "http://localhost:8000/v1/events?limit=10"
   ```

6. **Test edge ingestion** (optional): Send a `frame_event` to `POST /v1/edge/events` with `X-API-Key: dev-key`, then check `GET /debug/buffers`. See [docs/EDGE_DATA_SHAPE.md](docs/EDGE_DATA_SHAPE.md) for the full payload format and examples.
