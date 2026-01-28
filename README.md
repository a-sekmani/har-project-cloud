# Cloud HAR - Human Activity Recognition API

Cloud service for receiving Skeleton Windows from edge devices and returning mock activity recognition results. Results are stored in PostgreSQL and can be viewed through a web dashboard.

## Requirements

- Python 3.11+
- Docker and Docker Compose (required)
- PostgreSQL (managed via Docker Compose)

## Project Structure

```
cloud-har/
  app/
    __init__.py
    main.py              # FastAPI application and endpoints
    schemas.py           # Pydantic models for request/response validation
    config.py            # Configuration settings
    logging.py           # Logging setup
    health.py            # Health check endpoint
    database.py          # Database connection and session management
    models.py            # SQLAlchemy ORM models
    services.py          # Business logic and database operations
    api_schemas.py       # Pydantic schemas for API responses
    templates/           # Jinja2 templates for dashboard
      dashboard.html
      device_dashboard.html
  tests/                 # Test suite (40 automated tests)
    test_health.py
    test_infer_schema.py
    test_database.py
    test_api_edge_cases.py
    test_multiple_people.py
    test_multiple_devices.py
    test_response_format.py
    test_integration.py
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

# Run the container
docker run -p 8000:8000 cloud-har
```

### Using docker-compose (Recommended)

```bash
# Start all services (API + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f cloud-har

# Stop services
docker-compose down

# Stop and remove volumes (deletes database data)
docker-compose down -v
```

The docker-compose setup includes:
- **cloud-har**: API service (port 8000)
- **postgres**: PostgreSQL database (port 5433)
- Automatic database migrations on startup

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up database (requires PostgreSQL running)
# Update DATABASE_URL in app/config.py or set environment variable
export DATABASE_URL="postgresql+psycopg://cloudhar:cloudhar@localhost:5432/cloudhar"

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

Activity inference endpoint.

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

The project includes **40 automated tests** covering:
- Health check endpoint
- Request validation and schema
- Database operations
- API edge cases (limits, empty DB, sorting)
- Multiple people in requests
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
pytest tests/test_api_edge_cases.py -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing
```

### Test Files

- `test_health.py`: Health check endpoint (1 test)
- `test_infer_schema.py`: Request validation and mock inference (5 tests)
- `test_database.py`: Database operations and endpoints (4 tests)
- `test_api_edge_cases.py`: Edge cases and validation (16 tests)
- `test_multiple_people.py`: Multiple people scenarios (2 tests)
- `test_multiple_devices.py`: Multiple devices and upsert (4 tests)
- `test_response_format.py`: Response format validation (5 tests)
- `test_integration.py`: End-to-end workflows (3 tests)

**Note:** Tests use in-memory SQLite and don't require Docker or PostgreSQL to be running.

## Environment Variables

- `API_KEY`: API key for authentication (default: `dev-key`)
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `DATABASE_URL`: PostgreSQL connection string (default: local PostgreSQL)

**Note:** When using Docker Compose, `DATABASE_URL` is automatically configured.

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

Database migrations are managed with Alembic and run automatically on Docker Compose startup.

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
