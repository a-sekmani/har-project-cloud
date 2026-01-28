# Cloud HAR - Human Activity Recognition API

Cloud service for receiving Skeleton Windows from edge devices and returning mock activity recognition results.

## Requirements

- Python 3.11+
- Docker (required)

## Project Structure

```
cloud-har/
  app/
    __init__.py
    main.py          # FastAPI application
    schemas.py       # Pydantic models
    config.py        # Configuration
    logging.py       # Logging setup
    health.py        # Health check endpoint
  tests/
    test_health.py
    test_infer_schema.py
  Dockerfile
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

### Using docker-compose

```bash
docker-compose up
```

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /v1/activity/infer

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

## Testing

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_health.py
pytest tests/test_infer_schema.py
```

## Environment Variables

- `API_KEY`: API key for authentication (default: `dev-key`)
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Logging

Each inference request is logged with:
- `request_id` (UUID)
- `device_id`
- `camera_id`
- `num_people`
- `latency_ms`

**Note:** Full keypoints are not logged to avoid log bloat.
