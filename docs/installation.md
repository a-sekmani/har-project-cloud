# Installation and Running

## Requirements

- Python 3.11+
- Docker and Docker Compose (optional; for PostgreSQL or containerized deployment)
- PostgreSQL (optional; the app can run with SQLite for local development)

## Using Docker

Build the image:

```bash
docker build -t cloud-har .
```

Run the container (requires a running PostgreSQL instance; use Docker Compose for the full stack):

```bash
docker run -p 8000:8000 -e DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db cloud-har
```

## Using Docker Compose

The provided `docker-compose.yml` runs **PostgreSQL** only (for use when running the API locally):

```bash
# Start PostgreSQL (port 5433)
docker-compose up -d

# Run migrations and the API on your host (see Local Development)
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Stop and remove volumes (deletes database data)
docker-compose down -v
```

- **postgres**: PostgreSQL 16 (exposed on host port 5433; default `DATABASE_URL` uses `localhost:5433`).

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Option A: Use PostgreSQL (e.g. docker-compose up -d postgres)
# Default DATABASE_URL uses localhost:5433 (Docker Compose maps postgres to 5433).
# If your Postgres runs on port 5432, set:
#   export DATABASE_URL="postgresql+psycopg://cloudhar:cloudhar@localhost:5432/cloudhar"
# Then run migrations:
#   alembic upgrade head

# Option B: Use SQLite (no Docker required)
#   export DATABASE_URL="sqlite:///./cloudhar.db"
# Tables are created automatically on startup; no alembic needed.

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running with ONNX Models

Place an exported HAR-WindowNet-style model under the `models/` directory. The default path is `./models` (override with `MODELS_DIR`). Each model lives in a subdirectory named by `model_key` (e.g. `models/edge17_v6_lowlr/`) with:

- **model.onnx** — Exported ONNX model (input shape e.g. `[1, 30, 85]`)
- **label_map.json** — Mapping from class index to label name (e.g. `id_to_name`)
- **model_meta.json** — Metadata (e.g. `input_shape`, `window_size`, `input_features`, `feature_spec`)

Set `MODEL_KEY_DEFAULT` (e.g. `edge17_v6_lowlr`) for the dashboard default model.

**Dependencies:** `numpy` and `onnxruntime` are in `pyproject.toml`. Install with `pip install -e .`.

**Feature extraction pipeline** (must match training):

1. **Center on hips:** Subtract hips midpoint (keypoints 11, 12) from all keypoints
2. **Scale by shoulders:** Divide by shoulder width (distance between keypoints 5, 6)
3. **Clamp:** Clip x, y values to [-3, 3]
4. **Velocity:** Compute frame-to-frame difference (first frame = 0)
5. **Feature layout:** 85 features per frame = 51 pose [x, y, conf × 17] + 34 velocity [dx, dy × 17]

## Quick Start

1. **Database:** Either `docker-compose up -d` then `alembic upgrade head` (PostgreSQL), or `export DATABASE_URL="sqlite:///./cloudhar.db"` (SQLite; no migrations).
2. **Run API:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. **Health check:** `curl http://localhost:8000/health`
4. **Seed windows (with keypoints):** `python scripts/seed_windows.py --from labelled.jsonl --limit 20`  
   Or ingest one window: `curl -s -X POST "http://localhost:8000/v1/windows/ingest" -H "X-API-Key: dev-key" -H "Content-Type: application/json" --data-binary @window_sample.json`
5. **List windows:** `curl -s "http://localhost:8000/v1/windows?limit=5" -H "X-API-Key: dev-key"`
6. **Run ONNX predict** (replace `WINDOW_ID`):  
   `curl -s -X POST "http://localhost:8000/v1/windows/WINDOW_ID/predict" -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d '{"model_key":"edge17_v6_lowlr","store":true}'`
7. **Recent Windows UI:** Open `http://localhost:8000/windows`
8. **Label page:** `http://localhost:8000/windows/label`
