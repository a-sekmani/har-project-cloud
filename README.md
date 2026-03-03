# Cloud HAR - Human Activity Recognition API

Cloud service for human activity recognition from skeleton (pose) data. It provides window-level inference, pose window storage, ONNX-based activity prediction, a web dashboard for recent windows and labeling, window ingest from edge devices (with optional person identification), and face recognition (person management and face gallery for edge sync).

**Requirements:** Python 3.11+, Docker and Docker Compose, PostgreSQL.

## Documentation

Full documentation is in the **[docs/](docs/)** folder:

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [Installation](docs/installation.md) | Docker, local setup, ONNX models, quick start |
| [Configuration](docs/configuration.md) | Environment variables |
| [Database](docs/database.md) | Schema and migrations |
| [API Reference](docs/API.md) | All endpoints, request/response formats |
| [Edge Integration](docs/edge-integration.md) | Ingest, date/time format, troubleshooting |
| [Face Recognition](docs/face-recognition.md) | Persons, faces, face gallery |
| [Development](docs/development.md) | Project structure, scripts, testing, logging |

## Quick Start

1. **Start services:** `docker-compose up -d`
2. **Migrations:** `alembic upgrade head`
3. **Run API:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. **Health:** `curl http://localhost:8000/health`
5. **UI:** [http://localhost:8000/windows](http://localhost:8000/windows) (Recent Windows), [http://localhost:8000/persons](http://localhost:8000/persons) (Persons)

For detailed steps, examples, and options see [docs/installation.md](docs/installation.md).

## API Overview

All endpoints except `GET /health` require the `X-API-Key` header (default: `dev-key`).

| Area | Endpoints |
|------|-----------|
| Health | `GET /health` |
| Models | `GET /v1/models`, `GET /v1/models/{model_key}` |
| Windows | `GET /v1/windows`, `GET /v1/windows/{id}`, `POST /v1/windows/ingest`, `POST /v1/windows/{id}/label`, `POST /v1/windows/{id}/predict` |
| Dashboard | `GET /v1/dashboard/windows` |
| Persons & faces | `POST/GET /v1/persons`, `GET/PATCH/DELETE /v1/persons/{id}`, `GET/POST /v1/persons/{id}/faces`, `DELETE /v1/persons/{id}/faces/{face_id}` |
| Face gallery | `GET /v1/face-gallery`, `GET /v1/face-gallery/version` |
| Legacy | `POST /v1/activity/infer` (mock) |

See [docs/API.md](docs/API.md) for full details.
