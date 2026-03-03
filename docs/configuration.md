# Configuration

Configuration is done via environment variables. Defaults are defined in `app/config.py`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for all protected endpoints | `dev-key` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level (e.g. DEBUG, INFO, WARNING) | `INFO` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://cloudhar:cloudhar@localhost:5433/cloudhar` |
| `MODELS_DIR` | Directory for ONNX models | `models` |
| `MODEL_KEY_DEFAULT` | Default model key for dashboard and ingest prediction | `edge17_v6_lowlr` |

With Docker Compose, `DATABASE_URL` is typically set automatically to the postgres service.

## Authentication

All API endpoints except `GET /health` require the `X-API-Key` header. Use the same value as `API_KEY` (e.g. `dev-key` by default).
