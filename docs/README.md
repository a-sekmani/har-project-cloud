# Cloud HAR Documentation

This folder contains the full documentation for the Cloud HAR (Human Activity Recognition) API service. The web interface is branded **HAR Cloud App** and offers a consistent layout and navigation across all pages.

## Web UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Overview | `/dashboard` | Stats, activity distribution, timeline, person presence, recent important events (alerts). |
| Recent Windows | `/windows` | List of pose windows with predictions; filters, pagination, link to labeling. |
| Label Windows | `/windows/label` | Label windows for training; filter by model, device, camera, activity. |
| Person Management | `/persons` | Create, edit, delete persons; view face count and stats. |
| Person Detail | `/persons/{person_id}` | Single person: faces, activity stats, timeline, recent windows. |
| Unknown Persons | `/unknown-persons` | Windows with no identified person; stats, charts, assign to person, create person from window, alerts. |
| Alerts / Critical Events | `/alerts` | Windows with alert-type predictions (e.g. falling_down, chest_pain); acknowledge/resolve status. |
| Models / System | `/system` | System status: current model, face gallery version, edge status, health. |

## Documentation Index

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Requirements, Docker, Docker Compose, local development, and ONNX model setup |
| [Configuration](configuration.md) | Environment variables and server settings |
| [Database](database.md) | Schema, tables, and migrations |
| [API Reference](API.md) | Complete API documentation: endpoints, request/response formats, status codes |
| [Edge Integration](edge-integration.md) | Window ingest from edge devices, date/time format, and troubleshooting |
| [Face Recognition](face-recognition.md) | Person management, face upload, and face gallery for edge sync |
| [Development](development.md) | Project structure, scripts, testing, and logging |

## Quick Links

- **Get started:** [Installation](installation.md) and [Quick Start](installation.md#quick-start)
- **Web UI:** All pages use the same header; start at Overview (`/dashboard`) or Recent Windows (`/windows`) once the server is running.
- **Integrate from edge:** [Edge Integration](edge-integration.md) and [API Reference – Windows Ingest](API.md#post-v1windowsingest)
- **Face gallery for edge:** [Face Recognition](face-recognition.md) and [API Reference – Face Gallery](API.md#face-gallery-api)
