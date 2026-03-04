# Cloud HAR Documentation

This folder contains the full documentation for the Cloud HAR (Human Activity Recognition) API service. The web interface is branded **HAR Cloud App** and offers a consistent layout and navigation across Dashboard, Recent Windows, Label Windows, and Person Management.

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
- **Integrate from edge:** [Edge Integration](edge-integration.md) and [API Reference – Windows Ingest](API.md#post-v1windowsingest)
- **Face gallery for edge:** [Face Recognition](face-recognition.md) and [API Reference – Face Gallery](API.md#face-gallery-api)
