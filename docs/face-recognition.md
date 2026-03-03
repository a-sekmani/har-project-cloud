# Face Recognition

The service provides person management and face image storage for use with edge devices. Edge devices can sync a face gallery (persons and embeddings) for local recognition and send identified person data with ingested windows.

## Overview

- **Persons:** Create, list, get, update, and delete persons. Each person can have multiple face images.
- **Faces:** Upload face images per person. The service runs face detection and embedding (InsightFace) and stores the image and a 512-dimensional embedding. Duplicate uploads by filename for the same person are skipped; the response includes a `skipped` count.
- **Face gallery:** Edge devices fetch the full gallery via `GET /v1/face-gallery` (all active persons with their embeddings) and can poll `GET /v1/face-gallery/version` to know when to refresh.

## Storage

Face images are stored under `data/person_faces/` and served by the API at `/face-images/`. Embeddings and metadata are stored in the database (see [Database](database.md)).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/persons` | POST, GET | Create a person or list persons |
| `/v1/persons/{person_id}` | GET, PATCH, DELETE | Get, update, or delete a person |
| `/v1/persons/{person_id}/faces` | GET, POST | List faces or upload a face for a person |
| `/v1/persons/{person_id}/faces/{face_id}` | DELETE | Delete a specific face |
| `/v1/face-gallery` | GET | Full face gallery for edge (persons + embeddings) |
| `/v1/face-gallery/version` | GET | Current gallery version for cache invalidation |

Full request/response schemas, headers, and examples are in the [API Reference](API.md) (Persons API, Faces API, Face Gallery API).

## Web Pages

- **/persons** — List persons; add, edit, delete.
- **/persons/{id}** — Person details; upload and manage face images.

## Linking Windows to Persons

When ingesting a window, the edge can send an optional **person** object (person_id, person_name, person_conf, gallery_version). The cloud stores this with the window so the Recent Windows page and API can show who was identified. The person_id must refer to an existing person in the database; otherwise the ingest returns 400. See [Edge Integration](edge-integration.md) and [API Reference – Windows Ingest](API.md#post-v1windowsingest).
