# Database

The service uses PostgreSQL for persistence. Schema is managed with Alembic migrations.

## Running Migrations

- **Local:** `alembic upgrade head`
- **Docker Compose:** Migrations run automatically on container startup

If the database is unavailable, the API may return 503 for endpoints that require it.

## Tables

### devices

Registered devices (e.g. on first inference or ingest).

### activity_events

Activity predictions: device, camera, track, window metadata, activity, confidence, and optional quality fields.

### pose_windows

Pose windows from ingest or seed scripts.

- **device_id**, **camera_id**, **track_id**: Source identifiers
- **ts_start_ms**, **ts_end_ms**: Window time range (ms)
- **fps**, **window_size**: Frame rate and number of frames
- **label**: Optional manual or pre-assigned label
- **keypoints_json**: JSON array of keypoints (frames × 17 × [x, y, conf]); stored when provided (e.g. by ingest or `seed_windows.py`)
- **created_at**: When the window was created (date and time)
- **person_id**, **person_name**, **person_conf**, **gallery_version**: Optional person identification from edge face recognition

### window_predictions

ONNX prediction results per window: **window_id**, **model_key**, **pred_label**, **pred_conf**, **created_at**.

### persons

Registered persons for face recognition: **name**, **is_active**, **created_at**.

### person_faces

Face images and embeddings per person: **person_id**, **image_path**, **original_filename**, **embedding** (JSON, 512-dim), **det_score**, **quality_score**, **created_at**.

### gallery_versions

Face gallery version history: **version** (e.g. "v1", "v2"), **created_at**. Used by edge devices to know when to refresh their local gallery cache.
