# Cloud HAR API Reference

Complete API documentation for the Cloud HAR (Human Activity Recognition) service.

**Base URL:** `http://localhost:8000`

**Authentication:** All endpoints (except `/health`) require the `X-API-Key` header.

```bash
curl -H "X-API-Key: dev-key" http://localhost:8000/v1/windows
```

---

## Table of Contents

- [Health](#health)
- [Models API](#models-api)
- [Persons API](#persons-api)
- [Faces API](#faces-api)
- [Face Gallery API](#face-gallery-api)
- [Windows API](#windows-api)
- [Activity Inference API](#activity-inference-api)
- [Dashboard API](#dashboard-api)
- [Unknown Persons API](#unknown-persons-api)
- [Alerts API](#alerts-api)
- [System API](#system-api)
- [Error Responses](#error-responses)

---

## Health

### GET /health

Health check endpoint for monitoring and load balancers. No authentication required.

**Response:**
```json
{
  "status": "ok"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

## Models API

### GET /v1/models

List all available models.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Response:**
```json
{
  "models": ["c_norm_vel", "edge17_full", "edge17_v6_lowlr"]
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Missing or invalid API key

---

### GET /v1/models/{model_key}

Get metadata for a specific model.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| model_key | string | Model identifier (e.g., `edge17_v6_lowlr`) |

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Response:**
```json
{
  "model_key": "edge17_v6_lowlr",
  "model_version": "ntu120",
  "labels": ["drink water", "eat meal", "stand up", "sit down", "reading", "falling down", "headache", "chest pain", "back pain", "nausea/vomiting"]
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Model not found
- `503 Service Unavailable` - Model files missing

---

## Persons API

### POST /v1/persons

Create a new person for face recognition.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

**Request Body:**
```json
{
  "name": "Ahmad",
  "external_ref": "optional-external-id",
  "is_active": true
}
```
`external_ref` is optional; `is_active` defaults to `true`.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Ahmad",
  "external_ref": "optional-external-id",
  "is_active": true,
  "created_at": "2026-02-24T10:30:00.000000",
  "face_count": 0
}
```

---

### GET /v1/persons

List all persons.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| is_active | boolean | - | Filter by active status |
| limit | integer | 100 | Maximum results (1-500) |
| offset | integer | 0 | Skip first N results |
| include_stats | boolean | false | When true, each person includes `last_seen`, `total_windows`, `main_activity` (requires model_key or default model) |
| model_key | string | - | Model key for stats/predictions (used when include_stats=true) |

**Response:**
```json
{
  "persons": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Ahmad",
      "external_ref": null,
      "is_active": true,
      "created_at": "2026-02-24T10:30:00.000000",
      "face_count": 3,
      "last_seen": "2026-03-04T12:00:00Z",
      "total_windows": 42,
      "main_activity": "reading"
    }
  ],
  "total": 1
}
```
`last_seen`, `total_windows`, and `main_activity` are present only when `include_stats=true`.

---

### GET /v1/persons/{person_id}

Get a single person by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| person_id | UUID | Person identifier |

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Ahmad",
  "external_ref": null,
  "is_active": true,
  "created_at": "2026-02-24T10:30:00.000000",
  "face_count": 3
}
```

---

### GET /v1/persons/{person_id}/detail

Get full person detail: base info, stats, activity distribution, activity timeline, and recent windows. Use this for the Person Detail page.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| model_key | string | Model key for predictions (optional; default from config) |

**Response:** Single object with: `id`, `name`, `external_ref`, `is_active`, `created_at`, `face_count`, `first_seen`, `last_seen`, `total_windows`, `main_activity`, `activity_distribution` (array of `{ label, count }`), `activity_timeline` (array of `{ time, count }`), `recent_windows` (array of window objects with `id`, `created_at`, `device_id`, `camera_id`, `pred_label`, `pred_conf`).

---

### PATCH /v1/persons/{person_id}

Update a person's details.

**Request Body:**
```json
{
  "name": "Ahmad Updated",
  "external_ref": "new-ref",
  "is_active": false
}
```

All fields are optional. Changes to `is_active` will update the gallery version.

---

### DELETE /v1/persons/{person_id}

Delete a person and all their face images.

**Response:**
```json
{
  "deleted": true,
  "face_images_deleted": 3
}
```

---

## Faces API

### POST /v1/persons/{person_id}/faces

Upload face images for a person. Each image is processed to detect face and extract embedding.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `multipart/form-data` |

**Request:**
- `files`: One or more image files (JPEG, PNG, WebP)

**Response:**
```json
{
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "added": 3,
  "failed": 1,
  "errors": ["image4.jpg: No face detected"],
  "gallery_version": "v12"
}
```

**Validation:**
- Each image must contain at least one face
- If multiple faces detected, the one with highest detection score is used
- Images without detectable faces are rejected with error message

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/persons/{person_id}/faces" \
  -H "X-API-Key: dev-key" \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg"
```

---

### GET /v1/persons/{person_id}/faces

List all face images for a person.

**Response:**
```json
{
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "faces": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "person_id": "550e8400-e29b-41d4-a716-446655440000",
      "image_path": "550e8400.../660e8400....jpg",
      "det_score": 0.95,
      "quality_score": null,
      "created_at": "2026-02-24T10:35:00.000000"
    }
  ],
  "total": 1
}
```

---

### DELETE /v1/persons/{person_id}/faces/{face_id}

Delete a specific face image.

**Response:**
```json
{
  "deleted": true
}
```

---

## Face Gallery API

### GET /v1/face-gallery

Get face gallery for edge devices. Returns all active persons with their face embeddings.

**Response:**
```json
{
  "gallery_version": "v12",
  "embedding_dim": 512,
  "threshold": 0.45,
  "people": [
    {
      "person_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Ahmad",
      "embeddings": [
        [0.123, -0.456, ...],
        [0.234, -0.567, ...]
      ]
    }
  ]
}
```

**Notes:**
- Only returns persons with `is_active=true`
- Only includes persons with at least one face embedding
- `threshold` is the recommended cosine similarity threshold for matching
- Edge devices should cache this and refresh when `gallery_version` changes

---

### GET /v1/face-gallery/version

Get current gallery version for cache invalidation.

**Response:**
```json
{
  "version": "v12",
  "created_at": "2026-02-24T10:35:00.000000"
}
```

---

## Windows API

### GET /v1/windows

List pose windows (metadata only, keypoints not included).

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 100 | Maximum number of windows to return (1-500) |

**Response:**
```json
{
  "windows": [
    {
      "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
      "device_id": "projecthost",
      "camera_id": "default",
      "track_id": 1,
      "ts_start_ms": 1772385813818,
      "ts_end_ms": 1772385814782,
      "fps": 30,
      "window_size": 30,
      "label": null,
      "created_at": "2026-03-01T17:23:34.781775"
    }
  ]
}
```

**Example:**
```bash
curl -s "http://localhost:8000/v1/windows?limit=10" -H "X-API-Key: dev-key"
```

---

### GET /v1/windows/{window_id}

Get a single pose window by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| window_id | UUID | Window identifier |

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Response:**
```json
{
  "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
  "device_id": "projecthost",
  "camera_id": "default",
  "track_id": 1,
  "ts_start_ms": 1772385813818,
  "ts_end_ms": 1772385814782,
  "fps": 30,
  "window_size": 30,
  "label": null,
  "created_at": "2026-03-01T17:23:34.781775"
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Window not found

**Example:**
```bash
curl -s "http://localhost:8000/v1/windows/807dbb7e-61a4-4353-98ab-86da002cb291" \
  -H "X-API-Key: dev-key"
```

---

### POST /v1/windows/{window_id}/person

Assign or clear the identified person for a window. Used by the Unknown Persons page and person assignment flows.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| window_id | UUID | Window identifier |

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

**Request Body:**
```json
{
  "person_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
Send `person_id: null` to clear the person (set window to unknown).

**Response:**
```json
{
  "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "person_name": "Ahmad"
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid body
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Window or person not found

---

### POST /v1/windows/{window_id}/label

Set or update the label for a pose window.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| window_id | UUID | Window identifier |

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

**Request Body:**
```json
{
  "label": "drink water",
  "label_source": "manual"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| label | string | Yes | Activity label (empty string to clear) |
| label_source | string | No | Source of the label (default: `manual`) |

**Response:**
```json
{
  "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
  "label": "drink water"
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Window not found

**Example:**
```bash
curl -s -X POST "http://localhost:8000/v1/windows/807dbb7e-61a4-4353-98ab-86da002cb291/label" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"label": "drink water", "label_source": "manual"}'
```

---

### POST /v1/windows/{window_id}/predict

Run ONNX model prediction on a stored window.

**Prerequisites:** The window must have `keypoints_json` stored (e.g., ingested via `/v1/windows/ingest` or seeded from JSONL).

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| window_id | UUID | Window identifier |

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

**Request Body:**
```json
{
  "model_key": "edge17_v6_lowlr",
  "store": true,
  "return_probs": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| model_key | string | Yes | - | Model to use for prediction |
| store | boolean | No | true | Save prediction to database |
| return_probs | boolean | No | false | Include full class probabilities |

**Response:**
```json
{
  "pred_label": "drink water",
  "pred_conf": 0.92,
  "model_key": "edge17_v6_lowlr"
}
```

**Response with `return_probs: true`:**
```json
{
  "pred_label": "drink water",
  "pred_conf": 0.92,
  "model_key": "edge17_v6_lowlr",
  "probs": [
    {"label": "drink water", "score": 0.92},
    {"label": "eat meal", "score": 0.05},
    {"label": "reading", "score": 0.02}
  ]
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Window has no keypoints or feature extraction failed
- `401 Unauthorized` - Missing or invalid API key
- `404 Not Found` - Window or model not found
- `503 Service Unavailable` - Model files missing

**Example:**
```bash
curl -s -X POST "http://localhost:8000/v1/windows/807dbb7e-61a4-4353-98ab-86da002cb291/predict" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "edge17_v6_lowlr", "store": true, "return_probs": true}'
```

---

### POST /v1/windows/ingest

Ingest a full window from the edge device. Prediction runs automatically using the default model.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| predict | boolean | true | Run prediction after ingest |
| model_key | string | MODEL_KEY_DEFAULT | Model to use for prediction |
| store_prediction | boolean | true | Save prediction to database |
| return_probs | boolean | false | Include full class probabilities |

**Request Body:**
```json
{
  "device_id": "pi-001",
  "camera_id": "cam-1",
  "track_id": 1,
  "ts_start_ms": 1772385813818,
  "ts_end_ms": 1772385814782,
  "fps": 30,
  "window_size": 30,
  "keypoints": [
    [[0.52, 0.18, 0.91], [0.50, 0.22, 0.88], ...],
    ...
  ],
  "person": {
    "person_id": "550e8400-e29b-41d4-a716-446655440000",
    "person_name": "Ahmad",
    "person_conf": 0.85,
    "gallery_version": "v12"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| device_id | string | Yes | Device identifier |
| camera_id | string | Yes | Camera identifier |
| track_id | integer | Yes | Person tracking ID (≥0) |
| ts_start_ms | integer | Yes | Window start timestamp (ms) |
| ts_end_ms | integer | Yes | Window end timestamp (ms) |
| fps | integer | Yes | Frames per second (1-120) |
| window_size | integer | Yes | Number of frames (10-120) |
| keypoints | array | Yes | 3D array [frames][keypoints][x,y,conf] |
| id | UUID | No | Custom window ID (auto-generated if omitted) |
| session_id | string | No | Session identifier |
| mean_pose_conf | float | No | Mean pose confidence |
| label | string | No | Pre-assigned label |
| label_source | string | No | Label source |
| created_at | string | No | Date and time when the window was created; see **Date and time format (from edge)** below. |
| person | object | No | Person identification from edge face recognition |

**Date and time format (from edge):**

Each window is stored with a single timestamp `created_at` (date + time). The Recent Windows page shows both **Date** and **Time** columns. To have correct date and time stored and displayed, the edge should send `created_at` as an ISO 8601 string **with timezone**.

| Format | Example | Notes |
|--------|---------|-------|
| **UTC (recommended)** | `"2026-02-24T11:32:05.123Z"` or `"2026-02-24T11:32:05+00:00"` | Date = `YYYY-MM-DD`, time = `HH:mm:ss.sss`, suffix `Z` or `+00:00`. |
| **Local with offset** | `"2026-02-24T14:32:05.123+03:00"` | Same date/time pattern with explicit offset (e.g. `+03:00`). |

- **Do not** send a string without timezone (e.g. `"2026-02-24T14:32:05"`); the cloud will assume UTC and display may be wrong.
- **Simplest:** Omit `created_at`; the cloud will use the request receive time (UTC).
- **Python (edge):** `dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"` or use `isoformat()` on a timezone-aware datetime.

See [Edge Integration](edge-integration.md) for full details on date/time format and troubleshooting.

**Person Object (optional):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| person_id | UUID | No | Identified person ID (null if unknown) |
| person_name | string | No | Person name at time of identification |
| person_conf | float | Yes | Face recognition confidence (0-1) |
| gallery_version | string | No | Gallery version used for recognition |

**Keypoints Format:**
- Array of `window_size` frames
- Each frame has 17 keypoints (COCO-17 format)
- Each keypoint is `[x, y, confidence]` with values in [0, 1]

**Response:**
```json
{
  "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
  "device_id": "pi-001",
  "camera_id": "cam-1",
  "track_id": 1,
  "ts_start_ms": 1772385813818,
  "ts_end_ms": 1772385814782,
  "fps": 30,
  "window_size": 30,
  "label": null,
  "created_at": "2026-03-01T17:23:34.781775",
  "person_id": "550e8400-e29b-41d4-a716-446655440000",
  "person_name": "Ahmad",
  "person_conf": 0.85,
  "gallery_version": "v12",
  "pred_label": "drink water",
  "pred_conf": 0.92,
  "model_key": "edge17_v6_lowlr",
  "prediction": {
    "model_key": "edge17_v6_lowlr",
    "pred_label": "drink water",
    "pred_conf": 0.92
  }
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid request body or feature extraction failed
- `401 Unauthorized` - Missing or invalid API key
- `409 Conflict` - Window with this ID already exists
- `503 Service Unavailable` - Model not found or files missing

**Example:**
```bash
# Ingest with auto-prediction
curl -s -X POST "http://localhost:8000/v1/windows/ingest" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  --data-binary @window_sample.json

# Ingest with specific model
curl -s -X POST "http://localhost:8000/v1/windows/ingest?model_key=edge17_v6_lowlr&return_probs=true" \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  --data-binary @window_sample.json
```

**Troubleshooting ingest (edge camera stream):**

When monitoring the cloud while the edge sends windows, check server logs. Common causes of failure:

| HTTP | Cause | What to do |
|------|--------|------------|
| **422** | Request body validation failed | Check log: `ingest validation failed: path=... errors=...`. Typical: missing/invalid fields, wrong types, or **keypoints** shape/values. |
| **422 (keypoints)** | Keypoints must be `[window_size]` frames × 17 keypoints × `[x, y, conf]`, each value in **[0, 1]** | Normalize keypoints on the edge: x,y by image width/height, confidence in 0–1. Do not send pixel coordinates or values outside [0,1]. |
| **400** | `Person not found: <uuid>` | Edge sent `person.person_id` that does not exist in cloud DB. Sync face gallery to edge from cloud, or omit `person_id` when unknown. |
| **400** | `ts_end_ms must be greater than ts_start_ms` | Fix timestamps so end > start. |
| **409** | Window with this id already exists | Edge sent same `id` twice. Omit `id` in body to let cloud generate a new UUID, or ensure edge uses unique IDs per window. |
| **401** | Invalid API key | Send correct `X-API-Key` (e.g. `dev-key` or value of `API_KEY` on server). |
| **503** | Model not found | Install the requested (or default) model under `models/<model_key>/` or set `predict=false` for ingest-only. |

Run the server with `LOG_LEVEL=DEBUG` for more detail, or watch for log lines: `ingest request: ...`, `ingest ok: ...`, `ingest rejected: ...`, `ingest validation failed: ...`, `ingest duplicate window id: ...`.

---

## Activity Inference API

### POST /v1/activity/infer

Mock inference endpoint for activity recognition. Returns simulated predictions based on pose confidence.

**Note:** This endpoint does not use the ONNX model. For real predictions, use `/v1/windows/ingest` or `/v1/windows/{window_id}/predict`.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |
| Content-Type | Yes | `application/json` |

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
      "keypoints": [[[0.52, 0.18, 0.91], [0.50, 0.22, 0.88], ...]],
      "pose_conf": 0.83
    }
  ]
}
```

**Validation Rules:**
- `schema_version` must be 1
- `window.size` must be 10-120
- `window.fps` must be 1-120
- `people` must not be empty
- Each person: `track_id >= 0`, `pose_conf` in [0,1]
- `keypoints` shape: `[T][K][3]` where T = window.size, K = 17 or 25

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

---

## Dashboard API

### GET /v1/dashboard/windows

Get windows with predictions for the dashboard. Supports filtering and pagination.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_key | string | - | Filter by model |
| limit | integer | 100 | Maximum results (1-500) |
| offset | integer | 0 | Skip this many windows (for pagination) |
| device_id | string | - | Filter by device |
| camera_id | string | - | Filter by camera |
| track_id | integer | - | Filter by track ID |
| person_id | UUID | - | Filter by person |
| only_with_predictions | boolean | false | Only show windows with predictions |
| pred_label | string | - | Filter by predicted label |
| max_pred_conf | float | - | Only show predictions below this confidence |
| only_unlabeled | boolean | false | Only show unlabeled windows |
| only_labeled | boolean | false | Only show labeled windows |
| only_mismatches | boolean | false | Only show label/prediction mismatches |
| only_unknown_person | boolean | false | Only windows with no identified person (person_id IS NULL) |
| since | string | - | ISO datetime (inclusive) – start of time range |
| until | string | - | ISO datetime (inclusive) – end of time range |
| min_face_conf | float | - | Min face/person confidence (0–1) |
| max_face_conf | float | - | Max face/person confidence (0–1) |
| only_alerts | boolean | false | Only windows whose prediction is in alert activities (e.g. falling_down, chest_pain) |

**Response:** Object with `data` (array of windows) and `has_more` (boolean; true if more pages exist).
```json
{
  "data": [
    {
      "id": "807dbb7e-61a4-4353-98ab-86da002cb291",
      "created_at": "2026-03-01T17:23:34.781775",
      "device_id": "projecthost",
    "camera_id": "default",
    "track_id": 1,
    "ts_start_ms": 1772385813818,
    "ts_end_ms": 1772385814782,
    "fps": 30,
    "window_size": 30,
    "label": null,
    "prediction": {
      "model_key": "edge17_v6_lowlr",
      "model_version": "ntu120",
      "pred_label": "drink water",
      "pred_conf": 0.92,
      "created_at": "2026-03-01T17:23:35.123456"
    }
  }
  ],
  "has_more": true
}
```

**Example:**
```bash
# Get windows with low confidence predictions
curl -s "http://localhost:8000/v1/dashboard/windows?model_key=edge17_v6_lowlr&max_pred_conf=0.6&limit=50" \
  -H "X-API-Key: dev-key"
```

---

### GET /v1/dashboard/overview

Dashboard overview: aggregated stats, activity distribution, timeline, person presence, and recent important events (alerts). Used by the Overview Dashboard page. All times are UTC.

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| X-API-Key | Yes | API key for authentication |

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| since | string | ISO datetime (inclusive) – start of time range |
| until | string | ISO datetime (inclusive) – end of time range |
| model_key | string | Model for predictions (optional; when set, only windows with that model’s prediction are included in activity stats) |
| person_id | UUID | Filter by person |
| camera_id | string | Filter by camera |
| device_id | string | Filter by device |
| activity | string | Filter by prediction label (`pred_label`) |
| only_alerts | boolean | Only include windows whose prediction is in the alert activities set (e.g. falling_down, chest_pain) |
| only_unknown_person | boolean | Only windows with no identified person |
| only_known_person | boolean | Only windows with an identified person |

**Response:**
```json
{
  "stats": {
    "total_windows": 42,
    "recognized_persons": 2,
    "unknown_person_windows": 5,
    "detected_activities": 4,
    "fall_alerts": 1,
    "last_update": "2026-03-04T12:00:00.000000"
  },
  "activity_distribution": [
    { "label": "drink_water", "count": 15 },
    { "label": "reading", "count": 10 }
  ],
  "activity_timeline": [
    { "time": "2026-03-04T11:00:00.000000", "count": 8 },
    { "time": "2026-03-04T12:00:00.000000", "count": 12 }
  ],
  "timeline_by_day": false,
  "person_presence": [
    {
      "person_id": "550e8400-e29b-41d4-a716-446655440000",
      "person_name": "Ahmad",
      "last_seen": "2026-03-04T12:05:00.000000",
      "window_count": 20,
      "top_activity": "reading"
    },
    {
      "person_id": null,
      "person_name": "Unknown",
      "last_seen": "2026-03-04T11:58:00.000000",
      "window_count": 5,
      "top_activity": "standing"
    }
  ],
  "recent_important_events": [
    {
      "window_id": "807dbb7e-61a4-4353-98ab-86da002cb291",
      "time": "2026-03-04T12:00:00.000000",
      "person_name": "Ahmad",
      "activity": "falling_down",
      "confidence": 0.88
    }
  ]
}
```

| Section | Description |
|---------|-------------|
| stats | Total windows in range, distinct persons, unknown-person windows, distinct activities, count of alert activities, last window time |
| activity_distribution | Count of windows per predicted label |
| activity_timeline | Time-bucketed counts for charts (by hour when range ≤24h, by day when range >24h) |
| timeline_by_day | When true, timeline buckets are by calendar day (for week/month ranges); frontend may show dates instead of times |
| person_presence | Per person (or "Unknown"): last seen, window count, most common activity |
| recent_important_events | Windows whose prediction is in the alert set (e.g. fall, chest pain); up to 50, newest first |

**Example:**
```bash
curl -s "http://localhost:8000/v1/dashboard/overview?since=2026-03-04T10:00:00Z&until=2026-03-04T12:00:00Z&model_key=edge17_v6_lowlr" \
  -H "X-API-Key: dev-key"
```

---

### GET /v1/dashboard/filter-options

Distinct devices and cameras for dashboard filter dropdowns (e.g. Overview, Recent Windows, Unknown Persons, Label Windows).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| since | string | ISO datetime (inclusive) – optional |
| until | string | ISO datetime (inclusive) – optional |

**Response:**
```json
{
  "devices": ["pi-001", "projecthost"],
  "cameras": ["cam-1", "default"]
}
```

---

## Unknown Persons API

### GET /v1/unknown-persons/overview

Overview for the Unknown Persons page: stats, timeline, and activity distribution for windows with no identified person (`person_id IS NULL`).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| since | string | ISO datetime (inclusive) – start of time range |
| until | string | ISO datetime (inclusive) – end of time range |
| model_key | string | Model for predictions (optional; default from config) |

**Response:**
```json
{
  "stats": {
    "total_unknown_windows": 42,
    "unknown_tracks": 8,
    "unknown_tracks_today": 2,
    "most_common_activity": "standing",
    "cameras_with_unknowns": ["cam-1", "default"],
    "camera_with_most_unknowns": "cam-1"
  },
  "activity_distribution": [
    { "label": "standing", "count": 20 },
    { "label": "walking", "count": 12 }
  ],
  "timeline": [
    { "time": "2026-03-04T11:00:00.000000", "count": 10 },
    { "time": "2026-03-04T12:00:00.000000", "count": 15 }
  ]
}
```

---

## Alerts API

### GET /v1/alerts

List alert events: windows whose prediction is in the alert activities set (e.g. falling_down, chest_pain, nausea_vomiting, headache, back_pain). Used by the Alerts / Critical Events page.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_key | string | - | Model for predictions |
| since | string | - | ISO datetime (inclusive) |
| until | string | - | ISO datetime (inclusive) |
| limit | integer | 100 | Maximum results (1-500) |
| status | string | - | Filter by status: `new`, `acknowledged`, `resolved` |

**Response:**
```json
{
  "alerts": [
    {
      "window_id": "807dbb7e-61a4-4353-98ab-86da002cb291",
      "time": "2026-03-04T12:00:00.000000",
      "person": "Ahmad",
      "event": "falling_down",
      "confidence": 0.88,
      "camera": "cam-1",
      "status": "new"
    }
  ]
}
```

---

### POST /v1/alerts/{window_id}/status

Set alert status for a window: acknowledged or resolved.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| window_id | UUID | Window identifier |

**Request Body:**
```json
{
  "status": "acknowledged"
}
```
`status` must be `acknowledged` or `resolved`.

**Response:**
```json
{
  "window_id": "807dbb7e-61a4-4353-98ab-86da002cb291",
  "status": "acknowledged"
}
```

---

## System API

### GET /v1/system/status

System status for the Models / System page: current activity model, face gallery version, edge status, health.

**Response:** Object with fields such as `model_key`, `gallery_version`, `edge_status`, `health` (and any other status the backend returns).

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Missing or invalid API key |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model or database unavailable |

### Authentication Error

```json
{
  "detail": "Invalid API key"
}
```

### Validation Error (422)

```json
{
  "detail": [
    {
      "loc": ["body", "window_size"],
      "msg": "ensure this value is greater than or equal to 10",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## COCO-17 Keypoint Order

The API expects keypoints in COCO-17 format:

| Index | Keypoint |
|-------|----------|
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |
