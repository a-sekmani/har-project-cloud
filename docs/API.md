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
- [Windows API](#windows-api)
- [Activity Inference API](#activity-inference-api)
- [Dashboard API](#dashboard-api)
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
  ]
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
| created_at | datetime | No | Creation timestamp (ISO format) |

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

---

## Activity Inference API

### POST /v1/activity/infer

Mock inference endpoint for activity recognition. Returns simulated predictions based on pose confidence.

**Note:** This endpoint does not use the ONNX model. For real predictions, use `/v1/windows/ingest` or `/v1/windows/{id}/predict`.

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
| device_id | string | - | Filter by device |
| camera_id | string | - | Filter by camera |
| track_id | integer | - | Filter by track ID |
| only_with_predictions | boolean | false | Only show windows with predictions |
| pred_label | string | - | Filter by predicted label |
| max_pred_conf | float | - | Only show predictions below this confidence |
| only_unlabeled | boolean | false | Only show unlabeled windows |
| only_labeled | boolean | false | Only show labeled windows |
| only_mismatches | boolean | false | Only show label/prediction mismatches |

**Response:**
```json
[
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
]
```

**Example:**
```bash
# Get windows with low confidence predictions
curl -s "http://localhost:8000/v1/dashboard/windows?model_key=edge17_v6_lowlr&max_pred_conf=0.6&limit=50" \
  -H "X-API-Key: dev-key"
```

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
