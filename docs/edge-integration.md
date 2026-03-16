# Edge Integration

This document describes how edge devices send pose windows to the cloud and how to troubleshoot common issues.

## Window Ingest

Edge devices send full pose windows via `POST /v1/windows/ingest`. The request body follows the HAR-WindowNet-style contract: device_id, camera_id, track_id, timestamps, fps, window_size, and keypoints. Optionally, the body can include a **person** object (person_id, person_name, person_conf, gallery_version) when the edge has identified a person via face recognition.

Full request/response details and query parameters are in the [API Reference – Windows Ingest](API.md#post-v1windowsingest).

## Date and Time Format (from Edge)

Each window is stored with a single timestamp `created_at` (date and time). The Activity Windows page displays both **Date** and **Time** columns. To ensure correct storage and display, the edge should send `created_at` as an **ISO 8601 string with timezone**.

### Recommended formats

| Format | Example |
|--------|---------|
| UTC | `"2026-02-24T11:32:05.123Z"` or `"2026-02-24T11:32:05+00:00"` |
| Local with offset | `"2026-02-24T14:32:05.123+03:00"` |

- **Date:** `YYYY-MM-DD`
- **Time:** `HH:mm:ss` or `HH:mm:ss.sss`
- **Timezone:** Use `Z` or `+00:00` for UTC, or an explicit offset (e.g. `+03:00`, `-05:00`).

Do not send a string without timezone (e.g. `"2026-02-24T14:32:05"`); the cloud will treat it as UTC and display may be incorrect. To use server receive time, omit `created_at` entirely.

**Python (edge) example:**

```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
created_at = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
```

## Logging

When the edge sends ingest requests, the server logs:

- **ingest request:** device_id, camera_id, track_id, window_size, whether person data is present
- **ingest ok:** window id, person_id, pred_label (if prediction ran)
- **ingest rejected:** when the request is rejected (e.g. person not found)
- **ingest duplicate window id:** when the same window id is sent twice
- **ingest validation failed:** path and validation errors (e.g. keypoints shape)

Use `LOG_LEVEL=DEBUG` for more detail.

## Troubleshooting

When monitoring the cloud while the edge sends windows, check server logs. Common causes of failure:

| HTTP | Cause | Action |
|------|--------|--------|
| **422** | Request body validation failed | Check log: `ingest validation failed: path=... errors=...`. Fix missing/invalid fields or types. |
| **422 (keypoints)** | Keypoints must be `[window_size]` frames × 17 keypoints × `[x, y, conf]`, each value in **[0, 1]** | Normalize keypoints on the edge (x, y by image dimensions; confidence in 0–1). Do not send pixel coordinates or values outside [0, 1]. |
| **400** | `Person not found: <uuid>` | Edge sent `person.person_id` that does not exist in the cloud. Sync the face gallery from cloud to edge, or omit `person_id` when unknown. |
| **400** | `ts_end_ms must be greater than ts_start_ms` | Ensure end timestamp is greater than start. |
| **409** | Window with this id already exists | Edge sent the same `id` twice. Omit `id` in the body to let the cloud generate a new UUID, or ensure unique IDs per window. |
| **401** | Invalid API key | Send the correct `X-API-Key` header (e.g. `dev-key` or the server’s `API_KEY` value). |
| **503** | Model not found | Install the requested (or default) model under `models/<model_key>/`, or use `predict=false` for ingest-only. |
