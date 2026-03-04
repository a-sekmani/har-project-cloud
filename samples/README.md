# Sample request payloads

This folder contains sample JSON bodies for manual testing and scripts.

| File | Description |
|------|-------------|
| `test_request.json` | Inference request payload for `POST /v1/activity/infer`. Used by `test_scenarios.sh` (Multiple Devices scenario). |
| `test_request_valid.json` | Valid inference payload (e.g. for manual curl or tools). |
| `test_request_low_conf.json` | Inference payload with low confidence data (for testing API behaviour). |

All files follow the inference request schema: `schema_version`, `device_id`, `camera_id`, `window`, `people` (with `track_id`, `keypoints`, optional `pose_conf`).

Run from project root, for example:

```bash
curl -s -X POST "http://localhost:8000/v1/activity/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d @samples/test_request.json
```

Or use the scenario script: `./test_scenarios.sh`.
