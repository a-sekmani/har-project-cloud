# ONNX model directory (Phase 6)

Place exported HAR-WindowNet models here. Each model lives in a subdirectory named by `model_key`.

## Layout

```
models/
  <model_key>/           e.g. custom10_tcn_v1_vel
    model.onnx           Exported ONNX model
    label_map.json       id -> label string (order = class index)
    model_meta.json       version, input shape, etc.
```

## Example: custom10_tcn_v1_vel

- `model_key`: `custom10_tcn_v1_vel`
- `label_map.json`: `{"0": "class_a", "1": "class_b", ...}` or `["class_a", "class_b", ...]`
- `model_meta.json`: `{"model_version": "v1", "features": "vel", "input_window_size": 30, "input_kpts": 17}`

Set `MODELS_DIR` (or use default `./models`) and `MODEL_KEY_DEFAULT` in config so the app finds the model at runtime.
