"""
Run ONNX HAR model and return predicted label index and confidence.
Expects input shape (1, window_size, 85) and label_map for id_to_name.
"""
from pathlib import Path
from typing import Any

import numpy as np

from app.config import MODELS_DIR


def run_onnx_predict(
    model_key: str,
    input_tensor: np.ndarray,
    label_map_path: Path | None = None,
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Run ONNX model and return (pred_label_name, pred_conf, list of (label, prob)).
    input_tensor: shape (1, window_size, 85), float32.
    """
    import onnxruntime as ort

    base = Path(MODELS_DIR).resolve()
    model_dir = base / model_key
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"model.onnx not found: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0]  # (1, num_classes)
    probs = _softmax(logits[0])
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])

    path = label_map_path or (model_dir / "label_map.json")
    if not path.is_file():
        return f"class_{pred_idx}", pred_conf, [(f"class_{i}", float(p)) for i, p in enumerate(probs)]

    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_name = data.get("id_to_name") or data.get("label_names") or {}
    if isinstance(id_to_name, dict):
        pred_label = id_to_name.get(str(pred_idx), f"class_{pred_idx}")
    else:
        pred_label = id_to_name[pred_idx] if pred_idx < len(id_to_name) else f"class_{pred_idx}"
    all_probs = [(id_to_name.get(str(i), f"class_{i}"), float(p)) for i, p in enumerate(probs)]
    return pred_label, pred_conf, all_probs


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
