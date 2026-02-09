"""Lightweight model metadata: list models and get labels from label_map.json (no ONNX load)."""
from pathlib import Path
from typing import Optional
import json

from app.config import MODELS_DIR


def _base() -> Path:
    return Path(MODELS_DIR).resolve()


def list_available() -> list[str]:
    """Return model_key list (subdirs with label_map.json)."""
    base = _base()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "label_map.json").is_file())


def get_labels_and_version(model_key: str) -> tuple[list[str], Optional[str]]:
    """Read label_map.json for model_key. Returns (labels in index order, version or None)."""
    base = _base()
    model_dir = base / model_key
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    path = model_dir / "label_map.json"
    if not path.is_file():
        raise FileNotFoundError(f"label_map.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_name = data.get("id_to_name") or data.get("label_names")
    if isinstance(id_to_name, dict):
        keys = sorted(id_to_name.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        labels = [id_to_name[k] for k in keys]
    else:
        labels = list(id_to_name) if id_to_name else []
    version = None
    meta_path = model_dir / "model_meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        version = meta.get("model_version") or meta.get("training_dataset")
    return labels, version
