#!/usr/bin/env python3
"""
Seed pose_windows from sample JSON/JSONL files (e.g. HAR-WindowNet export or data_out/custom10/samples).

Usage (from project root):
  python scripts/seed_windows.py --from labelled.jsonl
  python scripts/seed_windows.py --from path/to/samples/
  python scripts/seed_windows.py --from path/to/samples/ --limit 20

Then: curl -s "http://localhost:8000/v1/windows?limit=5" -H "X-API-Key: dev-key"
      Open http://localhost:8000/dashboard
"""
import argparse
import json
import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models import PoseWindow


def load_samples(path: Path, limit: int) -> list[dict]:
    """Load sample objects from path (file or directory). Returns list of dicts with device_id, camera_id, track_id, ts_start_ms, ts_end_ms, fps, window_size, label."""
    samples = []
    if path.is_file():
        if path.suffix == ".jsonl" or path.name.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        samples.append(obj)
                        if len(samples) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                samples.extend(data[:limit])
            else:
                samples.append(data)
    else:
        for fpath in sorted(path.glob("*.json"))[:limit * 2]:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        samples.append(item)
                        if len(samples) >= limit:
                            break
                else:
                    samples.append(data)
                    if len(samples) >= limit:
                        break
            except (json.JSONDecodeError, OSError):
                continue
            if len(samples) >= limit:
                break
        for fpath in sorted(path.glob("*.jsonl"))[:limit * 2]:
            if len(samples) >= limit:
                break
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                        if len(samples) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
    return samples[:limit]


def sample_to_window(s: dict) -> dict:
    """Map sample dict to PoseWindow kwargs; include keypoints as JSON string when present."""
    fps = s.get("fps", 30)
    window_size = s.get("window_size", 30)
    out = {
        "device_id": s.get("device_id", "seed"),
        "camera_id": s.get("camera_id", "cam-1"),
        "track_id": int(s.get("track_id", 0)),
        "ts_start_ms": int(s.get("ts_start_ms", 0)),
        "ts_end_ms": int(s.get("ts_end_ms", 0)),
        "fps": int(fps) if isinstance(fps, (int, float)) else 30,
        "window_size": int(window_size) if isinstance(window_size, (int, float)) else 30,
        "label": s.get("label"),
    }
    if "keypoints" in s and s["keypoints"] is not None:
        out["keypoints_json"] = json.dumps(s["keypoints"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed pose_windows from sample JSON/JSONL")
    parser.add_argument("--from", dest="from_path", required=True, help="Path to .json, .jsonl file or directory of samples")
    parser.add_argument("--limit", type=int, default=50, help="Max number of windows to insert (default 50)")
    args = parser.parse_args()
    path = Path(args.from_path)
    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    samples = load_samples(path, args.limit)
    if not samples:
        print("No samples found.")
        return
    db = SessionLocal()
    try:
        created = 0
        for s in samples:
            try:
                kw = sample_to_window(s)
                w = PoseWindow(**kw)
                db.add(w)
                created += 1
            except Exception as e:
                print(f"Skip sample: {e}", file=sys.stderr)
        db.commit()
        print(f"Inserted {created} pose windows. Open http://localhost:8000/dashboard")
    finally:
        db.close()


if __name__ == "__main__":
    main()
