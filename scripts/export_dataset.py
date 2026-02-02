#!/usr/bin/env python3
"""
Export pose windows from the database to a dataset file for training or analysis.

Reads PoseWindow rows (optionally filtered by label, date range, labelled_only),
writes one file: dataset_windows.jsonl (default) or dataset_windows.npz.

Usage:
  python scripts/export_dataset.py [--label standing] [--labelled-only] [--limit 1000] [--output dataset_windows.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root so app is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models import PoseWindow


def parse_args():
    p = argparse.ArgumentParser(description="Export pose windows to dataset file.")
    p.add_argument("--label", type=str, default=None, help="Filter by label (e.g. standing, moving)")
    p.add_argument("--labelled-only", action="store_true", help="Only export rows with non-null label")
    p.add_argument("--from-ts", type=int, default=None, help="Filter ts_start_ms >= value (ms)")
    p.add_argument("--to-ts", type=int, default=None, help="Filter ts_end_ms <= value (ms)")
    p.add_argument("--limit", type=int, default=10000, help="Max number of windows to export (default 10000)")
    p.add_argument("--output", "-o", type=str, default="dataset_windows.jsonl", help="Output file path")
    p.add_argument("--format", choices=("jsonl", "npz"), default="jsonl", help="Output format (default jsonl)")
    return p.parse_args()


def row_to_record(w: PoseWindow) -> dict:
    """One window as a serializable dict for export."""
    return {
        "id": str(w.id),
        "device_id": w.device_id,
        "camera_id": w.camera_id,
        "session_id": w.session_id,
        "track_id": w.track_id,
        "ts_start_ms": w.ts_start_ms,
        "ts_end_ms": w.ts_end_ms,
        "fps": w.fps,
        "window_size": w.window_size,
        "coord_space": w.coord_space,
        "keypoints": w.keypoints,
        "mean_pose_conf": w.mean_pose_conf,
        "missing_ratio": w.missing_ratio,
        "label": w.label,
        "label_source": w.label_source,
        "labeled_at": w.labeled_at.isoformat() if w.labeled_at else None,
        "created_at": w.created_at.isoformat() if w.created_at else None,
    }


def export_jsonl(db, args) -> int:
    """Write one JSON object per line. Returns count written."""
    q = db.query(PoseWindow)
    if args.label is not None:
        q = q.filter(PoseWindow.label == args.label)
    if args.labelled_only:
        q = q.filter(PoseWindow.label.isnot(None))
    if args.from_ts is not None:
        q = q.filter(PoseWindow.ts_start_ms >= args.from_ts)
    if args.to_ts is not None:
        q = q.filter(PoseWindow.ts_end_ms <= args.to_ts)
    q = q.order_by(PoseWindow.created_at.desc()).limit(args.limit)
    rows = q.all()
    count = 0
    with open(args.output, "w") as f:
        for w in rows:
            f.write(json.dumps(row_to_record(w), ensure_ascii=False) + "\n")
            count += 1
    return count


def export_npz(db, args) -> int:
    """Write NumPy .npz with arrays: ids, keypoints, labels, device_ids, etc. Returns count written."""
    try:
        import numpy as np
    except ImportError:
        print("NumPy required for --format npz. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)
    q = db.query(PoseWindow)
    if args.label is not None:
        q = q.filter(PoseWindow.label == args.label)
    if args.labelled_only:
        q = q.filter(PoseWindow.label.isnot(None))
    if args.from_ts is not None:
        q = q.filter(PoseWindow.ts_start_ms >= args.from_ts)
    if args.to_ts is not None:
        q = q.filter(PoseWindow.ts_end_ms <= args.to_ts)
    q = q.order_by(PoseWindow.created_at.desc()).limit(args.limit)
    rows = q.all()
    if not rows:
        np.savez(args.output, ids=[], keypoints=np.array([]), labels=[])
        return 0
    ids = [str(w.id) for w in rows]
    keypoints = np.array([w.keypoints for w in rows], dtype=object)
    labels = np.array([w.label or "" for w in rows], dtype=object)
    device_ids = np.array([w.device_id for w in rows], dtype=object)
    np.savez(
        args.output,
        ids=ids,
        keypoints=keypoints,
        labels=labels,
        device_ids=device_ids,
        camera_ids=np.array([w.camera_id for w in rows], dtype=object),
        track_ids=np.array([w.track_id for w in rows]),
        ts_start_ms=np.array([w.ts_start_ms for w in rows]),
        ts_end_ms=np.array([w.ts_end_ms for w in rows]),
        mean_pose_conf=np.array([w.mean_pose_conf if w.mean_pose_conf is not None else 0.0 for w in rows]),
    )
    return len(rows)


def main():
    args = parse_args()
    db = SessionLocal()
    try:
        if args.format == "jsonl":
            count = export_jsonl(db, args)
        else:
            count = export_npz(db, args)
        print(f"Exported {count} windows to {args.output}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
