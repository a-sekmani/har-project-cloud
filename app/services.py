"""Business logic: windows, predictions, dashboard."""
import json
from collections import Counter
from datetime import datetime, UTC, timedelta
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.constants import ALERT_ACTIVITIES
from app.models import PoseWindow, WindowPrediction, Person, PersonFace, AlertStatus
from app.schemas import IngestWindowBody
from app.utils import isoformat_utc


def get_windows(db: Session, limit: int = 100):
    return db.query(PoseWindow).order_by(desc(PoseWindow.created_at)).limit(limit).all()


def get_window_by_id(db: Session, window_id: UUID) -> PoseWindow | None:
    return db.query(PoseWindow).filter(PoseWindow.id == window_id).first()


def update_window_label(db: Session, window_id: UUID, label: str) -> PoseWindow | None:
    w = get_window_by_id(db, window_id)
    if w is None:
        return None
    w.label = label
    db.commit()
    db.refresh(w)
    return w


def set_window_person(db: Session, window_id: UUID, person_id: UUID | None) -> PoseWindow | None:
    """Set or clear the identified person for a window. person_id=None clears."""
    w = get_window_by_id(db, window_id)
    if w is None:
        return None
    if person_id is None:
        w.person_id = None
        w.person_name = None
        w.person_conf = None
        w.gallery_version = None
    else:
        person = db.query(Person).filter(Person.id == person_id).first()
        if person is None:
            return None
        w.person_id = person.id
        w.person_name = person.name
        w.person_conf = 1.0  # manual assignment
        w.gallery_version = None
    db.commit()
    db.refresh(w)
    return w


def get_recent_windows_with_predictions(db: Session, limit: int = 100, model_key: str | None = None) -> list[dict]:
    windows = db.query(PoseWindow).order_by(desc(PoseWindow.created_at)).limit(limit).all()
    out = []
    for w in windows:
        row = {
            "id": str(w.id),
            "device_id": w.device_id,
            "camera_id": w.camera_id,
            "track_id": w.track_id,
            "ts_start_ms": w.ts_start_ms,
            "ts_end_ms": w.ts_end_ms,
            "fps": w.fps,
            "window_size": w.window_size,
            "label": w.label,
            "created_at": isoformat_utc(w.created_at),
            "prediction": None,
        }
        if model_key:
            pred = (
                db.query(WindowPrediction)
                .filter(WindowPrediction.window_id == w.id, WindowPrediction.model_key == model_key)
                .order_by(desc(WindowPrediction.created_at))
                .limit(1)
                .first()
            )
            if pred:
                row["prediction"] = {"pred_label": pred.pred_label, "pred_conf": pred.pred_conf, "model_key": pred.model_key}
        out.append(row)
    return out


def get_dashboard_windows(
    db: Session,
    model_key: str | None,
    limit: int = 100,
    offset: int = 0,
    device_id: str | None = None,
    camera_id: str | None = None,
    track_id: int | None = None,
    person_id: UUID | None = None,
    only_with_predictions: bool = False,
    pred_label: str | None = None,
    max_pred_conf: float | None = None,
    only_unlabeled: bool = False,
    only_labeled: bool = False,
    only_mismatches: bool = False,
    only_unknown_person: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    min_person_conf: float | None = None,
    max_person_conf: float | None = None,
    only_alerts: bool = False,
) -> dict:
    """Windows for dashboard with optional filters; each row has latest prediction per model_key. Returns {data, has_more}."""
    from app.models_meta import get_labels_and_version

    query = db.query(PoseWindow).order_by(desc(PoseWindow.created_at))
    if device_id is not None and device_id != "":
        query = query.filter(PoseWindow.device_id == device_id)
    if camera_id is not None and camera_id != "":
        query = query.filter(PoseWindow.camera_id == camera_id)
    if track_id is not None:
        query = query.filter(PoseWindow.track_id == track_id)
    if person_id is not None:
        query = query.filter(PoseWindow.person_id == person_id)
    if only_unknown_person:
        query = query.filter(PoseWindow.person_id.is_(None))
    if since is not None:
        query = query.filter(PoseWindow.created_at >= since)
    if until is not None:
        query = query.filter(PoseWindow.created_at <= until)
    if min_person_conf is not None:
        query = query.filter(PoseWindow.person_conf >= min_person_conf)
    if max_person_conf is not None:
        query = query.filter(PoseWindow.person_conf <= max_person_conf)
    fetch_limit = limit * 4 if (
        only_with_predictions or pred_label is not None or max_pred_conf is not None
        or only_unlabeled or only_labeled or only_mismatches
    ) else limit
    windows = query.offset(offset).limit(fetch_limit).all()

    model_version = None
    if model_key:
        try:
            _, model_version = get_labels_and_version(model_key)
        except FileNotFoundError:
            pass

    # Single bulk query: latest prediction per (window_id, model_key) — ORDER BY created_at DESC, then take first per window_id
    latest_pred_by_window: dict[UUID, WindowPrediction] = {}
    if model_key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == model_key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p

    out = []
    for w in windows:
        row = {
            "id": str(w.id),
            "created_at": isoformat_utc(w.created_at),
            "device_id": w.device_id,
            "camera_id": w.camera_id,
            "track_id": w.track_id,
            "ts_start_ms": w.ts_start_ms,
            "ts_end_ms": w.ts_end_ms,
            "fps": w.fps,
            "window_size": w.window_size,
            "label": w.label,
            "prediction": None,
            # Person identification fields
            "person_id": str(w.person_id) if w.person_id else None,
            "person_name": w.person_name,
            "person_conf": w.person_conf,
            "gallery_version": w.gallery_version,
        }
        pred = latest_pred_by_window.get(w.id) if model_key else None
        if pred:
            row["prediction"] = {
                "model_key": pred.model_key,
                "model_version": model_version,
                "features": None,
                "pred_label_id": None,
                "pred_label": pred.pred_label,
                "pred_conf": pred.pred_conf,
                "created_at": isoformat_utc(pred.created_at),
            }
        if only_with_predictions and row["prediction"] is None:
            continue
        if pred_label is not None and (row["prediction"] is None or row["prediction"].get("pred_label") != pred_label):
            continue
        if max_pred_conf is not None and (row["prediction"] is None or row["prediction"].get("pred_conf", 1) > max_pred_conf):
            continue
        label_val = (row.get("label") or "").strip()
        if only_unlabeled and label_val != "":
            continue
        if only_labeled and label_val == "":
            continue
        if only_mismatches:
            pred_l = (row["prediction"] or {}).get("pred_label")
            if not label_val or not pred_l or label_val == pred_l:
                continue
        if only_alerts and (row["prediction"] is None or row["prediction"].get("pred_label") not in ALERT_ACTIVITIES):
            continue
        out.append(row)
        if len(out) >= limit:
            break
    has_more = len(windows) == fetch_limit
    return {"data": out, "has_more": has_more}


def create_window_prediction(
    db: Session,
    window_id: UUID,
    model_key: str,
    pred_label: str,
    pred_conf: float,
) -> WindowPrediction:
    """Insert a new window prediction row (keeps history)."""
    pred = WindowPrediction(
        window_id=window_id,
        model_key=model_key,
        pred_label=pred_label,
        pred_conf=pred_conf,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def create_pose_window_from_ingest(db: Session, body: IngestWindowBody) -> PoseWindow:
    """Build and persist a PoseWindow from ingest body; keypoints stored as JSON string."""
    window_id = body.id if body.id is not None else uuid4()
    if body.created_at is not None:
        # Normalize to UTC: if edge sent naive datetime, treat as UTC for correct display
        created = body.created_at if body.created_at.tzinfo is not None else body.created_at.replace(tzinfo=UTC)
    else:
        created = datetime.now(UTC)
    
    # Handle person identification from edge
    person_id = None
    person_name = None
    person_conf = None
    gallery_version = None
    
    if body.person is not None:
        person_conf = body.person.person_conf
        gallery_version = body.person.gallery_version
        person_name = body.person.person_name
        
        if body.person.person_id is not None:
            # Validate that person exists in database
            person = db.query(Person).filter(Person.id == body.person.person_id).first()
            if person is None:
                raise ValueError(f"Person not found: {body.person.person_id}")
            person_id = body.person.person_id
            # Use current name from DB if not provided
            if person_name is None:
                person_name = person.name
    
    w = PoseWindow(
        id=window_id,
        device_id=body.device_id,
        camera_id=body.camera_id,
        track_id=body.track_id,
        ts_start_ms=body.ts_start_ms,
        ts_end_ms=body.ts_end_ms,
        fps=int(body.fps) if isinstance(body.fps, float) else body.fps,
        window_size=body.window_size,
        label=body.label,
        keypoints_json=json.dumps(body.keypoints),
        created_at=created,
        person_id=person_id,
        person_name=person_name,
        person_conf=person_conf,
        gallery_version=gallery_version,
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return w


def run_predict_for_window(
    db: Session,
    window_id: UUID,
    model_key: str,
    store: bool = True,
    return_probs: bool = False,
) -> dict:
    """
    Run ONNX prediction for a stored window. Returns dict with pred_label, pred_conf, model_key,
    and optionally probs. Raises FileNotFoundError if model missing (503), ValueError on feature/inference error (400).
    """
    from app.ml.features import keypoints_to_model_input
    from app.ml.onnx_runner import run_onnx_predict

    w = get_window_by_id(db, window_id)
    if w is None:
        raise ValueError("Window not found")
    if not w.keypoints_json:
        raise ValueError("Window has no keypoints")
    keypoints = json.loads(w.keypoints_json)
    inp = keypoints_to_model_input(keypoints, window_size=w.window_size or 30)
    pred_label, pred_conf, all_probs = run_onnx_predict(model_key, inp)
    if store:
        create_window_prediction(db, window_id, model_key, pred_label, pred_conf)
    out = {"pred_label": pred_label, "pred_conf": pred_conf, "model_key": model_key}
    if return_probs:
        out["probs"] = [{"label": l, "score": s} for l, s in all_probs]
    return out


def get_dashboard_overview(
    db: Session,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    model_key: str | None = None,
    person_id: UUID | None = None,
    camera_id: str | None = None,
    device_id: str | None = None,
    pred_label: str | None = None,
    only_alerts: bool = False,
    only_unknown_person: bool = False,
    only_known_person: bool = False,
    max_windows: int = 5000,
) -> dict:
    """
    Dashboard overview: stats, activity distribution, timeline, person presence, recent important events.
    Filters on PoseWindow.created_at (since/until) and optional person_id, camera_id, device_id;
    prediction filters (pred_label, only_alerts) apply when model_key is set.
    """
    query = db.query(PoseWindow).order_by(desc(PoseWindow.created_at))
    if since is not None:
        query = query.filter(PoseWindow.created_at >= since)
    if until is not None:
        query = query.filter(PoseWindow.created_at <= until)
    if person_id is not None:
        query = query.filter(PoseWindow.person_id == person_id)
    if camera_id is not None and camera_id != "":
        query = query.filter(PoseWindow.camera_id == camera_id)
    if device_id is not None and device_id != "":
        query = query.filter(PoseWindow.device_id == device_id)
    if only_unknown_person:
        query = query.filter(PoseWindow.person_id.is_(None))
    if only_known_person:
        query = query.filter(PoseWindow.person_id.isnot(None))
    windows = query.limit(max_windows).all()

    latest_pred_by_window: dict[UUID, WindowPrediction] = {}
    if model_key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == model_key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p

    # Rows: (window, prediction or None)
    rows: list[tuple[PoseWindow, WindowPrediction | None]] = []
    for w in windows:
        pred = latest_pred_by_window.get(w.id) if model_key else None
        if model_key and pred is None:
            continue
        if pred_label is not None and (pred is None or pred.pred_label != pred_label):
            continue
        if only_alerts and (pred is None or pred.pred_label not in ALERT_ACTIVITIES):
            continue
        rows.append((w, pred))

    # Stats
    person_ids_seen = {w.person_id for w, _ in rows if w.person_id is not None}
    unknown_count = sum(1 for w, _ in rows if w.person_id is None)
    pred_labels_seen = {p.pred_label for _, p in rows if p is not None}
    alert_count = sum(1 for _, p in rows if p is not None and p.pred_label in ALERT_ACTIVITIES)
    last_created = max((w.created_at for w, _ in rows), default=None)

    stats = {
        "total_windows": len(rows),
        "recognized_persons": len(person_ids_seen),
        "unknown_person_windows": unknown_count,
        "detected_activities": len(pred_labels_seen),
        "fall_alerts": alert_count,
        "last_update": isoformat_utc(last_created) if last_created else None,
    }

    # Activity distribution
    dist_counter: Counter[str] = Counter()
    for _, p in rows:
        if p is not None:
            dist_counter[p.pred_label] += 1
    activity_distribution = [{"label": label, "count": count} for label, count in dist_counter.most_common()]

    # Activity timeline: by day when range > 24h (week/month), else by hour
    activity_timeline: list[dict] = []
    timeline_by_day = False
    if rows and since is not None and until is not None:
        range_seconds = (until - since).total_seconds()
        bucket_by_day = range_seconds > 86400  # > 24h
        timeline_by_day = bucket_by_day
        buckets: dict[datetime, int] = {}
        for w, _ in rows:
            t = w.created_at
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            if bucket_by_day:
                bucket_ts = t.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket_ts = t.replace(minute=0, second=0, microsecond=0)
            buckets[bucket_ts] = buckets.get(bucket_ts, 0) + 1
        for bucket_ts in sorted(buckets.keys()):
            activity_timeline.append({
                "time": isoformat_utc(bucket_ts),
                "count": buckets[bucket_ts],
            })

    # Person presence: (last_seen, label, person_name) per person_id
    presence_by_person: dict[UUID | None, list[tuple[datetime, str, str]]] = {}
    for w, p in rows:
        pid = w.person_id
        name = (w.person_name or "Unknown").strip() or "Unknown"
        if pid is None:
            name = "Unknown"
        t = w.created_at
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        label = p.pred_label if p is not None else ""
        if pid not in presence_by_person:
            presence_by_person[pid] = []
        presence_by_person[pid].append((t, label, name))

    person_presence = []
    for pid, entries in presence_by_person.items():
        last_seen = max(e[0] for e in entries)
        labels = [e[1] for e in entries if e[1]]
        top_activity = Counter(labels).most_common(1)[0][0] if labels else None
        person_name = entries[0][2] if entries else "Unknown"
        person_presence.append({
            "person_id": str(pid) if pid else None,
            "person_name": person_name,
            "last_seen": isoformat_utc(last_seen),
            "window_count": len(entries),
            "top_activity": top_activity,
        })
    person_presence.sort(key=lambda x: x["last_seen"] or "", reverse=True)

    # Recent important events (alerts only)
    recent_important_events = []
    for w, p in rows:
        if p is None or p.pred_label not in ALERT_ACTIVITIES:
            continue
        recent_important_events.append({
            "window_id": str(w.id),
            "time": isoformat_utc(w.created_at),
            "person_name": (w.person_name or "Unknown").strip() or "Unknown",
            "activity": p.pred_label,
            "confidence": p.pred_conf,
        })
    recent_important_events.sort(key=lambda x: x["time"], reverse=True)
    recent_important_events = recent_important_events[:50]

    return {
        "stats": stats,
        "activity_distribution": activity_distribution,
        "activity_timeline": activity_timeline,
        "timeline_by_day": timeline_by_day,
        "person_presence": person_presence,
        "recent_important_events": recent_important_events,
    }


def get_unknown_persons_overview(
    db: Session,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    model_key: str | None = None,
    max_windows: int = 5000,
) -> dict:
    """
    Overview for Unknown Persons page: stats, timeline, activity distribution.
    Only windows with person_id IS NULL. model_key used for latest prediction per window.
    """
    from app.config import MODEL_KEY_DEFAULT
    key = model_key or MODEL_KEY_DEFAULT

    query = (
        db.query(PoseWindow)
        .filter(PoseWindow.person_id.is_(None))
        .order_by(desc(PoseWindow.created_at))
    )
    if since is not None:
        query = query.filter(PoseWindow.created_at >= since)
    if until is not None:
        query = query.filter(PoseWindow.created_at <= until)
    windows = query.limit(max_windows).all()

    latest_pred_by_window: dict[UUID, WindowPrediction] = {}
    if key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p

    rows: list[tuple[PoseWindow, WindowPrediction | None]] = []
    for w in windows:
        pred = latest_pred_by_window.get(w.id) if key else None
        if key and pred is None:
            continue
        rows.append((w, pred))

    total_unknown_windows = len(rows)
    track_keys = set((w.camera_id or "", w.track_id) for w, _ in rows)
    unknown_tracks = len(track_keys)

    start_today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    rows_today = []
    for w, p in rows:
        t = w.created_at
        if t is None:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        if t >= start_today:
            rows_today.append((w, p))
    track_keys_today = set((w.camera_id or "", w.track_id) for w, _ in rows_today)
    unknown_tracks_today = len(track_keys_today)

    activity_counter: Counter[str] = Counter()
    for _, p in rows:
        if p is not None:
            activity_counter[p.pred_label] += 1
    most_common_activity = activity_counter.most_common(1)[0][0] if activity_counter else None
    activity_distribution = [{"label": label, "count": count} for label, count in activity_counter.most_common()]

    camera_counts: Counter[str] = Counter()
    for w, _ in rows:
        cam = w.camera_id or ""
        if cam:
            camera_counts[cam] += 1
    cameras_with_unknowns = sorted(camera_counts.keys())
    camera_with_most_unknowns = camera_counts.most_common(1)[0][0] if camera_counts else None

    timeline: list[dict] = []
    if rows and since is not None and until is not None:
        range_seconds = (until - since).total_seconds()
        bucket_by_day = range_seconds > 86400
        buckets: dict[datetime, int] = {}
        for w, _ in rows:
            t = w.created_at
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            if bucket_by_day:
                bucket_ts = t.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket_ts = t.replace(minute=0, second=0, microsecond=0)
            buckets[bucket_ts] = buckets.get(bucket_ts, 0) + 1
        for bucket_ts in sorted(buckets.keys()):
            timeline.append({"time": isoformat_utc(bucket_ts), "count": buckets[bucket_ts]})

    stats = {
        "total_unknown_windows": total_unknown_windows,
        "unknown_tracks": unknown_tracks,
        "unknown_tracks_today": unknown_tracks_today,
        "most_common_activity": most_common_activity,
        "cameras_with_unknowns": cameras_with_unknowns,
        "camera_with_most_unknowns": camera_with_most_unknowns,
    }
    return {
        "stats": stats,
        "activity_distribution": activity_distribution,
        "timeline": timeline,
    }


def get_dashboard_filter_options(
    db: Session,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict:
    """Return distinct device_id and camera_id from PoseWindow for filter dropdowns.
    Returns all distinct values in the table (since/until ignored) so dropdowns always show available options."""
    dev_rows = db.query(PoseWindow.device_id).filter(
        PoseWindow.device_id.isnot(None),
        PoseWindow.device_id != "",
    ).distinct().all()
    cam_rows = db.query(PoseWindow.camera_id).filter(
        PoseWindow.camera_id.isnot(None),
        PoseWindow.camera_id != "",
    ).distinct().all()
    devices = sorted({r[0] for r in dev_rows if r[0]})
    cameras = sorted({r[0] for r in cam_rows if r[0]})
    return {"devices": devices, "cameras": cameras}


def get_alerts(
    db: Session,
    *,
    model_key: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    status_filter: str | None = None,
) -> list[dict]:
    """
    Return alert events: windows with prediction in ALERT_ACTIVITIES.
    Each item: time, person, event, confidence, camera, status (new/acknowledged/resolved), window_id.
    """
    from app.config import MODEL_KEY_DEFAULT
    key = model_key or MODEL_KEY_DEFAULT
    preds = (
        db.query(WindowPrediction)
        .filter(
            WindowPrediction.model_key == key,
            WindowPrediction.pred_label.in_(list(ALERT_ACTIVITIES)),
        )
        .order_by(desc(WindowPrediction.created_at))
        .limit(limit * 3)
        .all()
    )
    if not preds:
        return []
    seen_w: set[UUID] = set()
    preds_by_window: list[WindowPrediction] = []
    for p in preds:
        if p.window_id in seen_w:
            continue
        seen_w.add(p.window_id)
        preds_by_window.append(p)
        if len(preds_by_window) >= limit:
            break
    window_ids = [p.window_id for p in preds_by_window]
    windows = {w.id: w for w in db.query(PoseWindow).filter(PoseWindow.id.in_(window_ids)).all()}
    status_map: dict[UUID, str] = {}
    for row in db.query(AlertStatus).filter(AlertStatus.window_id.in_(window_ids)).all():
        status_map[row.window_id] = row.status
    out = []
    for p in preds_by_window:
        w = windows.get(p.window_id)
        if w is None:
            continue
        if since is not None and w.created_at < since:
            continue
        if until is not None and w.created_at > until:
            continue
        st = status_map.get(w.id, "new")
        if status_filter is not None and st != status_filter:
            continue
        out.append({
            "window_id": str(w.id),
            "time": isoformat_utc(w.created_at),
            "person": (w.person_name or "Unknown").strip() or "Unknown",
            "event": p.pred_label,
            "confidence": p.pred_conf,
            "camera": w.camera_id or "—",
            "status": st,
        })
    return out


def set_alert_status(db: Session, window_id: UUID, status: str) -> AlertStatus | None:
    """Set status for an alert (window). status: new, acknowledged, resolved."""
    w = get_window_by_id(db, window_id)
    if w is None:
        return None
    if status not in ("new", "acknowledged", "resolved"):
        return None
    row = db.query(AlertStatus).filter(AlertStatus.window_id == window_id).first()
    now = datetime.now(UTC)
    if row is None:
        row = AlertStatus(window_id=window_id, status=status, updated_at=now)
        db.add(row)
    else:
        row.status = status
        row.updated_at = now
    db.commit()
    db.refresh(row)
    return row


def get_person_window_stats(
    db: Session,
    person_id: UUID,
    model_key: str | None = None,
) -> dict:
    """Return first_seen, last_seen, total_windows, main_activity for a person from PoseWindow + latest predictions."""
    windows = (
        db.query(PoseWindow)
        .filter(PoseWindow.person_id == person_id)
        .order_by(desc(PoseWindow.created_at))
        .all()
    )
    if not windows:
        return {
            "first_seen": None,
            "last_seen": None,
            "total_windows": 0,
            "main_activity": None,
        }
    created_times = [w.created_at for w in windows]
    first_seen = min(created_times)
    last_seen = max(created_times)
    latest_pred_by_window: dict = {}
    if model_key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == model_key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p.pred_label
    labels = [latest_pred_by_window.get(w.id) for w in windows if latest_pred_by_window.get(w.id)]
    main_activity = Counter(labels).most_common(1)[0][0] if labels else None
    return {
        "first_seen": isoformat_utc(first_seen) if first_seen else None,
        "last_seen": isoformat_utc(last_seen) if last_seen else None,
        "total_windows": len(windows),
        "main_activity": main_activity,
    }


def get_person_detail(
    db: Session,
    person_id: UUID,
    model_key: str | None = None,
    recent_windows_limit: int = 50,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict | None:
    """Return full person detail: base info, stats, activity_distribution, activity_timeline, recent_windows.
    Optional since/until filter windows by created_at for stats and activity data."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if person is None:
        return None
    face_count = db.query(PersonFace).filter(PersonFace.person_id == person_id).count()

    query = (
        db.query(PoseWindow)
        .filter(PoseWindow.person_id == person_id)
        .order_by(desc(PoseWindow.created_at))
    )
    if since is not None:
        query = query.filter(PoseWindow.created_at >= since)
    if until is not None:
        query = query.filter(PoseWindow.created_at <= until)
    windows = query.limit(max(recent_windows_limit * 2, 500)).all()
    latest_pred_by_window = {}
    if model_key and windows:
        window_ids = [w.id for w in windows]
        preds = (
            db.query(WindowPrediction)
            .filter(
                WindowPrediction.model_key == model_key,
                WindowPrediction.window_id.in_(window_ids),
            )
            .order_by(desc(WindowPrediction.created_at))
            .all()
        )
        for p in preds:
            if p.window_id not in latest_pred_by_window:
                latest_pred_by_window[p.window_id] = p

    rows = [(w, latest_pred_by_window.get(w.id)) for w in windows]
    created_times = [w.created_at for w in windows]
    first_seen = min(created_times) if created_times else None
    last_seen = max(created_times) if created_times else None
    dist_counter = Counter()
    for w, pred in rows:
        if pred is not None:
            dist_counter[pred.pred_label] += 1
    main_activity = dist_counter.most_common(1)[0][0] if dist_counter else None
    activity_distribution = [{"label": label, "count": c} for label, c in dist_counter.most_common()]

    timeline_buckets = {}
    for w in windows:
        t = w.created_at
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        bucket = t.replace(hour=0, minute=0, second=0, microsecond=0)
        timeline_buckets[bucket] = timeline_buckets.get(bucket, 0) + 1
    activity_timeline = [{"time": isoformat_utc(k), "count": timeline_buckets[k]} for k in sorted(timeline_buckets.keys())]

    recent_windows = []
    for w, pred in rows[:recent_windows_limit]:
        recent_windows.append({
            "id": str(w.id),
            "created_at": isoformat_utc(w.created_at),
            "device_id": w.device_id,
            "camera_id": w.camera_id,
            "pred_label": pred.pred_label if pred else None,
            "pred_conf": pred.pred_conf if pred else None,
        })

    return {
        "id": str(person.id),
        "name": person.name,
        "external_ref": person.external_ref,
        "is_active": person.is_active,
        "created_at": isoformat_utc(person.created_at),
        "face_count": face_count,
        "first_seen": isoformat_utc(first_seen) if first_seen else None,
        "last_seen": isoformat_utc(last_seen) if last_seen else None,
        "total_windows": len(windows),
        "main_activity": main_activity,
        "activity_distribution": activity_distribution,
        "activity_timeline": activity_timeline,
        "recent_windows": recent_windows,
    }
