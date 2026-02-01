"""
Feature-based activity inference from window features.

Quality gate: mean_pose_conf < 0.25 or missing_ratio > 0.4 → unknown.
Above the gate, pose_conf is used only to scale confidence (not to block classification).

Locomotion (moving) depends on center_drift (body center first→last frame), not motion_energy
(limb jitter), so walking is detected even when limbs vibrate or drop out.
Still + micro_motion → standing; locomotion → moving.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.features import WindowFeatures
from app.schemas import TopKItemSchema

# Quality gate
MEAN_POSE_CONF_MIN = 0.25
MISSING_RATIO_MAX = 0.4

# Locomotion: moving when center_drift > TH_CENTER_DRIFT (body displacement over window, normalized [0,1])
# Calibrate from data: median(center_drift) over 30 standing windows vs 30 walking windows; set between.
TH_CENTER_DRIFT = 0.02

# Micro_motion vs still (per-frame limb displacement): for motion_type only; standing = still + micro_motion
TH_STILL = 0.005
TH_MOVE = 0.02

# Confidence bounds
CONF_MIN = 0.3
CONF_MAX = 0.9
UNKNOWN_CONFIDENCE = 0.2
# Scale confidence by pose when above gate: [MEAN_POSE_CONF_MIN, 1.0] → [0.5, 1.0]
POSE_CONF_SCALE_LOW = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scale_confidence_by_pose(confidence: float, mean_pose_conf: float) -> float:
    """Use pose_conf only to reduce confidence; do not block classification above gate."""
    if mean_pose_conf >= 1.0:
        return confidence
    t = (mean_pose_conf - MEAN_POSE_CONF_MIN) / (1.0 - MEAN_POSE_CONF_MIN) if MEAN_POSE_CONF_MIN < 1.0 else 1.0
    t = max(0.0, min(1.0, t))
    scale = POSE_CONF_SCALE_LOW + (1.0 - POSE_CONF_SCALE_LOW) * t
    return _clamp(confidence * scale, CONF_MIN, CONF_MAX)


def infer_activity(
    features: WindowFeatures,
    quality: Tuple[int, float, float],
) -> Tuple[str, float, List[TopKItemSchema], Dict[str, Any]]:
    """
    Infer activity from window features and quality metrics.

    Args:
        features: WindowFeatures (missing_ratio, mean_pose_conf, motion_energy).
        quality: (k_count, avg_pose_conf, frames_ok_ratio) from compute_input_quality.

    Returns:
        (activity, confidence, top_k, debug_dict).
        debug_dict contains: features (motion_energy, missing_ratio, mean_pose_conf),
        thresholds (TH_STILL, TH_MOVE).
    """
    _k_count, avg_pose_conf, frames_ok_ratio = quality
    missing_ratio = features.missing_ratio
    mean_pose_conf = features.mean_pose_conf
    motion_energy = features.motion_energy
    center_drift = features.center_drift
    window_duration_s = features.window_duration_s

    debug_base = {
        "features": {
            "motion_energy": features.motion_energy,
            "center_drift": features.center_drift,
            "window_duration_s": features.window_duration_s,
            "avg_center_speed": features.avg_center_speed,
            "missing_ratio": features.missing_ratio,
            "mean_pose_conf": features.mean_pose_conf,
        },
        "thresholds": {"TH_CENTER_DRIFT": TH_CENTER_DRIFT, "TH_STILL": TH_STILL, "TH_MOVE": TH_MOVE},
    }

    # Quality gate: only unknown when quality is very poor
    if mean_pose_conf < MEAN_POSE_CONF_MIN or missing_ratio > MISSING_RATIO_MAX:
        activity = "unknown"
        confidence = UNKNOWN_CONFIDENCE
        motion_type = None
        top_k = [
            TopKItemSchema(label="unknown", score=UNKNOWN_CONFIDENCE),
            TopKItemSchema(label="standing", score=0.1),
            TopKItemSchema(label="moving", score=0.1),
        ]
        return activity, confidence, top_k, debug_base

    # Locomotion = moving when body center drifts (center_drift), not limb jitter (motion_energy)
    if center_drift > TH_CENTER_DRIFT:
        motion_type = "locomotion"
        activity = "moving"
        confidence = _clamp(
            center_drift / TH_CENTER_DRIFT if TH_CENTER_DRIFT > 0 else 0.6,
            CONF_MIN,
            CONF_MAX,
        )
    else:
        # Still or micro_motion → standing
        if motion_energy < TH_STILL:
            motion_type = "still"
        else:
            motion_type = "micro_motion"
        activity = "standing"
        confidence = _clamp(
            1.0 - center_drift / TH_CENTER_DRIFT if TH_CENTER_DRIFT > 0 else 0.9,
            CONF_MIN,
            CONF_MAX,
        )

    # Use pose_conf only to reduce confidence (not to block)
    confidence = _scale_confidence_by_pose(confidence, mean_pose_conf)

    if activity == "standing":
        top_k = [
            TopKItemSchema(label="standing", score=round(confidence, 2)),
            TopKItemSchema(label="moving", score=round(1.0 - confidence, 2)),
            TopKItemSchema(label="unknown", score=0.1),
        ]
    else:
        top_k = [
            TopKItemSchema(label="moving", score=round(confidence, 2)),
            TopKItemSchema(label="standing", score=round(1.0 - confidence, 2)),
            TopKItemSchema(label="unknown", score=0.1),
        ]

    debug = {**debug_base, "motion_type": motion_type}
    return activity, round(confidence, 4), top_k, debug
