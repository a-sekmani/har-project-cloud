"""
Temporal features extracted from a keypoints window [T][K][3].

- motion_energy: mean per-frame displacement of hips+shoulders (limb/torso jitter).
- center_drift: displacement of body center from first to last frame (actual travel).
  Body center = centroid of (left_shoulder, right_shoulder, left_hip, right_hip).
  Locomotion should depend on center_drift, not motion_energy, to avoid standing
  when the person is walking but limbs jitter or drop out.

Coordinates normalized to [0,1] when max(x,y) > 1 so thresholds are in one unit.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# COCO-17: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12 (body center = centroid of these)
BODY_CENTER_INDICES = (5, 6, 11, 12)
# For per-frame limb displacement (micro_motion vs still)
MOTION_KEYPOINT_INDICES = (5, 6, 11, 12)
CONF_MIN_FOR_MOTION = 0.1
CONF_MISSING_THRESHOLD = 0.1


@dataclass
class WindowFeatures:
    """Numeric features from a keypoints window (T frames, K keypoints per frame)."""
    missing_ratio: float
    mean_pose_conf: float
    motion_energy: float  # mean per-frame displacement (limbs/torso); can be noisy
    center_drift: float  # distance(center_first, center_last) — body displacement over window
    window_duration_s: float  # (T-1)/fps for calibration and avg_center_speed
    avg_center_speed: float  # center_drift / window_duration_s (for calibration)


def _body_center_for_frame(frame: List[List[float]], indices: Tuple[int, ...] = BODY_CENTER_INDICES) -> Optional[Tuple[float, float]]:
    """Center = centroid of hips+shoulders; only points with c >= CONF_MIN_FOR_MOTION. Returns (cx, cy) or None."""
    K = len(frame)
    xs, ys, n = 0.0, 0.0, 0
    for i in indices:
        if i >= K or len(frame[i]) < 3:
            continue
        c = frame[i][2]
        if c >= CONF_MIN_FOR_MOTION and math.isfinite(c):
            xs += frame[i][0]
            ys += frame[i][1]
            n += 1
    if n == 0:
        return None
    return (xs / n, ys / n)


def _normalize_xy_to_unit(keypoints: List[List[List[float]]]) -> List[List[List[float]]]:
    """If any x or y > 1, scale all x,y so max is 1. Preserves confidence (index 2)."""
    if not keypoints:
        return keypoints
    flat_xy = []
    for frame in keypoints:
        for kp in frame:
            if len(kp) >= 2:
                flat_xy.extend([kp[0], kp[1]])
    max_xy = max(flat_xy) if flat_xy else 1.0
    if max_xy <= 1.0:
        return keypoints
    scale = max_xy
    out = []
    for frame in keypoints:
        row = []
        for kp in frame:
            if len(kp) >= 3:
                row.append([kp[0] / scale, kp[1] / scale, kp[2]])
            else:
                row.append(list(kp))
        out.append(row)
    return out


def extract_window_features(
    keypoints_tk3: List[List[List[float]]],
    fps: int,
) -> WindowFeatures:
    """
    Extract temporal features from keypoints [T][K][3] (x, y, confidence).

    Coordinates: If max(x,y) > 1 they are normalized to [0,1] so motion_energy
    is in the same unit as used by inference thresholds (TH_STILL, TH_MOVE).
    When keypoints are already in [0,1] (e.g. from aggregation), motion_energy
    is typically in range ~0.001--0.02 for still, higher for moving.

    Args:
        keypoints_tk3: [T][K][3] with K >= 12 (we use indices 5,6,11,12).
        fps: Frames per second (for future use; not used in motion_energy here).

    Returns:
        WindowFeatures with missing_ratio, mean_pose_conf, motion_energy.
    """
    if not keypoints_tk3:
        return WindowFeatures(
            missing_ratio=1.0,
            mean_pose_conf=0.0,
            motion_energy=0.0,
            center_drift=0.0,
            window_duration_s=0.0,
            avg_center_speed=0.0,
        )

    keypoints_tk3 = _normalize_xy_to_unit(keypoints_tk3)
    T = len(keypoints_tk3)
    K = len(keypoints_tk3[0]) if keypoints_tk3 else 0

    conf_sum = 0.0
    conf_count = 0
    missing_count = 0
    total_kp = 0

    for frame in keypoints_tk3:
        for kp in frame:
            if len(kp) < 3:
                missing_count += 1
                total_kp += 1
                continue
            c = float(kp[2])
            if not math.isfinite(c):
                missing_count += 1
            elif c < CONF_MISSING_THRESHOLD:
                missing_count += 1
            else:
                conf_count += 1
            conf_sum += c if math.isfinite(c) else 0.0
            total_kp += 1

    # mean_pose_conf = average c over all keypoints (same as services)
    mean_pose_conf = conf_sum / total_kp if total_kp else 0.0
    missing_ratio = missing_count / total_kp if total_kp else 1.0

    # motion_energy: average displacement per frame for MOTION_KEYPOINT_INDICES
    motion_sum = 0.0
    motion_count = 0
    for idx in MOTION_KEYPOINT_INDICES:
        if idx >= K:
            continue
        for t in range(1, T):
            prev = keypoints_tk3[t - 1][idx]
            curr = keypoints_tk3[t][idx]
            if len(prev) < 3 or len(curr) < 3:
                continue
            c_prev = prev[2]
            c_curr = curr[2]
            if c_prev < CONF_MIN_FOR_MOTION or c_curr < CONF_MIN_FOR_MOTION:
                continue
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            motion_sum += math.sqrt(dx * dx + dy * dy)
            motion_count += 1

    motion_energy = motion_sum / motion_count if motion_count else 0.0

    # Body center drift (first frame → last frame) for locomotion
    center_first = _body_center_for_frame(keypoints_tk3[0]) if T else None
    center_last = _body_center_for_frame(keypoints_tk3[T - 1]) if T else None
    center_drift = 0.0
    if center_first and center_last:
        dx = center_last[0] - center_first[0]
        dy = center_last[1] - center_first[1]
        center_drift = math.sqrt(dx * dx + dy * dy)
    window_duration_s = (T - 1) / fps if fps > 0 and T > 1 else 0.0
    avg_center_speed = center_drift / window_duration_s if window_duration_s > 0 else 0.0

    return WindowFeatures(
        missing_ratio=round(missing_ratio, 4),
        mean_pose_conf=round(mean_pose_conf, 4),
        motion_energy=round(motion_energy, 6),
        center_drift=round(center_drift, 6),
        window_duration_s=round(window_duration_s, 4),
        avg_center_speed=round(avg_center_speed, 6),
    )
