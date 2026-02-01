"""
Unit tests for extract_window_features (app/features.py).

Covers: static window → motion_energy ≈ 0, linear motion → motion_energy > 0,
missing_ratio when many keypoints have c=0.
"""
import pytest
from app.features import (
    MOTION_KEYPOINT_INDICES,
    extract_window_features,
    WindowFeatures,
)


def _make_frame(kp_list):
    """kp_list: list of [x, y, c]. Returns one frame [T=1] of keypoints."""
    return [list(kp) for kp in kp_list]


def _make_static_window(num_frames: int, num_kp: int = 17, x: float = 0.5, y: float = 0.2, c: float = 0.9):
    """Same keypoints every frame → motion_energy should be 0."""
    frame = [[x, y, c] for _ in range(num_kp)]
    return [list(frame) for _ in range(num_frames)]


def _make_linear_motion_window(num_frames: int, dx_per_frame: float, num_kp: int = 17):
    """x increases by dx_per_frame each frame for first keypoint; others fixed. motion_energy > 0."""
    window = []
    for t in range(num_frames):
        frame = []
        for i in range(num_kp):
            # Move only index 5 (left_shoulder) so motion_energy picks it up
            if i in MOTION_KEYPOINT_INDICES:
                frame.append([0.5 + t * dx_per_frame, 0.2, 0.9])
            else:
                frame.append([0.5, 0.2, 0.9])
        window.append(frame)
    return window


def test_static_window_motion_energy_zero():
    """Static window (same keypoints every frame) → motion_energy ≈ 0, center_drift = 0."""
    keypoints = _make_static_window(30, 17)
    features = extract_window_features(keypoints, fps=30)
    assert isinstance(features, WindowFeatures)
    assert features.motion_energy == 0.0
    assert features.center_drift == 0.0
    assert features.window_duration_s > 0
    assert features.mean_pose_conf == 0.9
    assert features.missing_ratio == 0.0


def test_linear_motion_motion_energy_positive():
    """Window with linear motion (x increases per frame) → motion_energy > 0."""
    keypoints = _make_linear_motion_window(30, dx_per_frame=0.01)
    features = extract_window_features(keypoints, fps=30)
    assert features.motion_energy > 0
    # 4 points * 29 frame pairs, each displacement 0.01 → mean ≈ 0.01
    assert features.motion_energy >= 0.005
    assert features.mean_pose_conf == 0.9


def test_missing_ratio_increases_with_low_confidence():
    """Many keypoints with c=0 → missing_ratio increases."""
    # All c=0.05 (below CONF_MISSING_THRESHOLD 0.1)
    keypoints = _make_static_window(10, 17, x=0.5, y=0.2, c=0.05)
    features = extract_window_features(keypoints, fps=30)
    assert features.missing_ratio == 1.0
    assert features.mean_pose_conf == 0.05

    # Half low c
    keypoints = []
    for _ in range(5):
        frame = [[0.5, 0.2, 0.9] for _ in range(9)] + [[0.5, 0.2, 0.05] for _ in range(8)]
        keypoints.append(frame)
    features = extract_window_features(keypoints, fps=30)
    assert features.missing_ratio > 0.4
    assert features.mean_pose_conf < 0.9


def test_empty_window():
    """Empty keypoints → missing_ratio=1, mean_pose_conf=0, motion_energy=0, center_drift=0."""
    features = extract_window_features([], fps=30)
    assert features.missing_ratio == 1.0
    assert features.mean_pose_conf == 0.0
    assert features.motion_energy == 0.0
    assert features.center_drift == 0.0
    assert features.window_duration_s == 0.0


def test_center_drift_body_moves():
    """Body center moves from first to last frame → center_drift > 0."""
    # Frame 0: body center at (0.5, 0.3); last frame: body center at (0.6, 0.3) → drift 0.1
    keypoints = []
    for t in range(30):
        frame = []
        for i in range(17):
            if i in (5, 6, 11, 12):  # shoulders, hips
                frame.append([0.5 + t * (0.1 / 29), 0.3, 0.9])
            else:
                frame.append([0.5, 0.2, 0.9])
        keypoints.append(frame)
    features = extract_window_features(keypoints, fps=30)
    assert features.center_drift > 0.09  # ~0.1
    assert features.window_duration_s > 0
    assert features.avg_center_speed > 0


def test_normalized_coordinates_unchanged():
    """When max(x,y) <= 1, coordinates are not scaled."""
    keypoints = _make_static_window(5, 17, x=0.8, y=0.7, c=0.9)
    features = extract_window_features(keypoints, fps=30)
    assert features.motion_energy == 0.0
    assert features.mean_pose_conf == 0.9


def test_pixel_coordinates_normalized():
    """When max(x,y) > 1, coordinates normalized so motion_energy in same unit."""
    # Pixel-like: x from 100 to 130 in 30 frames (displacement 1 per frame in pixel)
    # After normalize by 130, displacement per frame = 1/130 ≈ 0.0077
    window = []
    for t in range(30):
        frame = []
        for i in range(17):
            if i in MOTION_KEYPOINT_INDICES:
                frame.append([100.0 + t * 1.0, 50.0, 0.9])
            else:
                frame.append([100.0, 50.0, 0.9])
        window.append(frame)
    features = extract_window_features(window, fps=30)
    assert features.motion_energy > 0
    assert features.motion_energy < 0.1  # normalized
    assert features.mean_pose_conf == 0.9
