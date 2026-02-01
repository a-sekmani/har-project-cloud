"""
Unit tests for infer_activity (app/inference.py).

Covers: low center_drift + good quality → standing; high center_drift → moving (locomotion);
low mean_pose_conf or high missing_ratio → unknown.
"""
import pytest
from app.features import WindowFeatures
from app.inference import (
    MEAN_POSE_CONF_MIN,
    MISSING_RATIO_MAX,
    TH_CENTER_DRIFT,
    TH_MOVE,
    TH_STILL,
    infer_activity,
)


def _features(center_drift=0.0, motion_energy=0.0, missing_ratio=0.0, mean_pose_conf=0.8, window_duration_s=1.0):
    return WindowFeatures(
        missing_ratio=missing_ratio,
        mean_pose_conf=mean_pose_conf,
        motion_energy=motion_energy,
        center_drift=center_drift,
        window_duration_s=window_duration_s,
        avg_center_speed=center_drift / window_duration_s if window_duration_s > 0 else 0.0,
    )


def test_low_center_drift_good_quality_standing():
    """Low center_drift (body not traveling) + good quality → standing."""
    features = _features(center_drift=0.001, motion_energy=0.001)
    quality = (17, 0.8, 1.0)
    activity, confidence, top_k, debug = infer_activity(features, quality)
    assert activity == "standing"
    assert 0.3 <= confidence <= 0.9
    assert "center_drift" in debug["features"]
    assert debug["thresholds"]["TH_CENTER_DRIFT"] == TH_CENTER_DRIFT


def test_high_center_drift_moving():
    """High center_drift (body traveled) → moving (locomotion)."""
    features = _features(center_drift=0.05, motion_energy=0.01)
    quality = (17, 0.8, 1.0)
    activity, confidence, top_k, debug = infer_activity(features, quality)
    assert activity == "moving"
    assert 0.3 <= confidence <= 0.9


def test_low_mean_pose_conf_unknown():
    """mean_pose_conf below MEAN_POSE_CONF_MIN → unknown."""
    features = _features(center_drift=0.0, mean_pose_conf=0.1)
    quality = (17, 0.1, 0.5)
    activity, confidence, top_k, debug = infer_activity(features, quality)
    assert activity == "unknown"
    assert confidence == 0.2


def test_high_missing_ratio_unknown():
    """missing_ratio > MISSING_RATIO_MAX → unknown."""
    features = _features(missing_ratio=0.5, mean_pose_conf=0.5)
    quality = (17, 0.5, 0.3)
    activity, confidence, top_k, debug = infer_activity(features, quality)
    assert activity == "unknown"
    assert confidence == 0.2


def test_low_center_drift_standing_even_with_limb_jitter():
    """High motion_energy but low center_drift (in-place) → standing."""
    features = _features(center_drift=0.001, motion_energy=(TH_STILL + TH_MOVE) / 2)
    quality = (17, 0.8, 1.0)
    activity, confidence, top_k, debug = infer_activity(features, quality)
    assert activity == "standing"
    assert 0.3 <= confidence <= 0.9


def test_top_k_has_three_items():
    """infer_activity returns top_k with at least 3 items."""
    features = _features(center_drift=0.0)
    quality = (17, 0.8, 1.0)
    _, _, top_k, _ = infer_activity(features, quality)
    assert len(top_k) >= 3
    labels = {item.label for item in top_k}
    assert "standing" in labels or "moving" in labels or "unknown" in labels
