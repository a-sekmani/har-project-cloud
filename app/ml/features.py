"""
Build model input from keypoints for c_norm_vel-style models.
Expects keypoints: list of T frames, each frame 17 keypoints [x, y, conf].
Output shape (1, window_size, 85) with 85 = 17*(x, y, conf, vel_x, vel_y).
"""
import numpy as np


def keypoints_to_model_input(
    keypoints: list,
    window_size: int = 30,
    num_keypoints: int = 17,
    features_per_kp: int = 5,
) -> np.ndarray:
    """
    Convert keypoints (T, 17, 3) to model input (1, window_size, 85).
    Adds velocity as frame-to-frame difference; first frame velocity zero.
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != num_keypoints or arr.shape[2] < 3:
        raise ValueError(
            f"keypoints must be (T, {num_keypoints}, 3), got shape {getattr(arr, 'shape', '?')}"
        )
    T, K, _ = arr.shape
    if T < 2:
        raise ValueError("Need at least 2 frames for velocity")
    # Take first window_size frames (or pad if needed)
    if T < window_size:
        pad = np.zeros((window_size - T, K, 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
        T = window_size
    else:
        arr = arr[:window_size].copy()
        T = window_size

    # x, y, conf for each keypoint
    xy_conf = arr[:, :, :3]  # (T, 17, 3)
    # velocity: diff; frame 0 vel = 0
    vel = np.zeros((T, K, 2), dtype=np.float32)
    vel[1:] = arr[1:, :, :2] - arr[:-1, :, :2]

    # Flatten to (T, 17*5) = (T, 85): for each kp [x, y, conf, vx, vy]
    out = np.zeros((T, K * features_per_kp), dtype=np.float32)
    out[:, 0::5] = xy_conf[:, :, 0]  # x
    out[:, 1::5] = xy_conf[:, :, 1]  # y
    out[:, 2::5] = xy_conf[:, :, 2]  # conf
    out[:, 3::5] = vel[:, :, 0]       # vel_x
    out[:, 4::5] = vel[:, :, 1]       # vel_y

    return out[np.newaxis, :, :]  # (1, window_size, 85)
