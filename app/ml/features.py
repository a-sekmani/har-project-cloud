"""
Build model input from keypoints matching the training preprocessing.
Expects keypoints: list of T frames, each frame 17 keypoints [x, y, conf].
Output shape (1, window_size, 85) with 85 = 51 (pose) + 34 (velocity).

Training preprocessing:
  - norm_center: auto = center on hips midpoint (keypoints 11, 12)
  - norm_scale: auto = divide by shoulder width (keypoints 5, 6)
  - clamp_range: [-3, 3]
  - features: vel = append velocity after pose
  - conf_mode: keep = preserve confidence values

Feature order (per frame):
  [x0,y0,c0, x1,y1,c1, ..., x16,y16,c16, dx0,dy0, dx1,dy1, ..., dx16,dy16]
   ←────────── 51 (pose) ──────────────→ ←────── 34 (velocity) ──────────→
"""
import numpy as np

# COCO-17 keypoint indices
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_HIP = 11
_R_HIP = 12

# Clamp range matching training
_CLAMP_LO = -3.0
_CLAMP_HI = 3.0


def _center_on_hips(arr: np.ndarray) -> np.ndarray:
    """
    Center keypoints on hips midpoint for each frame.
    center = (L_HIP + R_HIP) / 2
    
    Args:
        arr: shape (T, 17, 3) with [x, y, conf]
    Returns:
        centered array (T, 17, 3) with x, y shifted; conf unchanged
    """
    T = arr.shape[0]
    out = arr.copy()
    
    for t in range(T):
        hip_center = (arr[t, _L_HIP, :2] + arr[t, _R_HIP, :2]) / 2.0
        out[t, :, 0] -= hip_center[0]
        out[t, :, 1] -= hip_center[1]
    
    return out


def _scale_by_shoulders(arr: np.ndarray) -> np.ndarray:
    """
    Scale keypoints by shoulder width for each frame.
    scale = ||L_SHOULDER - R_SHOULDER||
    
    Args:
        arr: shape (T, 17, 3) with [x, y, conf] (should be centered first)
    Returns:
        scaled array (T, 17, 3) with x, y divided by shoulder width; conf unchanged
    """
    T = arr.shape[0]
    out = arr.copy()
    
    for t in range(T):
        shoulder_width = np.linalg.norm(arr[t, _L_SHOULDER, :2] - arr[t, _R_SHOULDER, :2])
        shoulder_width = max(shoulder_width, 1e-8)
        out[t, :, 0] /= shoulder_width
        out[t, :, 1] /= shoulder_width
    
    return out


def _clamp(arr: np.ndarray, lo: float = _CLAMP_LO, hi: float = _CLAMP_HI) -> np.ndarray:
    """
    Clamp x, y values to range [lo, hi]. Conf unchanged.
    
    Args:
        arr: shape (T, 17, 3) with [x, y, conf]
    Returns:
        clamped array with x, y in [lo, hi]
    """
    out = arr.copy()
    out[:, :, 0] = np.clip(out[:, :, 0], lo, hi)
    out[:, :, 1] = np.clip(out[:, :, 1], lo, hi)
    return out


def _compute_velocity(arr: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity. First frame velocity = 0.
    
    Args:
        arr: shape (T, 17, 3) with [x, y, conf]
    Returns:
        velocity array (T, 17, 2) with [dx, dy]
    """
    T, K, _ = arr.shape
    vel = np.zeros((T, K, 2), dtype=np.float32)
    vel[1:] = arr[1:, :, :2] - arr[:-1, :, :2]
    return vel


def keypoints_to_model_input(
    keypoints: list,
    window_size: int = 30,
    num_keypoints: int = 17,
) -> np.ndarray:
    """
    Convert keypoints (T, 17, 3) to model input (1, window_size, 85).
    
    Pipeline (matching training):
      1. Center on hips midpoint (L_HIP + R_HIP) / 2
      2. Scale by shoulder width ||L_SHOULDER - R_SHOULDER||
      3. Clamp x, y to [-3, 3]
      4. Compute velocity (frame diff, first frame = 0)
      5. Build 85 features: [pose (51)] + [velocity (34)]
    
    Feature layout per frame:
      pose:     [x0, y0, c0, x1, y1, c1, ..., x16, y16, c16]  = 51 values
      velocity: [dx0, dy0, dx1, dy1, ..., dx16, dy16]         = 34 values
      total:    51 + 34 = 85 values
    
    Args:
        keypoints: list of T frames, each frame is list of 17 [x, y, conf]
        window_size: expected window size (default 30)
        num_keypoints: number of keypoints (default 17)
    
    Returns:
        np.ndarray of shape (1, window_size, 85)
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != num_keypoints or arr.shape[2] < 3:
        raise ValueError(
            f"keypoints must be (T, {num_keypoints}, 3), got shape {getattr(arr, 'shape', '?')}"
        )
    T, K, _ = arr.shape
    if T < 2:
        raise ValueError("Need at least 2 frames for velocity")
    
    # Pad or truncate to window_size
    if T < window_size:
        pad = np.zeros((window_size - T, K, 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:window_size].copy()
    T = window_size

    # Step 1: Center on hips
    arr = _center_on_hips(arr)
    
    # Step 2: Scale by shoulder width
    arr = _scale_by_shoulders(arr)
    
    # Step 3: Clamp to [-3, 3]
    arr = _clamp(arr)
    
    # Step 4: Compute velocity
    vel = _compute_velocity(arr)

    # Step 5: Build 85 features per frame
    # Layout: [x0,y0,c0, x1,y1,c1, ..., x16,y16,c16, dx0,dy0, dx1,dy1, ..., dx16,dy16]
    pose_features = arr[:, :, :3].reshape(T, K * 3)      # (T, 51)
    vel_features = vel.reshape(T, K * 2)                  # (T, 34)
    
    out = np.concatenate([pose_features, vel_features], axis=1)  # (T, 85)
    
    return out[np.newaxis, :, :].astype(np.float32)  # (1, window_size, 85)
