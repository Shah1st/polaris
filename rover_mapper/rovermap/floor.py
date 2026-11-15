"""Floor plane of a depth map, numpy only (no torch): the perception service uses it with
the D405's measured depth, without the depth model. Moved out of mapper.py, which
re-exports both names."""
from __future__ import annotations

import numpy as np


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 1):
    """Depth map -> camera-frame points (h, w, 3) on a strided pixel grid."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)


def fit_ground(depth: np.ndarray, K: np.ndarray, valid: np.ndarray, rng: np.random.Generator):
    """RANSAC floor plane in the camera frame.

    Returns (normal pointing up towards the camera, camera height in depth
    units) or None. Camera axes are OpenCV style: x right, y down, z forward.
    """
    h, w = depth.shape
    pts = backproject(depth, K, 4)
    ok = valid[::4, ::4].copy()
    ok[: ok.shape[0] // 2] = False  # the floor is in the lower half of the image
    P = pts[ok]
    if len(P) < 200:
        return None
    if len(P) > 5000:
        P = P[rng.choice(len(P), 5000, replace=False)]
    thresh = 0.015 * float(np.median(P[:, 2]))
    cam_up = np.array([0.0, -1.0, 0.0])
    best, best_count = None, 0
    for _ in range(300):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ a
        if d < 0:  # orient the normal towards the camera (origin)
            n, d = -n, -d
        if n @ cam_up < np.cos(np.radians(45)):  # floor must be roughly below us
            continue
        count = int((np.abs(P @ n + d) < thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    if best is None or best_count < 0.3 * len(P):
        return None
    n, d = best
    inl = P[np.abs(P @ n + d) < thresh]
    centroid = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[2]
    d = -n @ centroid
    if d < 0:
        n, d = -n, -d
    return n, float(d)
