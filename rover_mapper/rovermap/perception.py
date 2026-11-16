"""Obstacle cells in front of the rover, from ONE image's depth.

No tracking: the rover stands still, one depth map (DA3, or a true one) is turned into a
-1/0/1 grid of 5 cm cells in the rover frame:

* the floor plane is fitted in the lower part of the image (RANSAC, floor.fit_ground);
* its distance to the camera, in depth units, against the known camera height above the floor
  gives the metric scale (DA3's depth has no reliable scale of its own);
* rover frame: origin on the floor under the camera, x = optical axis projected on the floor,
  y to the left, z up from the fitted plane;
* a cell is an obstacle (1) when it holds points between band[0] and band[1] above the floor
  (something the rover would hit; higher things such as a table top are passed under), free (0)
  when the floor itself is seen there, and unknown (-1) otherwise. Cells farther along a ray
  than the nearest obstacle on that ray are unknown even if a few floor points land there:
  a 0 erases the drone's obstacles in the fusion, so a hidden cell must never say "free".

Only numpy and OpenCV here, the floor fit included (floor.fit_ground): the geometry and the
D405 run without torch or a GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

UNKNOWN, FREE, OBSTACLE = -1, 0, 1
RES = 0.05  # the fusion's finest cell (merging_map, level z = 4)


@dataclass
class Perception:
    cells: np.ndarray        # (rows, cols) int8, -1/0/1, cell [i][j] = x0 + j*res, y0 + i*res
    x0: float
    y0: float
    res: float
    scale: float             # metres per depth unit (1.0 for a metric depth)
    floor: bool              # False: no floor plane found, every cell is unknown
    # Per valid pixel kept: rover-frame point (N, 3) and pixel (N, 2: u, v), for drawing/eval.
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    pixels: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    plane: tuple | None = None  # (normal, d) in the camera frame, depth units
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            'res': self.res,
            'x0': round(self.x0, 6),
            'y0': round(self.y0, 6),
            'cells': self.cells.astype(int).tolist(),
            'scale': self.scale,
            'floor': self.floor,
            'stats': self.stats,
        }


def valid_depth(depth: np.ndarray, conf: np.ndarray | None = None,
                conf_percentile: float = 25.0) -> np.ndarray:
    """Trusted pixels: finite, positive, confident, not on a depth edge ("flying pixels" between
    a near object and the background land in mid-air and would be taken for obstacles).
    Same rule as Mapper._valid_mask."""
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 1e-6)
    if conf is not None:
        ok &= conf >= np.percentile(conf, conf_percentile)
    d0 = np.where(ok, d, 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(d0, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d0, cv2.CV_32F, 0, 1, ksize=3))
    ok &= (gx + gy) / 8.0 < 0.04 * d0
    return ok


def fit_floor(depth: np.ndarray, K: np.ndarray, valid: np.ndarray, seed: int = 0):
    """(normal towards the camera, camera distance to the plane in depth units) or None."""
    from .floor import fit_ground  # numpy only: the D405 needs no torch

    return fit_ground(depth, K, valid, np.random.default_rng(seed))


def rover_axes(normal: np.ndarray) -> np.ndarray:
    """Rows: rover x (optical axis projected on the floor), y (left), z (up), camera frame."""
    z = np.asarray(normal, np.float64)
    z = z / np.linalg.norm(z)
    f = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x = f / np.linalg.norm(f)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def to_rover(depth: np.ndarray, K: np.ndarray, valid: np.ndarray, plane, cam_height: float,
             scale: float | None = None, stride: int = 1):
    """Rover-frame points (N, 3) in metres, their pixels (N, 2) and the scale used.

    `scale`: metres per depth unit; None = from the floor (cam_height / plane distance)."""
    n, d = plane
    if scale is None:
        scale = cam_height / d
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = valid[::stride, ::stride]
    us, vs, z = us[ok], vs[ok], z[ok].astype(np.float64)
    P = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], -1)
    A = rover_axes(n)
    # floor point under the camera: -d * n (depth units); in metres the camera is at
    # height scale * d, which is cam_height when the scale comes from the floor
    q = (P * scale + scale * d * np.asarray(n)) @ A.T
    return q.astype(np.float32), np.stack([us, vs], -1).astype(np.int32), float(scale)


def classify_points(points: np.ndarray, max_range: float = 1.5, min_range: float = 0.0,
                    band: tuple[float, float] = (0.04, 0.40), res: float = RES,
                    min_obstacle_points: int = 3, min_free_points: int = 2,
                    shadow_bin_deg: float = 2.0, obstacle_range: float | None = None,
                    band_slope: float = 0.0):
    """Rover-frame points -> (cells, x0, y0, stats). The grid spans x in [0, max_range],
    y in [-max_range, max_range] on multiples of `res`.

    For a short-range depth camera on a rover that pitches when it brakes (both off by
    default: one still picture of a metric depth needs neither):
    - `band_slope`: the height from which a point is an obstacle grows with the distance,
      band[0] + band_slope * distance. A floor plane fitted once is wrong by the rover's
      pitch, and the error grows with the distance (2 degrees: 3.5 cm at 1 m); so does the
      depth noise. Below that height a point is floor.
    - `obstacle_range`: no obstacle is reported farther than this (the floor still is, as
      free): far cells are where phantoms come from, and near is early enough to stop."""
    n_cols = math.ceil(max_range / res - 1e-9)
    half = math.ceil(max_range / res - 1e-9)
    x0, y0 = 0.0, -half * res
    n_rows = 2 * half
    cells = np.full((n_rows, n_cols), UNKNOWN, np.int8)
    p = np.asarray(points, np.float64).reshape(-1, 3)
    rng = np.hypot(p[:, 0], p[:, 1])
    keep = (rng <= max_range) & (rng >= min_range) & (p[:, 0] >= 0)
    p, rng = p[keep], rng[keep]
    j = np.floor((p[:, 0] - x0) / res).astype(np.int64)
    i = np.floor((p[:, 1] - y0) / res).astype(np.int64)
    inside = (i >= 0) & (i < n_rows) & (j >= 0) & (j < n_cols)
    p, i, j = p[inside], i[inside], j[inside]
    h = p[:, 2]
    rng = rng[inside]
    low = band[0] + band_slope * rng
    hit = (h >= low) & (h <= band[1])
    if obstacle_range is not None:
        hit &= rng <= obstacle_range
    floor = np.abs(h) < low
    n_hit = np.zeros(cells.shape, np.int32)
    n_floor = np.zeros(cells.shape, np.int32)
    np.add.at(n_hit, (i[hit], j[hit]), 1)
    np.add.at(n_floor, (i[floor], j[floor]), 1)
    obstacle = n_hit >= min_obstacle_points
    free = (n_floor >= min_free_points) & ~obstacle

    # Shadow: along each bearing from the camera foot, nothing beyond the nearest obstacle is
    # free (a few noisy floor points behind a box must not clear what the box hides).
    yc = y0 + (np.arange(n_rows) + 0.5) * res
    xc = x0 + (np.arange(n_cols) + 0.5) * res
    XC, YC = np.meshgrid(xc, yc)
    bearing = np.degrees(np.arctan2(YC, XC))
    dist = np.hypot(XC, YC)
    nb = int(math.ceil(180 / shadow_bin_deg))
    b = np.clip(((bearing + 90) / shadow_bin_deg).astype(np.int64), 0, nb - 1)
    nearest = np.full(nb, np.inf)
    # an obstacle cell shades every bearing its width covers, not only its centre's
    for bc, dc in zip(bearing[obstacle], dist[obstacle], strict=True):
        half_deg = math.degrees(math.atan2(0.75 * res, dc))
        lo = max(0, int((bc - half_deg + 90) / shadow_bin_deg))
        hi = min(nb - 1, int((bc + half_deg + 90) / shadow_bin_deg))
        nearest[lo : hi + 1] = np.minimum(nearest[lo : hi + 1], dc)
    shadow = dist > nearest[b] + res
    free &= ~shadow

    cells[free] = FREE
    cells[obstacle] = OBSTACLE
    stats = {
        'points': int(len(p)),
        'obstacle_cells': int(obstacle.sum()),
        'free_cells': int(free.sum()),
        'shadowed_cells': int((shadow & (n_floor >= min_free_points) & ~obstacle).sum()),
    }
    return cells, x0, y0, stats


def perceive(depth: np.ndarray, K: np.ndarray, cam_height: float = 0.22,
             conf: np.ndarray | None = None, valid: np.ndarray | None = None,
             plane=None, scale: float | None = None, max_range: float = 1.5,
             min_range: float = 0.0, band: tuple[float, float] = (0.04, 0.40),
             res: float = RES, stride: int = 1, obstacle_range: float | None = None,
             band_slope: float = 0.0, min_obstacle_points: int = 3) -> Perception:
    """One depth image (any unit; z along the optical axis, OpenCV axes) -> Perception.

    `plane`: (normal, d) if already known; `scale`: metres per depth unit if known (a metric
    depth: 1.0); otherwise both come from the floor and `cam_height`."""
    depth = np.asarray(depth, np.float32)
    if valid is None:
        valid = valid_depth(depth, conf)
    if plane is None and valid.sum() >= 200:  # fit_ground's own minimum
        plane = fit_floor(depth, K, valid)
    if plane is None:
        cells, x0, y0, stats = classify_points(np.zeros((0, 3)), max_range, min_range, band, res)
        return Perception(cells, x0, y0, res, float('nan'), False, stats={'floor': False})
    pts, pix, s = to_rover(depth, K, valid, plane, cam_height, scale, stride)
    cells, x0, y0, stats = classify_points(
        pts, max_range, min_range, band, res, min_obstacle_points=min_obstacle_points,
        obstacle_range=obstacle_range, band_slope=band_slope)
    n, d = plane
    tilt = float(np.degrees(np.arccos(np.clip(-np.asarray(n)[1], -1.0, 1.0))))
    stats.update({'floor_tilt_deg': round(tilt, 2), 'floor_height_m': round(s * d, 4)})
    return Perception(cells, x0, y0, res, s, True, pts, pix, (np.asarray(n), float(d)), stats)


def draw(image: np.ndarray, per: Perception, K: np.ndarray, max_range: float | None = None
         ) -> np.ndarray:
    """BGR picture for humans: obstacle cells red, free cells green, projected on the image.
    `image` is BGR at the resolution K belongs to."""
    out = image.copy()
    if not per.floor or per.plane is None:
        cv2.putText(out, 'pas de sol trouve', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        return out
    n, d = per.plane
    A = rover_axes(n)
    s = per.scale
    overlay = out.copy()
    rows, cols = per.cells.shape
    for i in range(rows):
        for j in range(cols):
            v = per.cells[i, j]
            if v == UNKNOWN:
                continue
            x, y = per.x0 + j * per.res, per.y0 + i * per.res
            corners = np.array([[x, y, 0], [x + per.res, y, 0], [x + per.res, y + per.res, 0],
                                [x, y + per.res, 0]], np.float64)
            # rover (metres) -> camera (depth units): inverse of to_rover
            P = (corners @ A) / s - d * np.asarray(n)
            if (P[:, 2] <= 1e-6).any():
                continue
            uv = np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2],
                           K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], -1)
            colour = (0, 0, 255) if v == OBSTACLE else (0, 200, 0)
            cv2.fillConvexPoly(overlay, np.round(uv).astype(np.int32), colour)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    txt = f'echelle {s:.3g} m/u  sol incl. {per.stats.get("floor_tilt_deg", 0):.1f} deg'
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out
