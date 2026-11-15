"""What the rest of the robot stack needs from one mapper step, transport-free.

Conventions (ROS REP 103/105, so a ROS bridge can publish these as is):
* `odom` frame = the mapper's world: x forward (at start), y left, z up,
  metres, floor at z = 0. Continuous, may drift; never the global `map`
  (that one belongs to the overhead camera in polaris).
* `base_link` = rover body centre on the floor, x forward, y left, z up.
* Points are in `base_link`, moving objects (people…) already removed.

Caveats to tell consumers: the metric scale is only as good as the floor
calibration (about ±15% on the sequence tested); the pose freezes while
`lost` is True; the whole odom frame is re-expressed once or twice in the
first seconds (metric calibration) — `map_version` changes when it happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapper import Mapper, TrackResult, backproject


@dataclass
class RoverState:
    stamp: float              # seconds (time.time() at the end of the step)
    x: float                  # base_link position in odom (m)
    y: float
    yaw: float                # rad, 0 = odom x axis, counter-clockwise
    lost: bool                # pose not trusted (frozen at the last good one)
    quality: float            # 0..1
    map_version: int          # changes when odom was re-expressed (calibration)
    points: np.ndarray        # (N, 3) float32 in base_link, obstacles candidates


def rover_state(mapper: Mapper, r: TrackResult, stamp: float, cam_forward: float = 0.07,
                stride: int = 4, max_range: float = 4.0) -> RoverState:
    """Robot pose and local point cloud from one mapper step."""
    R, t = r.c2w[:3, :3], r.c2w[:3, 3]
    fwd = R[:, 2]
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.array([t[0] - cam_forward * c, t[1] - cam_forward * s, 0.0])

    pts = np.zeros((0, 3), np.float32)
    if not r.lost and r.depth is not None:
        P = backproject(r.depth, r.K, stride).reshape(-1, 3)
        z = P[:, 2]
        keep = (z > 0.05) & (z < max_range)
        if r.valid is not None:  # trusted depth only, people etc. removed
            keep &= r.valid[::stride, ::stride].reshape(-1)
        W = P[keep] @ R.T + t                       # camera -> odom
        d = W - base                                # odom -> base_link (yaw only:
        pts = np.stack([c * d[:, 0] + s * d[:, 1],  # the rover is on the floor)
                        -s * d[:, 0] + c * d[:, 1],
                        d[:, 2]], 1).astype(np.float32)
    return RoverState(stamp, float(base[0]), float(base[1]), yaw, bool(r.lost),
                      float(r.quality), int(r.map_version), pts)
