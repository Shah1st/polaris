"""Rover pose in the map frame from the ArUco marker on its top.

The marker is mounted so that its top edge (the edge between its first two corners) faces the
rover's forward direction. `yaw_offset` corrects any other mounting.
"""

import math

import numpy as np

from guidance_calib.ground_plane import pixels_to_map


def rover_pose_from_marker(
    corners: np.ndarray, homography: np.ndarray, yaw_offset: float = 0.0
) -> tuple[float, float, float]:
    """Return (x, y, yaw) in the map frame from the marker's 4x2 corner pixels.

    Yaw is in radians, counter-clockwise from the map x axis, in (-pi, pi].
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center_px = corners.mean(axis=0)
    front_px = corners[:2].mean(axis=0)
    center, front = pixels_to_map(homography, np.array([center_px, front_px]))
    yaw = math.atan2(front[1] - center[1], front[0] - center[0]) + yaw_offset
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return float(center[0]), float(center[1]), yaw
