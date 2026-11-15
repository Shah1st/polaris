"""Rover pose in the map frame from the ArUco marker on its top.

The marker is mounted so that its top edge (the edge between its first two corners) faces the
rover's forward direction. `yaw_offset` corrects any other mounting.

The image -> map homography is the floor's. The marker sits `marker_height` above the floor,
so the floor homography places it too far from the point under the camera (the nadir), by a
factor H / (H - h) for a camera at height H. `correct_parallax` undoes that; it needs the
camera to look straight down (the nadir is then the image centre).
"""

import math

import numpy as np

from guidance_calib.ground_plane import pixels_to_map


def correct_parallax(
    points: np.ndarray,
    nadir: tuple[float, float],
    camera_height: float,
    marker_height: float,
) -> np.ndarray:
    """Move Nx2 map points seen at `marker_height` to the floor point below them.

    `nadir` is the map point straight under the camera. Marker height 0 changes nothing.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if marker_height == 0.0:
        return points
    if not 0.0 <= marker_height < camera_height:
        raise ValueError('marker_height must be at least 0 and below camera_height')
    nadir = np.asarray(nadir, dtype=np.float64).reshape(1, 2)
    return nadir + (points - nadir) * (camera_height - marker_height) / camera_height


def rover_pose_from_marker(
    corners: np.ndarray,
    homography: np.ndarray,
    yaw_offset: float = 0.0,
    marker_height: float = 0.0,
    camera_height: float | None = None,
    nadir: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Return (x, y, yaw) in the map frame from the marker's 4x2 corner pixels.

    Yaw is in radians, counter-clockwise from the map x axis, in (-pi, pi]. With a
    `marker_height` above 0, `camera_height` and `nadir` (map point under the camera) are
    needed and the position is corrected for the parallax; the yaw does not change.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center_px = corners.mean(axis=0)
    front_px = corners[:2].mean(axis=0)
    center, front = pixels_to_map(homography, np.array([center_px, front_px]))
    if marker_height:
        if camera_height is None or nadir is None:
            raise ValueError('marker_height needs camera_height and nadir')
        center, front = correct_parallax(
            np.array([center, front]), nadir, camera_height, marker_height
        )
    yaw = math.atan2(front[1] - center[1], front[0] - center[0]) + yaw_offset
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return float(center[0]), float(center[1]), yaw
