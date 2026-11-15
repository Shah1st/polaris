"""Find the rover in a picture from the air: the ArUco marker on its roof.

The marker gives the rover's position AND heading in one picture, in the floor frame of
pose_camera.py, whatever the camera's pose. No model, a few milliseconds per picture.

    finder = RoverFinder(marker_id=0, size=0.10, height=0.12)
    seen = finder(image, pose)       # None, or a RoverSeen (x, y, heading, quality)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from pose_camera import Pose, PoseError


@dataclass(frozen=True)
class RoverSeen:
    x: float  # metres, floor frame: the centre of the marker
    y: float
    heading: float  # radians, the rover's forward direction in the floor frame
    side_px: float  # the marker's side in the picture: under ~20 px the reading is fragile
    scale_error: float  # measured side / printed side - 1: far from 0 = pose or height wrong


class RoverFinder:
    def __init__(
        self,
        marker_id: int = 0,
        size: float = 0.10,
        height: float = 0.12,
        forward_deg: float = 90.0,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """`size`: printed side of the marker (m). `height`: of the marker above the floor (m).
        `forward_deg`: the rover's forward direction, counter-clockwise from the marker's top
        edge (corner 0 -> 1); 90 = the top of the printed marker points to the rover's front."""
        self.marker_id = marker_id
        self.size = size
        self.height = height
        self.forward = math.radians(forward_deg)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary), params
        )

    def corners(self, image: np.ndarray) -> np.ndarray | None:
        """The marker's four corners in the picture (4 x 2), or None."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        for quad, marker_id in zip(found, ids.ravel(), strict=False):
            if marker_id == self.marker_id:
                return quad.reshape(4, 2)
        return None

    def __call__(self, image: np.ndarray, pose: Pose) -> RoverSeen | None:
        quad = self.corners(image)
        if quad is None:
            return None
        try:
            m = pose.locate_marker(quad, self.height)
        except PoseError:
            return None
        side_px = float(np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]))
        # Seen from above, the top of the printed marker is its top edge (corner 0 -> 1) turned
        # by +90 degrees: with forward_deg = 90 the top of the print is the rover's front.
        heading = m.heading + self.forward
        heading = math.atan2(math.sin(heading), math.cos(heading))
        return RoverSeen(m.x, m.y, heading, side_px, m.side / self.size - 1.0)
