import math

import numpy as np
import pytest

from guidance_calib.aruco_compat import detect_markers
from guidance_calib.ground_plane import estimate_homography
from guidance_calib.synthetic import Scene, render_camera_view
from guidance_rover_pose.pose import rover_pose_from_marker


def angle_difference(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@pytest.mark.parametrize(
    'true_pose',
    [
        (1.00, 0.75, 0.0),
        (0.30, 1.10, math.radians(-30.0)),
        (1.60, 0.40, math.radians(90.0)),
        (0.80, 0.50, math.radians(170.0)),
        (1.20, 1.00, math.radians(-135.0)),
    ],
)
def test_rover_pose_matches_ground_truth(true_pose: tuple[float, float, float]) -> None:
    scene = Scene(rover_pose=true_pose)
    image, _ = render_camera_view(scene)
    detections = detect_markers(image)
    homography = estimate_homography(detections, scene.marker_positions)

    x, y, yaw = rover_pose_from_marker(detections[scene.rover_marker_id], homography)

    assert abs(x - true_pose[0]) < 0.01
    assert abs(y - true_pose[1]) < 0.01
    assert angle_difference(yaw, true_pose[2]) < math.radians(2.0)


def test_yaw_offset_is_added_and_wrapped() -> None:
    # Identity homography, marker top edge facing +x, so yaw is 0 before the offset.
    corners = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])

    _, _, yaw = rover_pose_from_marker(corners, np.eye(3), yaw_offset=math.radians(270.0))

    assert yaw == pytest.approx(math.radians(-90.0))
