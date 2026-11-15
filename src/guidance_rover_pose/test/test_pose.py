import math

import numpy as np
import pytest

from guidance_calib.aruco_compat import detect_markers
from guidance_calib.ground_plane import estimate_homography, straight_down_homography
from guidance_calib.synthetic import Scene, render_camera_view
from guidance_rover_pose.pose import correct_parallax, rover_pose_from_marker


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


def project_straight_down(points_3d, camera_xy, camera_height, focal_px, centre_px):
    """Pinhole camera looking straight down: map (x, y, z) -> image pixels, image v down."""
    pixels = []
    for x, y, z in points_3d:
        depth = camera_height - z
        u = centre_px[0] + focal_px * (x - camera_xy[0]) / depth
        v = centre_px[1] - focal_px * (y - camera_xy[1]) / depth
        pixels.append((u, v))
    return np.array(pixels)


@pytest.mark.parametrize('marker_height', [0.0, 0.12, 0.22])
def test_raised_marker_is_put_back_over_its_floor_point(marker_height: float) -> None:
    # Camera 1.9 m up, 1000 px focal length (floor scale 1.9 mm per pixel), 2600 x 2000 image.
    height, focal, centre = 1.9, 1000.0, (1300.0, 1000.0)
    scale = height / focal
    homography = straight_down_homography(scale, 2000)  # map origin at the image's bottom-left
    nadir = (centre[0] * scale, (2000 - centre[1]) * scale)
    true_x, true_y, true_yaw = nadir[0] - 1.0, nadir[1] + 0.8, math.radians(40.0)
    distance = math.hypot(1.0, 0.8)
    half = 0.05
    forward = np.array([math.cos(true_yaw), math.sin(true_yaw)]) * half
    left = np.array([-math.sin(true_yaw), math.cos(true_yaw)]) * half
    center = np.array([true_x, true_y])
    corners_map = [center + forward + left, center + forward - left]
    corners_map += [center - forward - left, center - forward + left]
    corners = project_straight_down(
        [(*c, marker_height) for c in corners_map], nadir, height, focal, centre
    )

    raw_x, raw_y, _ = rover_pose_from_marker(corners, homography)
    x, y, yaw = rover_pose_from_marker(
        corners, homography, marker_height=marker_height, camera_height=height, nadir=nadir
    )

    assert (x, y) == pytest.approx((true_x, true_y), abs=0.002)
    assert angle_difference(yaw, true_yaw) < math.radians(0.5)
    error = math.dist((raw_x, raw_y), (true_x, true_y))
    assert error == pytest.approx(distance * marker_height / (height - marker_height), abs=0.002)


def test_parallax_correction_needs_the_camera_height() -> None:
    corners = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError):
        rover_pose_from_marker(corners, np.eye(3), marker_height=0.1)
    with pytest.raises(ValueError):
        correct_parallax(corners, (0.0, 0.0), camera_height=0.1, marker_height=0.2)
    assert correct_parallax(corners, (5.0, 5.0), 1.9, 0.0) == pytest.approx(corners)
