"""Synthetic overhead-camera images with known ground truth.

Used by the tests of the overhead packages and to make a sample image when no real photo is at
hand:

    python -m guidance_calib.synthetic data/overhead/synthetic_scene.png
"""

import math
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from guidance_calib.aruco_compat import generate_marker
from guidance_calib.ground_plane import MapBounds

FLOOR_COLOR = (175, 180, 185)
DEFAULT_MARKER_POSITIONS = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (2.0, 1.5), 3: (0.0, 1.5)}


@dataclass
class Box:
    """Axis-aligned obstacle in map metres, with a BGR colour."""

    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int] = (40, 60, 200)


@dataclass
class Scene:
    bounds: MapBounds = field(default_factory=lambda: MapBounds(0.0, 0.0, 2.0, 1.5))
    marker_positions: dict = field(default_factory=lambda: dict(DEFAULT_MARKER_POSITIONS))
    marker_side: float = 0.10
    boxes: list = field(default_factory=list)
    rover_marker_id: int = 10
    rover_pose: tuple[float, float, float] | None = None  # x, y, yaw
    margin: float = 0.30
    pixels_per_metre: float = 300.0


def _canvas_to_map_matrix(scene: Scene) -> np.ndarray:
    scale = 1.0 / scene.pixels_per_metre
    return np.array(
        [
            [scale, 0.0, scene.bounds.x_min - scene.margin],
            [0.0, -scale, scene.bounds.y_max + scene.margin],
            [0.0, 0.0, 1.0],
        ]
    )


def _map_to_canvas(scene: Scene, points: np.ndarray) -> np.ndarray:
    matrix = np.linalg.inv(_canvas_to_map_matrix(scene))
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


def _paste_marker(
    canvas: np.ndarray, scene: Scene, marker_id: int, x: float, y: float, yaw: float
) -> None:
    """Draw a marker centred at (x, y) whose top edge faces the direction `yaw`."""
    side_px = 200
    border = side_px // 4
    patch = cv2.copyMakeBorder(
        generate_marker(marker_id, side_px),
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    half = scene.marker_side * (side_px + 2 * border) / side_px / 2.0
    forward = np.array([math.cos(yaw), math.sin(yaw)]) * half
    left = np.array([-math.sin(yaw), math.cos(yaw)]) * half
    center = np.array([x, y])
    corners_map = [
        center + forward + left,
        center + forward - left,
        center - forward - left,
        center - forward + left,
    ]
    size = patch.shape[0]
    source = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    target = _map_to_canvas(scene, np.array(corners_map)).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    shape = (canvas.shape[1], canvas.shape[0])
    warped = cv2.warpPerspective(cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR), matrix, shape)
    covered = cv2.warpPerspective(np.full(patch.shape, 255, np.uint8), matrix, shape) > 0
    canvas[covered] = warped[covered]


def render_top_down(scene: Scene) -> np.ndarray:
    """Render the scene as seen from straight above, including a margin around the bounds."""
    width = scene.bounds.x_max - scene.bounds.x_min + 2 * scene.margin
    height = scene.bounds.y_max - scene.bounds.y_min + 2 * scene.margin
    rows = int(round(height * scene.pixels_per_metre))
    cols = int(round(width * scene.pixels_per_metre))
    canvas = np.empty((rows, cols, 3), dtype=np.uint8)
    canvas[:] = FLOOR_COLOR
    noise = np.random.default_rng(0).normal(0.0, 2.0, canvas.shape)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

    for box in scene.boxes:
        corners = np.array([[box.x, box.y + box.height], [box.x + box.width, box.y]])
        (u0, v0), (u1, v1) = np.round(_map_to_canvas(scene, corners)).astype(int)
        cv2.rectangle(canvas, (u0, v0), (u1, v1), box.color, -1)
    for marker_id, (x, y) in scene.marker_positions.items():
        _paste_marker(canvas, scene, marker_id, x, y, math.pi / 2.0)
    if scene.rover_pose is not None:
        _paste_marker(canvas, scene, scene.rover_marker_id, *scene.rover_pose)
    return canvas


def render_camera_view(scene: Scene, tilt: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Render the scene through a tilted camera.

    Returns (image, homography), where `homography` is the true image -> map mapping. `tilt`
    is how far the far edge is squeezed, as a fraction of the image width.
    """
    canvas = render_top_down(scene)
    rows, cols = canvas.shape[:2]
    source = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype=np.float32)
    target = np.array(
        [[cols * tilt, rows * 0.05], [cols * (1 - tilt), 0], [cols, rows], [0, rows * 0.97]],
        dtype=np.float32,
    )
    canvas_to_image = cv2.getPerspectiveTransform(source, target)
    image = cv2.warpPerspective(
        canvas, canvas_to_image, (cols, rows), borderValue=FLOOR_COLOR, flags=cv2.INTER_LINEAR
    )
    homography = _canvas_to_map_matrix(scene) @ np.linalg.inv(canvas_to_image)
    return image, homography / homography[2, 2]


def demo_scene() -> Scene:
    return Scene(
        boxes=[
            Box(0.50, 0.30, 0.30, 0.20),
            Box(1.20, 0.90, 0.25, 0.35, color=(60, 140, 40)),
            Box(0.90, 0.55, 0.15, 0.15, color=(30, 30, 30)),
        ],
        rover_pose=(0.30, 1.10, math.radians(-30.0)),
    )


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_scene.png'
    image, _ = render_camera_view(demo_scene())
    cv2.imwrite(path, image)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
