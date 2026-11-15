"""Image <-> ground plane geometry for the overhead camera.

The map frame is metric, x right and y up when the area is viewed from above. A homography
`H` maps image pixels (u, v) to map coordinates (x, y) in metres. It is only valid for points
on the floor.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MapBounds:
    """Axis-aligned area of the map frame covered by the grid, in metres."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def grid_shape(self, resolution: float) -> tuple[int, int]:
        """Return (rows, cols) of a grid with `resolution` metres per cell."""
        cols = int(round((self.x_max - self.x_min) / resolution))
        rows = int(round((self.y_max - self.y_min) / resolution))
        return rows, cols


def marker_centers(detections: dict) -> dict:
    """Return {marker id: (u, v) centre pixel} from {marker id: 4x2 corners}."""
    return {marker_id: corners.mean(axis=0) for marker_id, corners in detections.items()}


def estimate_homography(detections: dict, marker_positions: dict) -> np.ndarray | None:
    """Estimate the image -> map homography from floor markers at known map positions.

    `detections` is {marker id: 4x2 corner pixels}; `marker_positions` is {marker id: (x, y)} of
    the marker centres in metres. Returns None unless at least 4 known markers are visible.
    """
    centers = marker_centers(detections)
    ids = sorted(set(centers) & set(marker_positions))
    if len(ids) < 4:
        return None
    image_points = np.array([centers[i] for i in ids], dtype=np.float64)
    map_points = np.array([marker_positions[i] for i in ids], dtype=np.float64)
    homography, _ = cv2.findHomography(image_points, map_points)
    return homography


def metres_per_pixel_from_height(
    camera_height: float,
    focal_length_35mm: float,
    sensor_width_px: int,
    sensor_height_px: int,
) -> float:
    """Ground size of one pixel for a camera looking straight down from `camera_height`.

    `focal_length_35mm` is the 35 mm equivalent focal length (in a photo's EXIF data), which
    relates to the diagonal of the full, uncropped sensor image of the given pixel size. A
    crop keeps the scale; a resize does not.
    """
    ground_diagonal = camera_height * 43.27 / focal_length_35mm
    return ground_diagonal / math.hypot(sensor_width_px, sensor_height_px)


def metres_per_pixel_for_image(
    camera_height: float,
    focal_length_35mm: float,
    sensor_width_px: int,
    sensor_height_px: int,
    image_width_px: int,
) -> float:
    """Ground size of one pixel of an image `image_width_px` wide showing the sensor's view.

    A video frame or a resized photo shows the same field of view with fewer pixels, so each
    pixel covers more ground: 4032 / 1280 times more for a 1280-px frame of a 4032-px sensor.
    Assumes the whole sensor width is in view; a cropped image (most 16:9 video modes, digital
    stabilisation) breaks that, so measure the scale with a ruler for video.
    """
    at_sensor = metres_per_pixel_from_height(
        camera_height, focal_length_35mm, sensor_width_px, sensor_height_px
    )
    return at_sensor * sensor_width_px / image_width_px


def aspect_differs(image_size: tuple[int, int], sensor_size: tuple[int, int]) -> bool:
    """True when (width, height) of the image and of the sensor differ in shape by over 1 %."""
    (width, height), (sensor_width, sensor_height) = image_size, sensor_size
    return abs(width * sensor_height - height * sensor_width) > 0.01 * width * sensor_height


def straight_down_homography(metres_per_pixel: float, image_rows: int) -> np.ndarray:
    """Image -> map homography for a camera looking straight down, without floor markers.

    The map origin is the bottom-left corner of the image, x to the right and y up.
    """
    return np.array(
        [
            [metres_per_pixel, 0.0, 0.0],
            [0.0, -metres_per_pixel, image_rows * metres_per_pixel],
            [0.0, 0.0, 1.0],
        ]
    )


def pixels_to_map(homography: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Map an Nx2 array of pixels to an Nx2 array of map coordinates in metres."""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def map_to_top_down_matrix(bounds: MapBounds, resolution: float) -> np.ndarray:
    """Return the 3x3 matrix taking map metres to top-down image pixels.

    The top-down image has x to the right and y up, so its first row is y_max.
    """
    return np.array(
        [
            [1.0 / resolution, 0.0, -bounds.x_min / resolution],
            [0.0, -1.0 / resolution, bounds.y_max / resolution],
            [0.0, 0.0, 1.0],
        ]
    )


def map_to_top_down(points: np.ndarray, bounds: MapBounds, resolution: float) -> np.ndarray:
    """Map an Nx2 array of map coordinates to top-down image pixels."""
    matrix = map_to_top_down_matrix(bounds, resolution)
    return pixels_to_map(matrix, points)


def warp_to_top_down(
    image: np.ndarray, homography: np.ndarray, bounds: MapBounds, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the camera image to a metric top-down view of `bounds`.

    Returns (top_down, valid), where `valid` is a bool mask of the top-down pixels that the
    camera image actually covers.
    """
    rows, cols = bounds.grid_shape(resolution)
    matrix = map_to_top_down_matrix(bounds, resolution) @ homography

    # A phone image has many pixels per grid cell. Shrink it with area averaging first, or
    # the warp would sample single pixels and turn floor texture into noise.
    center = np.array([[image.shape[1] / 2.0, image.shape[0] / 2.0]])
    step = pixels_to_map(matrix, center + [[1.0, 0.0]]) - pixels_to_map(matrix, center)
    shrink = float(np.linalg.norm(step))  # top-down pixels per image pixel
    if shrink < 0.5:
        factor = shrink * 2.0
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        matrix = matrix @ np.diag([1.0 / factor, 1.0 / factor, 1.0])

    top_down = cv2.warpPerspective(image, matrix, (cols, rows), flags=cv2.INTER_LINEAR)
    coverage = np.full(image.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(coverage, matrix, (cols, rows), flags=cv2.INTER_NEAREST) > 0
    return top_down, valid
