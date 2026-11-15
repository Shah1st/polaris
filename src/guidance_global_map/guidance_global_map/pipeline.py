"""The whole overhead pipeline on one image, with no ROS: image in, grid and rover pose out.

For a camera looking straight down. The map origin is the image's bottom-left corner, x to the
right and y up. Used by `segment_image` (photos) and `drone.OverheadDrone` (the map fusion).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from guidance_calib.aruco_compat import DEFAULT_DICTIONARY, detect_markers
from guidance_calib.ground_plane import (
    MapBounds,
    map_to_top_down,
    metres_per_pixel_from_height,
    pixels_to_map,
    straight_down_homography,
    warp_to_top_down,
)
from guidance_global_map.segmentation import (
    Blob,
    build_cell_grid,
    disc_mask,
    find_blobs,
    find_rover_blob,
    segment_obstacles,
)
from guidance_rover_pose.pose import rover_pose_from_marker


@dataclass(frozen=True)
class OverheadSettings:
    """Parameters of the pipeline. Lengths in metres, angles in radians."""

    # Scale. Either give the ground size of one image pixel directly (measured with a ruler in
    # view), or leave it None to derive it from the camera height and the lens.
    metres_per_pixel: float | None = None
    camera_height: float = 1.9
    focal_length_35mm: float = 26.0
    sensor_size: tuple[int, int] = (4032, 3024)
    # Obstacle detection.
    resolution: float = 0.005
    color_threshold: float = 25.0
    lightness_weight: float = 0.7
    floor_window: float = 1.0
    min_obstacle_area: float = 0.0005
    # Parts of the obstacle mask thinner than this are dropped (morphological opening): tape
    # lines on the floor, a goal marker, cables. 0 = off. Also drops real obstacles that thin.
    min_obstacle_width: float = 0.0
    # Grid.
    cell_size: float = 0.05
    margin: float = 0.0
    # Rover. With a marker id, the rover is the ArUco marker on its top, which also gives its
    # heading. With None, it is the obstacle blob of the rover's size, and has no heading.
    rover_marker_id: int | None = None
    dictionary: str = DEFAULT_DICTIONARY
    rover_radius: float = 0.20
    yaw_offset: float = 0.0
    rover_size: tuple[float, float] = (0.20, 0.23)
    rover_size_tolerance: float = 0.2
    # Map point (x, y) known to be the rover's centre: the rover is then the blob under it
    # (or the nearest within 0.25 m), whatever its size (cables, payload), and its position is
    # this point. None: found by size only.
    rover_hint: tuple[float, float] | None = None
    # Discs (x, y, radius) in map metres forced to free floor, e.g. a goal marker taped on the
    # floor, which is not an obstacle.
    clear_areas: tuple[tuple[float, float, float], ...] = ()

    def scale(self) -> float:
        """Ground size of one image pixel."""
        if self.metres_per_pixel is not None:
            return self.metres_per_pixel
        return metres_per_pixel_from_height(
            self.camera_height, self.focal_length_35mm, *self.sensor_size
        )


@dataclass(frozen=True)
class OverheadResult:
    """What the pipeline found in one image."""

    grid: np.ndarray  # cell grid, row 0 at y = 0; see segmentation.build_cell_grid
    bounds: MapBounds  # area the image covers
    rover_position: tuple[float, float] | None  # x, y; None when the rover is not found
    rover_heading: float | None  # counter-clockwise from the x axis; None when unknown
    top_down: np.ndarray  # the view from above at `resolution`
    blobs: list[Blob]  # obstacle blobs, the rover's included when it was found by size
    rover_mask: np.ndarray | None  # top-down pixels of the rover


def _blob_at(blobs, col: float, row: float, max_distance_px: float):
    """The blob covering top-down pixel (col, row), else the nearest one within reach."""
    best, best_distance = None, max_distance_px
    for blob in blobs:
        r, c = int(round(row)), int(round(col))
        if 0 <= r < blob.mask.shape[0] and 0 <= c < blob.mask.shape[1] and blob.mask[r, c]:
            return blob
        distance = float(np.hypot(blob.center[0] - col, blob.center[1] - row))
        if distance <= best_distance:
            best, best_distance = blob, distance
    return best


def process_image(image: np.ndarray, settings: OverheadSettings | None = None) -> OverheadResult:
    """Run the pipeline on one BGR image taken straight down."""
    settings = settings or OverheadSettings()
    resolution = settings.resolution
    metres_per_pixel = settings.scale()
    rows, cols = image.shape[:2]
    homography = straight_down_homography(metres_per_pixel, rows)
    bounds = MapBounds(0.0, 0.0, cols * metres_per_pixel, rows * metres_per_pixel)
    top_down, valid = warp_to_top_down(image, homography, bounds, resolution)

    rover_position, rover_heading, rover_mask = None, None, None
    if settings.rover_marker_id is not None:
        corners = detect_markers(image, settings.dictionary).get(settings.rover_marker_id)
        if corners is not None:
            x, y, rover_heading = rover_pose_from_marker(corners, homography, settings.yaw_offset)
            rover_position = (x, y)
            center = map_to_top_down(pixels_to_map(homography, corners), bounds, resolution)
            rover_mask = disc_mask(
                valid.shape, center.mean(axis=0), settings.rover_radius / resolution
            )

    obstacles = segment_obstacles(
        top_down,
        valid,
        rover_mask,
        color_threshold=settings.color_threshold,
        lightness_weight=settings.lightness_weight,
        min_area_px=int(round(settings.min_obstacle_area / resolution**2)),
        floor_window_px=int(round(settings.floor_window / resolution)),
    )
    for x, y, radius in settings.clear_areas:
        centre = ((x - bounds.x_min) / resolution, (bounds.y_max - y) / resolution)
        obstacles = obstacles & ~disc_mask(obstacles.shape, centre, radius / resolution)
    if settings.min_obstacle_width > 0:
        side = max(1, int(round(settings.min_obstacle_width / resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        obstacles = cv2.morphologyEx(obstacles.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    blobs = find_blobs(obstacles, resolution)

    if settings.rover_marker_id is None:
        rover = find_rover_blob(blobs, settings.rover_size, settings.rover_size_tolerance)
        if settings.rover_hint is not None:
            hx, hy = settings.rover_hint
            col, row = (hx - bounds.x_min) / resolution, (bounds.y_max - hy) / resolution
            rover = _blob_at(blobs, col, row, 0.25 / resolution)
        if rover is not None:
            rover_mask = rover.mask
            rover_position = (
                rover.center[0] * resolution,
                bounds.y_max - rover.center[1] * resolution,
            )
            if settings.rover_hint is not None:
                rover_position = tuple(settings.rover_hint)
            obstacles = obstacles & ~rover.mask

    grid = build_cell_grid(
        obstacles, valid, resolution, settings.cell_size, settings.margin, rover_mask
    )
    return OverheadResult(grid, bounds, rover_position, rover_heading, top_down, blobs, rover_mask)
