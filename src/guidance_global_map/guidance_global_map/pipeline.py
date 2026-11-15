"""The whole overhead pipeline on one image, with no ROS: image in, grid and rover pose out.

For a camera looking straight down. The map origin is the image's bottom-left corner, x to the
right and y up. Used by `segment_image` (photos) and `drone.OverheadDrone` (the map fusion).
"""

import logging
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

log = logging.getLogger(__name__)
_warned: set = set()


def _warn_once(key, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(message, *args)


@dataclass(frozen=True)
class OverheadSettings:
    """Parameters of the pipeline. Lengths in metres, angles in radians."""

    # Scale. Either give the ground size of one image pixel directly (measured with a ruler in
    # view: the recommended way for a live camera), or leave it None to derive it from the
    # camera height and the lens; the image is then taken to show the whole sensor's field of
    # view across its width, whatever its pixel size.
    metres_per_pixel: float | None = None
    # Also used to correct the marker's parallax (marker_height), even with metres_per_pixel.
    camera_height: float = 1.9
    focal_length_35mm: float = 26.0
    sensor_size: tuple[int, int] = (4032, 3024)
    # Obstacle detection.
    resolution: float = 0.005
    color_threshold: float = 25.0
    lightness_weight: float = 0.7
    floor_window: float = 1.0
    min_obstacle_area: float = 0.0005
    # Grid.
    cell_size: float = 0.05
    margin: float = 0.0
    # Rover. With a marker id, the rover is the ArUco marker on its top, which also gives its
    # heading. With None, it is the obstacle blob of the rover's size, and has no heading.
    rover_marker_id: int | None = None
    dictionary: str = DEFAULT_DICTIONARY
    rover_radius: float = 0.20
    yaw_offset: float = 0.0
    # Height of the rover's marker above the floor. Its position is moved towards the point
    # under the camera by (camera_height - marker_height) / camera_height. 0: no correction.
    marker_height: float = 0.0
    rover_size: tuple[float, float] = (0.20, 0.23)
    rover_size_tolerance: float = 0.2

    def scale(self, image_size: tuple[int, int] | None = None) -> float:
        """Ground size of one pixel of an image of `image_size` (width, height) pixels.

        Without `metres_per_pixel`, the scale follows from the camera height and the lens for
        the declared sensor size, then from the image width: a 1280-px video frame of the same
        field of view has pixels 4032 / 1280 times larger than the photo's. A different aspect
        ratio means the image is cropped (most video modes), so the scale is a guess: measure
        `metres_per_pixel` instead. None for `image_size` means the sensor size.
        """
        if self.metres_per_pixel is not None:
            return self.metres_per_pixel
        sensor_width, sensor_height = self.sensor_size
        at_sensor = metres_per_pixel_from_height(
            self.camera_height, self.focal_length_35mm, sensor_width, sensor_height
        )
        if image_size is None:
            return at_sensor
        width, height = image_size
        if abs(width * sensor_height - height * sensor_width) > 0.01 * width * sensor_height:
            _warn_once(
                ('aspect', width, height, sensor_width, sensor_height),
                'image %dx%d does not have the aspect ratio of the declared sensor %dx%d: the '
                'scale from the camera height assumes the full sensor width is in view and may '
                'be wrong; measure metres_per_pixel with a ruler on the floor',
                width,
                height,
                sensor_width,
                sensor_height,
            )
        return at_sensor * sensor_width / width


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


def _gray_sample(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    step = max(1, gray.shape[1] // 320)
    return gray[::step, ::step]


def brightness_spread(image: np.ndarray) -> float:
    """The statistic `frame_problem` compares against its `reference_std`."""
    return float(_gray_sample(image).std())


def frame_problem(
    image: np.ndarray, reference_std: float | None = None, collapse: float = 0.35
) -> str | None:
    """Why this frame must not be mapped (a reason), or None when it looks usable.

    A black, white, overexposed or nearly uniform frame (lens covered, camera starting up,
    exposure jump) would otherwise give "everything free and seen". `reference_std` is the
    brightness spread of the last good frame: a frame whose spread falls below `collapse`
    times it has lost the scene (hand in front, camera knocked over).
    """
    if image is None or image.size == 0:
        return 'no image'
    gray = _gray_sample(image)
    spread = float(gray.std())
    mean = float(gray.mean())
    # Thresholds leave room for a plain, evenly lit floor (spread about 3-10) and for a dim
    # hall, which a camera's automatic exposure brings to a mean of about 100.
    if spread < 2.5:
        return f'nearly uniform (brightness spread {spread:.1f})'
    if mean > 235.0 or (gray >= 250).mean() > 0.6:
        return f'overexposed (mean brightness {mean:.0f})'
    if mean < 35.0:
        return f'too dark (mean brightness {mean:.0f})'
    if reference_std is not None and spread < collapse * reference_std:
        return f'scene lost (brightness spread {spread:.1f}, was {reference_std:.1f})'
    return None


def process_image(image: np.ndarray, settings: OverheadSettings | None = None) -> OverheadResult:
    """Run the pipeline on one BGR image taken straight down."""
    settings = settings or OverheadSettings()
    resolution = settings.resolution
    rows, cols = image.shape[:2]
    metres_per_pixel = settings.scale((cols, rows))
    homography = straight_down_homography(metres_per_pixel, rows)
    bounds = MapBounds(0.0, 0.0, cols * metres_per_pixel, rows * metres_per_pixel)
    top_down, valid = warp_to_top_down(image, homography, bounds, resolution)

    rover_position, rover_heading, rover_mask = None, None, None
    if settings.rover_marker_id is not None:
        corners = detect_markers(image, settings.dictionary).get(settings.rover_marker_id)
        if corners is not None:
            nadir = (cols * metres_per_pixel / 2.0, rows * metres_per_pixel / 2.0)
            x, y, rover_heading = rover_pose_from_marker(
                corners,
                homography,
                settings.yaw_offset,
                settings.marker_height,
                settings.camera_height,
                nadir,
            )
            rover_position = (x, y)
            # In the image the rover spans from its floor footprint (the corrected position)
            # to its top (where the marker appears): mask around both.
            seen = pixels_to_map(homography, corners).mean(axis=0)
            centers = map_to_top_down(np.array([seen, (x, y)]), bounds, resolution)
            radius = settings.rover_radius / resolution
            rover_mask = disc_mask(valid.shape, centers[0], radius) | disc_mask(
                valid.shape, centers[1], radius
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
    blobs = find_blobs(obstacles, resolution)

    if settings.rover_marker_id is None:
        rover = find_rover_blob(blobs, settings.rover_size, settings.rover_size_tolerance)
        if rover is not None:
            rover_mask = rover.mask
            rover_position = (
                rover.center[0] * resolution,
                bounds.y_max - rover.center[1] * resolution,
            )
            obstacles = obstacles & ~rover.mask

    grid = build_cell_grid(
        obstacles, valid, resolution, settings.cell_size, settings.margin, rover_mask
    )
    return OverheadResult(grid, bounds, rover_position, rover_heading, top_down, blobs, rover_mask)
