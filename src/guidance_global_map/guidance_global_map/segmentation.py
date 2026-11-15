"""Obstacle segmentation on the metric top-down view, and conversion to an occupancy grid.

Baseline method: the floor is assumed to be the most common colour in the view. A pixel is an
obstacle when its colour is far from the floor colour.
"""

from dataclasses import dataclass

import cv2
import numpy as np

OCCUPIED = 100
MARGIN = 99
# The cells the rover itself stands on. Not a standard nav_msgs/OccupancyGrid meaning: a
# planner must treat these cells as its start, not as a cost.
ROVER = 50
FREE = 0
UNKNOWN = -1


def scale_polygon(polygon: np.ndarray, factor: float) -> np.ndarray:
    """Scale an Nx2 polygon about its centroid."""
    polygon = np.asarray(polygon, dtype=np.float64)
    center = polygon.mean(axis=0)
    return center + (polygon - center) * factor


def polygons_mask(shape: tuple[int, int], polygons: list) -> np.ndarray:
    """Return a bool mask of `shape` (rows, cols) that is True inside any of the polygons."""
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    return mask > 0


def disc_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    """Return a bool mask of `shape` (rows, cols) that is True inside the disc (pixel units)."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), int(round(radius)), 255, -1)
    return mask > 0


def _color_distance(lab: np.ndarray, floor: np.ndarray, lightness_weight: float) -> np.ndarray:
    weights = np.array([lightness_weight, 1.0, 1.0], dtype=np.float32)
    difference = (lab.astype(np.float32) - floor.astype(np.float32)) * weights
    return np.linalg.norm(difference, axis=2)


def local_floor_color(
    lab: np.ndarray,
    usable: np.ndarray,
    window_px: int,
    color_threshold: float,
    lightness_weight: float,
) -> np.ndarray:
    """Estimate the floor colour at every pixel of an 8-bit Lab image.

    The floor colour is the median over a `window_px` neighbourhood, so it follows uneven
    lighting. The window must be wider than about twice the largest obstacle, or the obstacle
    becomes its own background. Two refinement passes replace obstacle pixels by the current
    estimate before taking the median again, which keeps large obstacles from pulling it.
    """
    scale = max(1, window_px // 64)
    size = (max(1, lab.shape[1] // scale), max(1, lab.shape[0] // scale))
    small = cv2.resize(lab, size, interpolation=cv2.INTER_AREA)
    small_usable = cv2.resize(usable.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    kernel = max(3, (window_px // scale) | 1)

    work = small.copy()
    work[~small_usable] = np.median(lab[usable], axis=0).astype(np.uint8)
    for _ in range(2):
        floor = cv2.medianBlur(work, kernel)
        obstacles = _color_distance(small, floor, lightness_weight) > color_threshold
        replace = obstacles | ~small_usable
        work = small.copy()
        work[replace] = floor[replace]
    floor = cv2.medianBlur(work, kernel)
    return cv2.resize(floor, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)


def segment_obstacles(
    top_down: np.ndarray,
    valid: np.ndarray | None = None,
    ignore: np.ndarray | None = None,
    color_threshold: float = 25.0,
    lightness_weight: float = 0.7,
    min_area_px: int = 50,
    floor_window_px: int = 0,
) -> np.ndarray:
    """Return a bool mask of obstacle pixels in a BGR top-down image.

    `valid` marks pixels the camera covers; `ignore` marks pixels that must never be obstacles
    (floor markers, the rover). `lightness_weight` below 1 makes shadows count less than a
    change of hue. Blobs smaller than `min_area_px` are dropped. With `floor_window_px` > 0 the
    floor colour is estimated locally (see `local_floor_color`); with 0 it is one colour for
    the whole view.
    """
    rows, cols = top_down.shape[:2]
    if valid is not None:
        # Work on the part the camera covers only. Otherwise the empty surround would leak into
        # the blur and the floor estimate, and the result near the edges of the view would
        # depend on how much larger the map area is than the view.
        if not valid.any():
            return np.zeros((rows, cols), dtype=bool)
        seen_rows = np.flatnonzero(valid.any(axis=1))
        seen_cols = np.flatnonzero(valid.any(axis=0))
        box = (
            slice(seen_rows[0], seen_rows[-1] + 1),
            slice(seen_cols[0], seen_cols[-1] + 1),
        )
        if valid[box].shape != (rows, cols):
            obstacles = np.zeros((rows, cols), dtype=bool)
            obstacles[box] = segment_obstacles(
                top_down[box],
                valid[box],
                None if ignore is None else ignore[box],
                color_threshold,
                lightness_weight,
                min_area_px,
                floor_window_px,
            )
            return obstacles

    usable = np.ones((rows, cols), dtype=bool)
    if valid is not None:
        usable &= valid
    if ignore is not None:
        usable &= ~ignore
    if not usable.any():
        return np.zeros((rows, cols), dtype=bool)

    lab = cv2.cvtColor(cv2.GaussianBlur(top_down, (5, 5), 0), cv2.COLOR_BGR2LAB)
    if floor_window_px > 0:
        floor = local_floor_color(lab, usable, floor_window_px, color_threshold, lightness_weight)
    else:
        floor = np.median(lab[usable], axis=0)
    distance = _color_distance(lab, floor, lightness_weight)

    mask = ((distance > color_threshold) & usable).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[labels] & usable


def grid_to_fusion(grid: np.ndarray) -> np.ndarray:
    """Convert the cell grid to the value convention of the map fusion (`merging_map`).

    1 = obstacle, 0 = free and seen, -1 = not seen. The rover's own cells become free, since
    the fusion takes the rover position separately, and margin cells become free, since the
    fusion grows obstacles by the robot radius itself. Rows stay as they are: row 0 is south.
    """
    fusion = np.zeros(grid.shape, dtype=np.int8)
    fusion[grid == OCCUPIED] = 1
    fusion[grid == UNKNOWN] = -1
    return fusion


def grid_to_image(grid: np.ndarray, cell_px: int = 12) -> np.ndarray:
    """Draw an occupancy grid as a BGR picture, shown the way the area looks from above.

    Free cells are white, obstacle cells grey, margin cells black, rover cells blue, unknown
    cells light blue, with a line between cells.
    """
    colors = np.full((*grid.shape, 3), 255, dtype=np.uint8)
    colors[grid == OCCUPIED] = (150, 150, 150)
    colors[grid == MARGIN] = (0, 0, 0)
    colors[grid == ROVER] = (255, 80, 0)
    colors[grid == UNKNOWN] = (230, 200, 150)
    image = cv2.resize(
        np.flipud(colors), None, fx=cell_px, fy=cell_px, interpolation=cv2.INTER_NEAREST
    )
    image[::cell_px, :] = (120, 120, 120)
    image[:, ::cell_px] = (120, 120, 120)
    return image


@dataclass(frozen=True)
class Blob:
    """One connected obstacle region, measured by its tightest rotated rectangle."""

    center: tuple[float, float]  # top-down pixels
    size: tuple[float, float]  # short and long side, metres
    angle: float  # direction of the long side in the top-down image, radians, modulo pi
    mask: np.ndarray  # bool mask of the region


def find_blobs(obstacles: np.ndarray, resolution: float) -> list[Blob]:
    """Measure every connected region of an obstacle mask."""
    count, labels = cv2.connectedComponents(obstacles.astype(np.uint8), connectivity=8)
    blobs = []
    for label in range(1, count):
        mask = labels == label
        points = cv2.findNonZero(mask.astype(np.uint8))
        center, (width, height), angle = cv2.minAreaRect(points)
        long_side_angle = np.deg2rad(angle if width >= height else angle + 90.0)
        short, long = sorted([(width + 1) * resolution, (height + 1) * resolution])
        blobs.append(Blob(center, (short, long), float(long_side_angle % np.pi), mask))
    return blobs


def find_rover_blob(
    blobs: list[Blob], rover_size: tuple[float, float], tolerance: float = 0.2
) -> Blob | None:
    """Pick the blob whose size matches the rover's footprint as seen from above.

    `rover_size` is the two sides in metres, in any order. A blob matches when both sides are
    within `tolerance` (a fraction) of the rover's. Returns the closest match, or None.
    """
    expected = sorted(rover_size)
    best, best_error = None, tolerance
    for blob in blobs:
        error = max(
            abs(side - want) / want for side, want in zip(blob.size, expected, strict=True)
        )
        if error <= best_error:
            best, best_error = blob, error
    return best


def mask_to_grid(obstacles: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Convert a top-down obstacle mask to an int8 grid of the same resolution.

    The top-down image has y up (first row is y_max). `nav_msgs/OccupancyGrid` is row-major
    starting at the origin (minimum x and y), so the rows are flipped.
    """
    grid = np.where(obstacles, OCCUPIED, FREE).astype(np.int8)
    if valid is not None:
        grid[~valid] = UNKNOWN
    return np.flipud(grid)


def margin_cells(margin: float, cell_size: float) -> int:
    """Number of whole cells that cover `margin` metres, rounded up."""
    return int(np.ceil(margin / cell_size - 1e-9))


def _any_per_cell(mask: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a bool mask by `factor`: a cell is True when any pixel inside it is."""
    rows = -(-mask.shape[0] // factor)
    cols = -(-mask.shape[1] // factor)
    padded = np.zeros((rows * factor, cols * factor), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    return padded.reshape(rows, factor, cols, factor).any(axis=(1, 3))


def build_cell_grid(
    obstacles: np.ndarray,
    valid: np.ndarray | None,
    resolution: float,
    cell_size: float,
    margin: float,
    rover: np.ndarray | None = None,
) -> np.ndarray:
    """Build the planner's occupancy grid of `cell_size` cells from the fine obstacle mask.

    A cell that an obstacle touches anywhere is OCCUPIED (100). Free cells within `margin`
    metres of an occupied cell, rounded up to whole cells, are MARGIN (99): with the margin
    set to half the rover's width, a planner can treat the rover as a point. Cells the camera
    does not see at all are UNKNOWN (-1). Row 0 is minimum y, as in `nav_msgs/OccupancyGrid`.

    `obstacles` must not include the rover. With `rover` (the rover's mask), the cells it
    touches are marked ROVER (50) unless an obstacle is in them too.
    """
    factor = int(round(cell_size / resolution))
    if factor < 1 or abs(factor * resolution - cell_size) > 1e-9:
        raise ValueError('cell_size must be a whole multiple of resolution')
    if valid is None:
        valid = np.ones(obstacles.shape, dtype=bool)
    # Flip first, so that cells line up with the map origin at the bottom-left.
    occupied = _any_per_cell(np.flipud(obstacles & valid), factor)
    seen = _any_per_cell(np.flipud(valid), factor)

    grid = np.full(occupied.shape, FREE, dtype=np.int8)
    grid[~seen] = UNKNOWN
    reach = margin_cells(margin, cell_size)
    if reach > 0 and occupied.any():
        distance = cv2.distanceTransform(
            (~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        grid[distance <= reach + 1e-6] = MARGIN
    if rover is not None:
        grid[_any_per_cell(np.flipud(rover), factor)] = ROVER
    grid[occupied] = OCCUPIED
    return grid
