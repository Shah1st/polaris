import numpy as np
import pytest

from guidance_calib.aruco_compat import detect_markers
from guidance_calib.ground_plane import (
    estimate_homography,
    map_to_top_down,
    pixels_to_map,
    warp_to_top_down,
)
from guidance_calib.synthetic import Box, Scene, render_camera_view
from guidance_global_map.segmentation import (
    FREE,
    MARGIN,
    OCCUPIED,
    ROVER,
    UNKNOWN,
    build_cell_grid,
    find_blobs,
    find_rover_blob,
    grid_to_fusion,
    grid_to_image,
    margin_cells,
    mask_to_grid,
    polygons_mask,
    scale_polygon,
    segment_obstacles,
)

RESOLUTION = 0.01


def grid_from_scene(
    scene: Scene, mask_markers: bool = True, floor_window_px: int = 100
) -> np.ndarray:
    """Run the same steps as global_map_node on a synthetic camera image."""
    image, _ = render_camera_view(scene)
    detections = detect_markers(image)
    homography = estimate_homography(detections, scene.marker_positions)
    top_down, valid = warp_to_top_down(image, homography, scene.bounds, RESOLUTION)
    ignore = None
    if mask_markers:
        polygons = [
            scale_polygon(
                map_to_top_down(pixels_to_map(homography, corners), scene.bounds, RESOLUTION), 1.8
            )
            for corners in detections.values()
        ]
        ignore = polygons_mask(valid.shape, polygons)
    obstacles = segment_obstacles(top_down, valid, ignore, floor_window_px=floor_window_px)
    return mask_to_grid(obstacles, valid)


def cell(grid: np.ndarray, x: float, y: float) -> int:
    """Occupancy value at map position (x, y), indexed the way OccupancyGrid is."""
    return int(grid[int(y / RESOLUTION), int(x / RESOLUTION)])


def test_boxes_are_occupied_and_floor_is_free() -> None:
    scene = Scene(
        boxes=[
            Box(0.50, 0.30, 0.30, 0.20),
            Box(1.20, 0.90, 0.25, 0.35, color=(60, 140, 40)),
            Box(0.90, 0.55, 0.15, 0.15, color=(30, 30, 30)),
        ]
    )

    grid = grid_from_scene(scene)

    assert grid.shape == (150, 200)
    assert cell(grid, 0.65, 0.40) == OCCUPIED
    assert cell(grid, 1.32, 1.07) == OCCUPIED
    assert cell(grid, 0.97, 0.62) == OCCUPIED
    assert cell(grid, 1.60, 0.30) == FREE
    assert cell(grid, 0.20, 0.80) == FREE
    assert cell(grid, 1.00, 1.30) == FREE
    occupied_area = (grid == OCCUPIED).sum() * RESOLUTION**2
    true_area = sum(box.width * box.height for box in scene.boxes)
    assert abs(occupied_area - true_area) < 0.25 * true_area


def test_masked_markers_leave_an_empty_map() -> None:
    grid = grid_from_scene(Scene(marker_side=0.30, rover_pose=(1.0, 0.75, 0.0)))

    assert (grid == FREE).all()


def test_markers_are_obstacles_unless_masked() -> None:
    scene = Scene(marker_side=0.30, rover_pose=(1.0, 0.75, 0.0))

    grid = grid_from_scene(scene, mask_markers=False)

    assert (grid[60:90, 85:115] == OCCUPIED).any()


def test_small_specks_are_dropped() -> None:
    top_down = np.full((100, 100, 3), 180, dtype=np.uint8)
    top_down[10:13, 10:13] = (0, 0, 255)
    top_down[40:70, 40:70] = (0, 0, 255)

    obstacles = segment_obstacles(top_down, min_area_px=50)

    assert not obstacles[11, 11]
    assert obstacles[55, 55]


def test_grid_rows_are_flipped_and_uncovered_cells_are_unknown() -> None:
    obstacles = np.zeros((4, 3), dtype=bool)
    obstacles[0, 1] = True  # top row of the top-down view = maximum y
    valid = np.ones((4, 3), dtype=bool)
    valid[3, 0] = False

    grid = mask_to_grid(obstacles, valid)

    assert grid.dtype == np.int8
    assert grid[3, 1] == OCCUPIED
    assert grid[0, 0] == UNKNOWN
    assert grid[1, 1] == FREE


def mask_with_rectangles(rectangles: list) -> np.ndarray:
    """Bool mask, 5 mm per pixel, with filled (x, y, width, height) rectangles in metres."""
    mask = np.zeros((300, 300), dtype=bool)
    for x, y, width, height in rectangles:
        r0, c0 = int(y / 0.005), int(x / 0.005)
        mask[r0 : r0 + int(height / 0.005), c0 : c0 + int(width / 0.005)] = True
    return mask


def test_blobs_are_measured_in_metres() -> None:
    blobs = find_blobs(mask_with_rectangles([(0.2, 0.3, 0.40, 0.10)]), 0.005)

    assert len(blobs) == 1
    assert blobs[0].size == pytest.approx((0.10, 0.40), abs=0.006)
    assert blobs[0].center == pytest.approx((79.5, 69.5), abs=0.6)


def test_rover_is_the_blob_of_the_rover_size() -> None:
    obstacles = mask_with_rectangles(
        [(0.1, 0.1, 0.13, 0.15), (0.6, 0.2, 0.27, 0.36), (0.5, 1.0, 0.21, 0.24)]
    )
    blobs = find_blobs(obstacles, 0.005)

    rover = find_rover_blob(blobs, (0.20, 0.23), tolerance=0.2)

    assert rover is not None
    assert rover.size == pytest.approx((0.21, 0.24), abs=0.006)
    assert rover.mask[220, 120]
    assert not rover.mask[30, 30]


def test_no_rover_when_nothing_has_its_size() -> None:
    blobs = find_blobs(
        mask_with_rectangles([(0.1, 0.1, 0.13, 0.15), (0.6, 0.2, 0.27, 0.36)]), 0.005
    )

    assert find_rover_blob(blobs, (0.20, 0.23), tolerance=0.2) is None
    assert find_rover_blob([], (0.20, 0.23)) is None


def test_a_cell_touched_by_an_obstacle_is_occupied() -> None:
    # 5 mm pixels, 2.5 cm cells: 5 x 5 pixels per cell. One obstacle pixel in the top-left.
    obstacles = np.zeros((20, 20), dtype=bool)
    obstacles[0, 0] = True

    grid = build_cell_grid(obstacles, None, resolution=0.005, cell_size=0.025, margin=0.0)

    assert grid.shape == (4, 4)
    assert grid[3, 0] == OCCUPIED  # top-left of the view is the last row: row 0 is y = 0
    assert (grid == OCCUPIED).sum() == 1
    assert (grid == FREE).sum() == 15


def test_margin_is_rounded_up_to_whole_cells() -> None:
    assert margin_cells(0.115, 0.025) == 5
    assert margin_cells(0.10, 0.025) == 4
    assert margin_cells(0.101, 0.025) == 5
    assert margin_cells(0.0, 0.025) == 0

    obstacles = np.zeros((100, 100), dtype=bool)
    obstacles[52, 52] = True  # inside cell (row 10, col 10) of the top-down view

    grid = build_cell_grid(obstacles, None, resolution=0.005, cell_size=0.025, margin=0.115)

    row = grid[9]  # grid row 9 is top-down cell row 10
    assert row[10] == OCCUPIED
    assert (row[5:10] == MARGIN).all() and (row[11:16] == MARGIN).all()
    assert row[4] == FREE and row[16] == FREE
    assert grid[4, 10] == MARGIN and grid[3, 10] == FREE


def test_cells_the_camera_does_not_see_are_unknown() -> None:
    obstacles = np.zeros((20, 20), dtype=bool)
    valid = np.ones((20, 20), dtype=bool)
    valid[:, 15:] = False  # right-hand column of cells not covered
    valid[0:5, 0:4] = False  # one cell only partly covered: still known

    grid = build_cell_grid(obstacles, valid, resolution=0.005, cell_size=0.025, margin=0.05)

    assert (grid[:, 3] == UNKNOWN).all()
    assert (grid[:, :3] == FREE).all()


def test_cell_size_must_be_a_multiple_of_the_resolution() -> None:
    with pytest.raises(ValueError, match='multiple'):
        build_cell_grid(np.zeros((10, 10), dtype=bool), None, 0.005, 0.012, 0.0)


def test_grid_picture_is_upright_and_coloured_like_the_legend() -> None:
    grid = np.array([[100, 0], [99, -1]], dtype=np.int8)  # row 0 is y = 0, the bottom

    image = grid_to_image(grid, cell_px=10)

    assert image.shape == (20, 20, 3)
    assert tuple(image[15, 5]) == (150, 150, 150)  # obstacle cell, bottom-left: grey
    assert tuple(image[15, 15]) == (255, 255, 255)  # free: white
    assert tuple(image[5, 5]) == (0, 0, 0)  # margin: black
    assert tuple(image[5, 15]) == (230, 200, 150)  # unknown: light blue
    assert tuple(image[10, 5]) == (120, 120, 120)  # line between cells


def test_rover_cells_are_marked_but_never_hide_an_obstacle() -> None:
    obstacles = np.zeros((20, 20), dtype=bool)
    obstacles[0, 9] = True  # top-down cell (0, 1)
    rover = np.zeros((20, 20), dtype=bool)
    rover[0:5, 0:10] = True  # top-down cells (0, 0) and (0, 1)

    grid = build_cell_grid(obstacles, None, 0.005, 0.025, margin=0.0, rover=rover)

    assert grid[3, 0] == ROVER
    assert grid[3, 1] == OCCUPIED
    assert (grid == ROVER).sum() == 1
    assert tuple(grid_to_image(grid, cell_px=10)[5, 5]) == (255, 80, 0)


def test_result_does_not_depend_on_unseen_area_around_the_view() -> None:
    rng = np.random.default_rng(1)
    view = np.clip(rng.normal(150, 6, (120, 160, 3)), 0, 255).astype(np.uint8)
    view[:, :40] = (view[:, :40] * 0.6).astype(np.uint8)  # darker region up to the left edge
    view[50:70, 100:130] = (0, 0, 255)
    view[:, 150:] = 230  # light strip along the right edge
    alone = segment_obstacles(view, np.ones(view.shape[:2], dtype=bool), floor_window_px=60)

    padded = np.zeros((200, 260, 3), dtype=np.uint8)
    padded[30:150, 20:180] = view
    valid = np.zeros(padded.shape[:2], dtype=bool)
    valid[30:150, 20:180] = True
    inside = segment_obstacles(padded, valid, floor_window_px=60)

    assert alone[60, 115]
    assert np.array_equal(inside[30:150, 20:180], alone)
    assert not inside[~valid].any()


def test_fusion_convention_keeps_obstacles_and_unseen_and_frees_the_rest() -> None:
    grid = np.array([[OCCUPIED, FREE, UNKNOWN], [ROVER, MARGIN, OCCUPIED]], dtype=np.int8)

    fusion = grid_to_fusion(grid)

    assert fusion.dtype == np.int8
    assert fusion.tolist() == [[1, 0, -1], [0, 0, 1]]
