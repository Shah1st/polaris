"""Pipeline options for a single overhead photo of an arena: thin floor marks, a rover whose
footprint does not match (cables), and a goal marker taped on the floor."""

import numpy as np
import pytest

from guidance_global_map.pipeline import OverheadSettings, process_image

PPM = 200  # pixels per metre of the synthetic photo: 5 mm per pixel
FLOOR = (170, 176, 182)


def photo(width=2.0, height=1.6):
    rng = np.random.default_rng(0)
    img = np.full((int(height * PPM), int(width * PPM), 3), FLOOR, np.uint8)
    return np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def rect(img, x0, y0, x1, y1, color):
    """Fill a rectangle given in map metres (origin bottom-left, y up)."""
    rows = img.shape[0]
    img[rows - int(y1 * PPM) : rows - int(y0 * PPM), int(x0 * PPM) : int(x1 * PPM)] = color


def settings(**kw):
    return OverheadSettings(metres_per_pixel=1.0 / PPM, **kw)


def big_blobs(result, min_side=0.05):
    return [b for b in result.blobs if min(b.size) >= min_side]


def test_thin_floor_line_is_dropped_but_boxes_stay() -> None:
    img = photo()
    rect(img, 0.0, 0.80, 2.0, 0.815, (190, 70, 20))  # 1.5 cm blue tape across the arena
    rect(img, 0.3, 1.0, 0.6, 1.3, (230, 230, 230))  # a box
    without = process_image(img, settings())
    with_width = process_image(img, settings(min_obstacle_width=0.03))
    assert any(max(b.size) > 1.5 for b in without.blobs)  # the line: a 2 m wall
    assert not any(max(b.size) > 1.5 for b in with_width.blobs)
    kept = big_blobs(with_width)
    assert len(kept) == 1  # the box is kept, about the same size
    assert kept[0].size == pytest.approx((0.3, 0.3), abs=0.02)


def test_rover_with_cables_is_found_from_the_start_point() -> None:
    img = photo()
    rect(img, 0.9, 0.3, 1.1, 0.53, (40, 40, 45))  # rover body 0.20 x 0.23 m
    rect(img, 0.95, 0.15, 1.08, 0.3, (40, 40, 45))  # cables and connectors behind it
    by_size = process_image(img, settings())
    assert by_size.rover_position is None  # 0.20 x 0.38 m: not the rover's size
    hinted = process_image(img, settings(rover_hint=(1.0, 0.415)))
    assert hinted.rover_position == (1.0, 0.415)
    # the rover's cells are not obstacles: the planner starts there
    grid = hinted.grid
    row, col = int(0.415 / 0.05), int(1.0 / 0.05)
    assert grid[row, col] != 100


def test_goal_marker_is_cleared() -> None:
    img = photo()
    rect(img, 1.45, 1.20, 1.55, 1.22, (40, 200, 230))  # yellow X, two strips
    rect(img, 1.49, 1.16, 1.51, 1.26, (40, 200, 230))
    seen = process_image(img, settings())
    cleared = process_image(img, settings(clear_areas=((1.5, 1.21, 0.10),)))
    row, col = int(1.21 / 0.05), int(1.5 / 0.05)
    assert seen.grid[row, col] == 100
    assert cleared.grid[row, col] == 0
