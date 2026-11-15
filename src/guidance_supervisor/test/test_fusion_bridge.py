"""Tests of the ROS <-> map fusion bridge. No ROS needed: numpy and merging_map/ only."""

import math

import numpy as np
import pytest

from guidance_supervisor.fusion_bridge import (
    decision_summary,
    fusion_to_ros_grid,
    grid_area,
    heading_from_pose,
    load_merging_map,
    ros_grid_to_fusion,
    seen_area,
    tiles_from_ros_grid,
)

load_merging_map()
from fusion_carte import CarteFusion  # noqa: E402


def test_overhead_values_become_fusion_values() -> None:
    # row 0 (south): unknown, free, obstacle; row 1: rover, margin, 80 (another map's obstacle)
    data = [-1, 0, 100, 50, 99, 80]
    assert ros_grid_to_fusion(data, 3, 2).tolist() == [[-1, 0, 1], [0, 0, 1]]


def test_tiles_land_where_the_grid_is() -> None:
    """A 2.4 m x 1.2 m /map with one obstacle cell at (1.0, 0.5): after going through the
    tiles, the fusion sees it at the same place, and nothing outside the grid."""
    width, height, res = 48, 24, 0.05
    data = np.zeros((height, width), np.int16)
    data[10, 20] = 100  # x = 1.0..1.05, y = 0.5..0.55
    carte = CarteFusion()
    for tile in tiles_from_ros_grid(data.ravel().tolist(), width, height, res, 0.0, 0.0, carte):
        carte.recevoir(tile)
    g = carte.grille(*grid_area(width, height, res, 0.0, 0.0))
    assert g.val.shape == (height, width)
    assert g.val[10, 20] == 1 and (g.val == 1).sum() == 1
    assert (g.val == 0).sum() == width * height - 1
    assert (carte.grille(2.4, 0.0, 3.2, 1.2).val == -1).all()  # not seen beyond the grid


def test_wrong_cell_size_is_refused() -> None:
    try:
        tiles_from_ros_grid([0] * 4, 2, 2, 0.1, 0.0, 0.0, CarteFusion())
    except ValueError as e:
        assert 'cell_size' in str(e)
    else:
        raise AssertionError('a 10 cm grid should be refused')


def test_heading_is_none_when_its_variance_says_unknown() -> None:
    assert heading_from_pose(0.0, 1.0, 1e6) is None  # what the overhead node sends today
    yaw = math.radians(30)
    assert math.isclose(heading_from_pose(math.sin(yaw / 2), math.cos(yaw / 2), 0.01), yaw)


def test_merged_grid_goes_back_to_ros_values() -> None:
    assert fusion_to_ros_grid(np.array([[-1, 0], [1, 0]])) == [-1, 0, 100, 0]


def test_the_bridge_feeds_a_plan() -> None:
    """End to end without ROS: /map -> tiles -> fusion planner -> JSON summary."""
    width = height = 48
    data = np.zeros((height, width), np.int16)
    data[18:30, 20:28] = 100  # a box between start and goal
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for tile in tiles_from_ros_grid(data.ravel().tolist(), width, height, 0.05, 0.0, 0.0, carte):
        carte.recevoir(tile)
    zone = grid_area(width, height, 0.05, 0.0, 0.0)
    decision = carte.decider((0.4, 0.4), (2.0, 2.0), zone=zone, tolerance=0.15)
    summary = decision_summary(decision)
    assert summary['action'] == 'SUIVRE' and summary['path_length'] > math.dist((0.4, 0.4), (2, 2))
    assert 'chemin' not in summary  # no big arrays in the message


def test_planning_area_is_where_the_camera_saw() -> None:
    """A 2 m x 2 m /map of which the camera saw only x 0..1.6, y 0..1.75: the rest is unknown
    and must stay out of the planning area."""
    width = height = 40
    grid = np.full((height, width), -1, np.int8)
    grid[:35, :32] = 0
    grid[10, 10] = 100

    area = seen_area(grid.ravel().tolist(), width, height, 0.05, 0.0, 0.0)

    assert area == pytest.approx((0.0, 0.0, 1.6, 1.75))
    assert grid_area(width, height, 0.05, 0.0, 0.0) == pytest.approx((0.0, 0.0, 2.0, 2.0))


def test_planning_area_follows_the_grid_origin_and_is_none_when_nothing_is_seen() -> None:
    grid = np.full((4, 6), -1, np.int8)
    assert seen_area(grid.ravel().tolist(), 6, 4, 0.05, 1.0, 2.0) is None
    grid[1:3, 2:5] = 0
    assert seen_area(grid.ravel().tolist(), 6, 4, 0.05, 1.0, 2.0) == pytest.approx(
        (1.10, 2.05, 1.25, 2.15)
    )


def test_planner_stays_inside_the_seen_area() -> None:
    """Goal behind a wall that spans the camera's view: on the whole 2 m grid the planner
    goes around through unknown cells; inside the seen area it reports no way through."""
    width = height = 40
    grid = np.full((height, width), -1, np.int8)
    grid[:, :24] = 0  # the camera saw x 0..1.2
    grid[18:22, :24] = 100  # a wall across all of it, y 0.9..1.1
    data = grid.ravel().tolist()
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    for tile in tiles_from_ros_grid(data, width, height, 0.05, 0.0, 0.0, carte):
        carte.recevoir(tile)
    start, goal = (0.6, 0.4), (0.6, 1.6)
    whole_grid = grid_area(width, height, 0.05, 0.0, 0.0)
    seen = seen_area(data, width, height, 0.05, 0.0, 0.0)

    on_whole_grid = carte.decider(start, goal, zone=whole_grid, tolerance=0.15)
    on_seen_area = carte.decider(start, goal, zone=seen, tolerance=0.15)

    assert on_whole_grid['action'] == 'SUIVRE'
    assert max(x for x, _ in on_whole_grid['chemin']) > 1.2
    assert on_seen_area['action'] != 'SUIVRE'
