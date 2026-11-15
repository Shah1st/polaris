"""Tests of the ROS <-> map fusion bridge. No ROS needed: numpy and merging_map/ only."""

import math

import numpy as np

from guidance_supervisor.fusion_bridge import (
    decision_summary,
    fusion_to_ros_grid,
    grid_area,
    heading_from_pose,
    load_merging_map,
    ros_grid_to_fusion,
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
