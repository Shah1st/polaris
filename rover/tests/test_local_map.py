"""The rolling obstacle grid (local_map.py): placing a picture's grid, memory, path check."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_map import FREE, OBSTACLE, UNKNOWN, LocalMap  # noqa: E402

RES = 0.05


def picture(obstacles=(), reach: float = 1.25):
    """A camera grid like d405_server's: x in [0, reach], y in [-reach, reach], floor seen
    free everywhere except `obstacles`, a list of (x, y) in the camera frame."""
    cols, half = round(reach / RES), round(reach / RES)
    cells = [[FREE] * cols for _ in range(2 * half)]
    for x, y in obstacles:
        cells[math.floor((y + reach) / RES)][math.floor(x / RES)] = OBSTACLE
    return cells, 0.0, -reach, RES


def value(m: LocalMap, x: float, y: float) -> int:
    return m.cells.get((math.floor(y / RES), math.floor(x / RES)), (UNKNOWN, 0.0))[0]


def test_a_picture_is_placed_with_the_pose_of_the_camera() -> None:
    m = LocalMap()
    m.integrate(*picture([(0.52, 0.02)]), (0.0, 0.0, 0.0), 1.0)
    assert value(m, 0.52, 0.02) == OBSTACLE
    assert value(m, 0.3, 0.02) == FREE
    assert value(m, -0.3, 0.0) == UNKNOWN  # behind the camera: never seen

    # the same picture from a rover at (1, 2) facing left (+y): the obstacle is 0.5 m up
    m = LocalMap()
    m.integrate(*picture([(0.52, 0.02)]), (1.0, 2.0, math.pi / 2), 1.0)
    assert value(m, 1.0 - 0.02, 2.52) == OBSTACLE
    assert value(m, 1.0, 1.7) == UNKNOWN  # behind the rover: outside the picture


@pytest.mark.parametrize('heading_deg', [0, 17, 45, 90, 133, 180, -60])
def test_a_rotated_picture_leaves_no_hole_and_loses_no_obstacle(heading_deg) -> None:
    h = math.radians(heading_deg)
    m = LocalMap()
    obstacles = [(0.42, -0.28), (0.82, 0.27), (1.02, 0.02)]
    m.integrate(*picture(obstacles), (0.0, 0.0, h), 1.0)
    for x, y in obstacles:
        # the centre of the picture's cell, turned into the map
        px = (math.floor(x / RES) + 0.5) * RES
        py = (math.floor((y + 1.25) / RES) + 0.5) * RES - 1.25
        gx, gy = px * math.cos(h) - py * math.sin(h), px * math.sin(h) + py * math.cos(h)
        assert value(m, gx, gy) == OBSTACLE, (x, y)
    # every map cell well inside the picture has a value
    for k in range(200):
        px, py = 0.1 + 1.0 * (k % 20) / 20, -0.9 + 1.8 * (k // 20) / 10
        gx, gy = px * math.cos(h) - py * math.sin(h), px * math.sin(h) + py * math.cos(h)
        if abs(gx) < 1.2 and abs(gy) < 1.2:  # the map keeps 1.25 m around the rover
            assert value(m, gx, gy) != UNKNOWN


def test_seen_free_erases_an_obstacle_and_unseen_cells_expire() -> None:
    m = LocalMap(hold=5.0, near=0.45)
    m.integrate(*picture([(0.82, 0.02)]), (0.0, 0.0, 0.0), 0.0)
    assert value(m, 0.82, 0.02) == OBSTACLE
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.0)  # looked again: it left
    assert value(m, 0.82, 0.02) == FREE

    # an obstacle the camera no longer looks at (the rover turned away) is forgotten after hold
    m = LocalMap(hold=5.0, near=0.45)
    m.integrate(*picture([(0.82, 0.02)]), (0.0, 0.0, 0.0), 0.0)
    m.integrate(*picture(), (0.0, 0.0, math.pi), 4.0)
    assert value(m, 0.82, 0.02) == OBSTACLE
    m.integrate(*picture(), (0.0, 0.0, math.pi), 6.0)
    assert value(m, 0.82, 0.02) == UNKNOWN


def test_cells_under_the_nose_are_kept_and_far_ones_dropped() -> None:
    m = LocalMap(size=2.5, hold=5.0, near=0.45)
    m.integrate(*picture([(0.32, 0.02), (1.02, 0.02)]), (0.0, 0.0, 0.0), 0.0)
    m.integrate(*picture(), (0.0, 0.0, math.pi), 60.0)
    assert value(m, 0.32, 0.02) == OBSTACLE  # too close for the camera to clear: remembered
    assert value(m, 1.02, 0.02) == UNKNOWN
    # the rover drove 3 m away: nothing of the old place is left
    m.integrate(*picture(), (3.0, 0.0, 0.0), 61.0)
    assert value(m, 0.32, 0.02) == UNKNOWN


def test_first_obstacle_along_a_path() -> None:
    m = LocalMap()
    m.integrate(*picture([(0.62, 0.02)]), (0.0, 0.0, 0.0), 0.0)
    line = [(0.02 * k, 0.0) for k in range(60)]
    dist, (ox, oy) = m.first_obstacle(line, 0.12, 1.0)
    # the cell centre is at 0.625; the rover's centre comes within 0.12 + half a cell of it
    assert dist == pytest.approx(0.625 - 0.145, abs=0.03)
    assert (ox, oy) == pytest.approx((0.625, 0.025))
    assert m.first_obstacle(line, 0.12, 0.3) is None  # farther than what is looked at
    beside = [(0.02 * k, 0.25) for k in range(60)]
    assert m.first_obstacle(beside, 0.12, 1.0) is None  # 25 cm to the side: it passes
    assert m.first_obstacle(beside, 0.25, 1.0) is not None
    assert m.first_obstacle([], 0.12, 1.0) is None


def test_snapshot_is_a_square_around_the_rover() -> None:
    m = LocalMap(size=2.5)
    m.integrate(*picture([(0.52, 0.02)]), (1.0, 2.0, 0.0), 0.0)
    snap = m.snapshot((1.0, 2.0))
    assert len(snap['cells']) == 50 and len(snap['cells'][0]) == 50
    assert snap['x0'] == pytest.approx(1.0 - 1.25) and snap['y0'] == pytest.approx(2.0 - 1.25)
    i = math.floor((2.02 - snap['y0']) / RES)
    j = math.floor((1.52 - snap['x0']) / RES)
    assert snap['cells'][i][j] == OBSTACLE
    assert m.counts()['obstacle'] == 1 and m.counts()['free'] > 500
    m.clear()
    assert m.age(1.0) is None and m.counts() == {'obstacle': 0, 'free': 0}
