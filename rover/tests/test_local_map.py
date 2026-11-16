"""The rolling obstacle grid (local_map.py): placing a picture's grid, confirmation over
several pictures, forgetting by the clock, path check."""

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


def look(m: LocalMap, obstacles, pose, t: float, times: int = 2) -> None:
    """`times` pictures in a row of the same scene, 0.1 s apart, ending at `t`."""
    for k in range(times):
        m.integrate(*picture(obstacles), pose, t - 0.1 * (times - 1 - k))


def value(m: LocalMap, x: float, y: float) -> int:
    cell = m.cells.get((math.floor(y / RES), math.floor(x / RES)))
    return UNKNOWN if cell is None else cell[0]


def line(y: float = 0.0, n: int = 60):
    return [(0.02 * k, y) for k in range(n)]


def test_a_picture_is_placed_with_the_pose_of_the_camera() -> None:
    m = LocalMap()
    look(m, [(0.52, 0.02)], (0.0, 0.0, 0.0), 1.0)
    assert value(m, 0.52, 0.02) == OBSTACLE
    assert value(m, 0.3, 0.02) == FREE
    assert value(m, -0.3, 0.0) == UNKNOWN  # behind the camera: never seen

    # the same picture from a rover at (1, 2) facing left (+y): the obstacle is 0.5 m up
    m = LocalMap()
    look(m, [(0.52, 0.02)], (1.0, 2.0, math.pi / 2), 1.0)
    assert value(m, 1.0 - 0.02, 2.52) == OBSTACLE
    assert value(m, 1.0, 1.7) == UNKNOWN  # behind the rover: outside the picture


@pytest.mark.parametrize('heading_deg', [0, 17, 45, 90, 133, 180, -60])
def test_a_rotated_picture_leaves_no_hole_and_loses_no_obstacle(heading_deg) -> None:
    h = math.radians(heading_deg)
    m = LocalMap()
    obstacles = [(0.42, -0.28), (0.82, 0.27), (1.02, 0.02)]
    look(m, obstacles, (0.0, 0.0, h), 1.0)
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


# -- confirmation: one picture decides nothing ------------------------------------------


def test_a_speck_in_one_picture_is_not_an_obstacle() -> None:
    m = LocalMap()
    look(m, [], (0.0, 0.0, 0.0), 1.0)  # the floor, seen free
    m.integrate(*picture([(0.62, 0.02)]), (0.0, 0.0, 0.0), 1.1)  # noise, once
    assert value(m, 0.62, 0.02) == FREE
    assert m.first_obstacle(line(), 0.10, 1.0, 1.1) is None
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.2)
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.3)
    m.integrate(*picture([(0.62, 0.02)]), (0.0, 0.0, 0.0), 1.4)  # again, later: one in three
    assert value(m, 0.62, 0.02) == FREE

    # in a cell never seen before, one hit leaves it "not seen", not an obstacle
    m = LocalMap()
    m.integrate(*picture([(0.62, 0.02)]), (0.0, 0.0, 0.0), 1.0)
    assert value(m, 0.62, 0.02) == UNKNOWN and m.counts((0.0, 0.0), 1.0)['obstacle'] == 0


def test_two_pictures_out_of_three_make_an_obstacle_and_two_free_ones_clear_it() -> None:
    m = LocalMap()
    box = [(0.62, 0.02)]
    m.integrate(*picture(box), (0.0, 0.0, 0.0), 1.0)
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.1)  # missed once (flicker)
    assert value(m, 0.62, 0.02) != OBSTACLE
    m.integrate(*picture(box), (0.0, 0.0, 0.0), 1.2)
    assert value(m, 0.62, 0.02) == OBSTACLE  # two of the last three
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.3)
    assert value(m, 0.62, 0.02) == OBSTACLE  # one free look does not clear it
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.4)
    assert value(m, 0.62, 0.02) == FREE  # two do: it left


def test_confirm_1_takes_every_picture_at_its_word() -> None:
    m = LocalMap(confirm=1)
    m.integrate(*picture([(0.62, 0.02)]), (0.0, 0.0, 0.0), 1.0)
    assert value(m, 0.62, 0.02) == OBSTACLE
    m.integrate(*picture(), (0.0, 0.0, 0.0), 1.1)
    assert value(m, 0.62, 0.02) == FREE


# -- forgetting, by the clock -----------------------------------------------------------


def test_an_obstacle_no_longer_looked_at_is_forgotten_after_hold() -> None:
    m = LocalMap(hold=10.0)
    look(m, [(0.82, 0.02)], (0.0, 0.0, 0.0), 0.0)
    look(m, [], (0.0, 0.0, math.pi), 9.0)  # the rover turned away: it looks elsewhere
    assert value(m, 0.82, 0.02) == OBSTACLE
    assert m.first_obstacle(line(), 0.10, 1.0, 9.0) is not None
    look(m, [], (0.0, 0.0, math.pi), 10.5)
    assert value(m, 0.82, 0.02) == UNKNOWN
    assert m.first_obstacle(line(), 0.10, 1.0, 10.5) is None


def test_forgetting_does_not_need_pictures_to_keep_coming() -> None:
    """The capture server stops: what was seen still expires, for every reader."""
    m = LocalMap(hold=10.0)
    look(m, [(0.82, 0.02)], (0.0, 0.0, 0.0), 0.0)
    assert m.first_obstacle(line(), 0.10, 1.0, 5.0) is not None
    assert m.first_obstacle(line(), 0.10, 1.0, 10.5) is None  # no integrate() in between
    m = LocalMap(hold=10.0)
    look(m, [(0.82, 0.02)], (0.0, 0.0, 0.0), 0.0)
    assert m.counts((0.0, 0.0), 5.0)['obstacle'] == 1
    assert m.counts((0.0, 0.0), 10.5)['obstacle'] == 0
    assert m.counts((0.0, 0.0), 31.0) == {'obstacle': 0, 'free': 0}  # under the nose too
    m = LocalMap(hold=10.0)
    look(m, [(0.82, 0.02)], (0.0, 0.0, 0.0), 0.0)
    snap = m.snapshot((0.0, 0.0), 10.5)
    assert all(v != OBSTACLE for row in snap['cells'] for v in row)
    snap = m.snapshot((0.0, 0.0), 31.0)
    assert all(v == UNKNOWN for row in snap['cells'] for v in row)


def test_cells_under_the_nose_are_kept_longer_but_not_for_ever() -> None:
    m = LocalMap(size=2.5, hold=10.0, near=0.45, hold_near=30.0)
    look(m, [(0.32, 0.02), (1.02, 0.02)], (0.0, 0.0, 0.0), 0.0)
    look(m, [], (0.0, 0.0, math.pi), 20.0)
    assert value(m, 0.32, 0.02) == OBSTACLE  # too close for the camera to clear: remembered
    assert value(m, 1.02, 0.02) == UNKNOWN
    look(m, [], (0.0, 0.0, math.pi), 31.0)
    assert value(m, 0.32, 0.02) == UNKNOWN  # ... for hold_near, then gone too
    # the rover drove 3 m away: nothing of the old place is left
    m = LocalMap()
    look(m, [(0.32, 0.02)], (0.0, 0.0, 0.0), 0.0)
    look(m, [], (3.0, 0.0, 0.0), 1.0)
    assert value(m, 0.32, 0.02) == UNKNOWN


# -- reading ----------------------------------------------------------------------------


def test_first_obstacle_along_a_path() -> None:
    m = LocalMap()
    look(m, [(0.62, 0.02)], (0.0, 0.0, 0.0), 0.0)
    dist, (ox, oy) = m.first_obstacle(line(), 0.10, 1.0, 0.0)
    # the cell's centre is at 0.625: the rover's centre comes within 0.10 of it
    assert dist == pytest.approx(0.625 - 0.10, abs=0.03)
    assert (ox, oy) == pytest.approx((0.625, 0.025))
    assert m.first_obstacle(line(), 0.10, 0.3, 0.0) is None  # farther than what is looked at
    assert m.first_obstacle(line(0.15), 0.10, 1.0, 0.0) is None  # 15 cm to the side: it passes
    assert m.first_obstacle(line(0.15), 0.25, 1.0, 0.0) is not None
    assert m.first_obstacle([], 0.10, 1.0, 0.0) is None


def test_snapshot_is_a_square_around_the_rover() -> None:
    m = LocalMap(size=2.5)
    look(m, [(0.52, 0.02)], (1.0, 2.0, 0.0), 0.0)
    snap = m.snapshot((1.0, 2.0), 0.0)
    assert len(snap['cells']) == 50 and len(snap['cells'][0]) == 50
    assert snap['x0'] == pytest.approx(1.0 - 1.25) and snap['y0'] == pytest.approx(2.0 - 1.25)
    i = math.floor((2.02 - snap['y0']) / RES)
    j = math.floor((1.52 - snap['x0']) / RES)
    assert snap['cells'][i][j] == OBSTACLE
    counts = m.counts((1.0, 2.0), 0.0)
    assert counts['obstacle'] == 1 and counts['free'] > 500
    m.clear()
    assert m.age(1.0) is None and m.counts((1.0, 2.0), 0.0) == {'obstacle': 0, 'free': 0}
