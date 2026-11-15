"""The route around obstacles (the team's A*, merging_map/fusion_carte.py) and the guide that
drives it, replanning when the map blocks the way."""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from guidance import Pose2  # noqa: E402
from guide_rover import GuideService  # noqa: E402
from planner import RoutePlanner  # noqa: E402
from test_guidance import FakeRover, sighting  # noqa: E402

RES = 0.05
X0, Y0 = -1.03, -0.98  # a map placed anywhere: not a multiple of the cell


def grid_with(*boxes, known=True) -> dict:
    """A 2 m map, row 0 = south; each box (x0, y0, x1, y1) in metres is an obstacle."""
    n = round(2.0 / RES)
    cells = np.zeros((n, n), np.int8) if known else np.full((n, n), -1, np.int8)
    for bx0, by0, bx1, by1 in boxes:
        j0, j1 = math.floor((bx0 - X0) / RES), math.ceil((bx1 - X0) / RES)
        i0, i1 = math.floor((by0 - Y0) / RES), math.ceil((by1 - Y0) / RES)
        cells[i0:i1, j0:j1] = 1
    return {'res': RES, 'x0': X0, 'y0': Y0, 'cells': cells.tolist()}


def clearance(points, grid) -> float:
    """The smallest distance (m) from the segments joining `points` to an obstacle cell."""
    cells = np.array(grid['cells'])
    ii, jj = np.nonzero(cells == 1)
    cx0 = grid['x0'] + jj * RES
    cy0 = grid['y0'] + ii * RES
    best = math.inf
    for a, b in zip(points, points[1:], strict=False):
        for t in np.linspace(0, 1, max(2, int(math.dist(a, b) / 0.01))):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            dx = np.maximum(np.maximum(cx0 - x, x - (cx0 + RES)), 0)
            dy = np.maximum(np.maximum(cy0 - y, y - (cy0 + RES)), 0)
            best = min(best, float(np.hypot(dx, dy).min()))
    return best


WALL = (-1.0, -0.05, 0.45, 0.05)  # a wall across the map, open on its east side


def test_route_goes_through_the_gap_and_keeps_the_radius() -> None:
    grid = grid_with(WALL)
    planner = RoutePlanner(radius=0.15)
    start, goal = (-0.5, -0.6), (-0.5, 0.6)
    route = planner.route(grid, start, goal)
    assert route is not None and route[-1] == pytest.approx(goal)
    assert max(x for x, _ in route) > 0.45 + 0.15  # around the wall's end
    assert clearance([start, *route], grid) >= 0.15 - 0.01
    assert len(route) <= 4  # a few corners, not a point every 5 cm
    assert planner.blocked(grid, route, start) is None
    # a straight line is fine when nothing is in the way
    assert planner.route(grid_with(), start, goal) == [pytest.approx(goal)]


def test_no_route_into_an_obstacle_or_through_a_closed_wall() -> None:
    planner = RoutePlanner(radius=0.15)
    assert planner.route(grid_with(WALL), (-0.5, -0.6), (-0.5, 0.0)) is None  # in the wall
    closed = grid_with((-1.0, -0.05, 1.0, 0.05))
    assert planner.route(closed, (-0.5, -0.6), (-0.5, 0.6)) is None
    # unknown floor may be crossed (at a higher cost): nothing seen is not a wall
    assert planner.route(grid_with(known=False), (-0.5, -0.6), (-0.5, 0.6)) is not None


def test_a_new_obstacle_on_the_route_is_found() -> None:
    planner = RoutePlanner(radius=0.15)
    start, goal = (-0.5, -0.6), (-0.5, 0.6)
    route = planner.route(grid_with(), start, goal)
    hit = planner.blocked(grid_with((-0.7, -0.1, -0.3, 0.1)), route, start)
    assert hit is not None
    # a rover standing inside a grown zone may leave it: not reported as blocked at once
    near = (-0.5, -0.25)  # 0.15 m from the new box: inside its grown zone
    away = planner.route(grid_with((-0.7, -0.1, -0.3, 0.1)), near, (-0.5, -0.8))
    assert away is not None


class FixedMap:
    """Stands for the map worker: the grid the test chooses."""

    def __init__(self, grid):
        self.current = grid

    def grid(self):
        return self.current

    def stats(self):
        return {}

    def close(self):
        pass


def drive(service: GuideService, rover: FakeRover, seconds: float, on_tick=None):
    """The guide's loop in simulated time (state 10 Hz, pictures 15 Hz, plans 2 Hz)."""
    rng = random.Random(4)
    statuses = []
    for k in range(1, round(seconds / 0.02) + 1):
        rover.tick(0.02)
        t = rover.t
        if on_tick:
            on_tick(t)
        if k % 5 == 0:
            service.guide.on_state(rover.state(), t)
        if k % 3 == 0:
            service.guide.on_sighting(sighting(rover.truth_at(t - 0.15), rng), t - 0.15)
        if k % 25 == 0:
            service._replan()
        if k % 5 == 0:
            s = service.guide.step(t)
            if not statuses or statuses[-1] != s:
                statuses.append(s)
    return statuses


@pytest.fixture
def service():
    s = GuideService('http://127.0.0.1:9', FakeRover(Pose2(-0.5, -0.6, math.radians(80))),
                     finder=None, port=0)  # fmt: skip
    s.stream.close()
    yield s
    s.server.server_close()


def test_the_rover_drives_around_the_wall(service) -> None:
    rover = service.rover
    service.mapper = FixedMap(grid_with(WALL))
    drive(service, rover, 1.0)  # the rover is seen, the link is placed
    service.set_goal({'x': -0.5, 'y': 0.6})
    statuses = drive(service, rover, 40.0)
    assert statuses[-1].startswith('arrived'), statuses
    trail = [(p.x, p.y) for _, p in rover.trail]
    assert math.dist(trail[-1], (-0.5, 0.6)) < 0.06
    assert clearance(trail[::5], grid_with(WALL)) > 0.08  # never through the wall
    assert max(x for x, _ in trail) > 0.45  # it went round


def test_an_obstacle_appearing_on_the_way_is_avoided(service) -> None:
    rover = service.rover
    service.mapper = FixedMap(grid_with())
    drive(service, rover, 1.0)
    service.set_goal({'x': -0.5, 'y': 0.6})
    box = (-0.75, -0.1, -0.25, 0.15)

    def drop_box(t):  # a box is put on the straight line, 30 cm ahead of the rover
        if t > 2.0 and service.mapper.current is not boxed:
            service.mapper.current = boxed

    boxed = grid_with(box)
    statuses = drive(service, rover, 40.0, drop_box)
    assert statuses[-1].startswith('arrived'), statuses
    assert service.plans >= 2  # planned for the goal, then again around the box
    late = [(p.x, p.y) for t, p in rover.trail if t > 2.6]  # once the new route is out
    assert clearance(late[::5], boxed) > 0.10


def test_a_goal_behind_a_closed_wall_is_refused(service) -> None:
    rover = service.rover
    service.mapper = FixedMap(grid_with((-1.0, -0.05, 1.0, 0.05)))
    drive(service, rover, 1.0)
    with pytest.raises(ValueError, match='no way'):
        service.set_goal({'x': -0.5, 'y': 0.6})
    assert rover.paths == [] and service.guide.goal is None
