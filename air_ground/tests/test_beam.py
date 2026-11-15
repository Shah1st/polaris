"""The rover's D405 as a second stream: its looks placed on the floor, then believed over the
air map (add, confirm, clear) until the air looks again."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from beam import BeamReader, parse_look  # noqa: E402
from floor_map import FloorMap  # noqa: E402
from guidance import Pose2  # noqa: E402

RES = 0.05


def look_json(obstacles=(), age_s=0.1, rng=1.0) -> dict:
    """What rover_mapper/perception_service.py --d405 answers on GET /obstacles: a grid in
    the D405's frame (x ahead in [0, rng], y to the left), cell [i][j] at x0 + j res,
    y0 + i res; floor everywhere it sees (a 60 degree cone), `obstacles` as (x, y) boxes."""
    n = round(rng / RES)
    cells = np.full((2 * n, n), -1, np.int8)
    x0, y0 = 0.0, -n * RES
    for i in range(2 * n):
        for j in range(n):
            x, y = x0 + (j + 0.5) * RES, y0 + (i + 0.5) * RES
            if abs(math.atan2(y, x)) < math.radians(30) and x > 0.07:
                cells[i, j] = 0
                for bx0, by0, bx1, by1 in obstacles:
                    if bx0 <= x <= bx1 and by0 <= y <= by1:
                        cells[i, j] = 1
    return {'cells': cells.tolist(), 'x0': x0, 'y0': y0, 'res': RES, 'scale': 1.0,
            'floor': True, 'age_s': age_s, 'sensor': 'd405'}  # fmt: skip


def test_a_look_lands_where_the_rover_points() -> None:
    look = parse_look(look_json([(0.30, -0.05, 0.40, 0.05)]), time.monotonic(), 1)
    # rover at (1, 2) facing +y (90 deg); the D405 10 cm ahead of its centre
    obstacle, floor = look.on_floor(Pose2(1.0, 2.0, math.pi / 2), offset=(0.10, 0.0))
    centre = obstacle.mean(axis=0)
    assert centre == pytest.approx((1.0, 2.0 + 0.10 + 0.35), abs=0.03)  # 45 cm ahead of it
    assert len(floor) > 10 * len(obstacle)
    assert (floor[:, 1] - 2.1 <= 1.0 + 0.03).all()  # within its range ahead of it
    blind = parse_look(look_json() | {'floor': False}, time.monotonic(), 2)
    assert (blind.cells == -1).all()  # no floor found: it says nothing


def test_the_d405_adds_confirms_and_clears_cells() -> None:
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    rover = Pose2(0.0, 0.0, 0.0)  # at the middle, facing +x

    def obstacle_at(x, y):
        _, _, ob = fmap.obstacles()
        c, r = fmap._cells(np.asarray([[x, y]]))[0].astype(int)
        return bool(ob[r, c])

    # the air thinks there is something at (0.5, 0.2): a false obstacle
    k = fmap.colour_k
    for dx in range(4):
        for dy in range(4):
            c, r = (fmap._cells(np.asarray([[0.45 + 0.03 * dx, 0.15 + 0.03 * dy]]))[0] / k).astype(
                int
            )
            fmap.height_hits[r, c] = 3
    assert obstacle_at(0.5, 0.2)
    # the D405 looks: a box at 0.35 m ahead the air never saw, and floor where the air was wrong
    look = parse_look(look_json([(0.30, -0.05, 0.40, 0.05)]), time.monotonic(), 1)
    out = fmap.add_beam(*look.on_floor(rover))
    assert out['obstacle_cells'] > 0
    assert obstacle_at(0.35, 0.0)  # added
    assert not obstacle_at(0.5, 0.2)  # cleared, and the air's hits forgotten
    c, r = (fmap._cells(np.asarray([[0.5, 0.2]]))[0] / k).astype(int)
    assert fmap.height_hits[r, c] == 0
    # the box is taken away: the next look sees floor there
    fmap.add_beam(*parse_look(look_json(), time.monotonic(), 2).on_floor(rover))
    assert not obstacle_at(0.35, 0.0)


def test_a_box_right_in_front_is_not_taken_for_the_rover() -> None:
    """The air map frees a patch touching the rover (its own parts); a box the D405 sees
    15 cm in front of it must stay an obstacle."""
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    fmap.set_rover(0.0, 0.0)
    look = parse_look(look_json([(0.08, -0.06, 0.20, 0.06)]), time.monotonic(), 1)
    fmap.add_beam(*look.on_floor(Pose2(0.0, 0.0, 0.0), offset=(0.0, 0.0)))
    _, _, ob = fmap.obstacles()
    c, r = fmap._cells(np.asarray([[0.15, 0.0]]))[0].astype(int)
    assert ob[r, c]


class FakePerception(BaseHTTPRequestHandler):
    obstacles = [(0.30, -0.05, 0.40, 0.05)]

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps(look_json(self.obstacles, age_s=0.15)).encode()
        self.send_response(200 if self.path == '/obstacles' else 404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_reader_keeps_the_newest_look() -> None:
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakePerception)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reader = BeamReader(f'http://127.0.0.1:{srv.server_address[1]}', period=0.05)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and reader.count < 3:
            time.sleep(0.02)
        look = reader.latest(max_age=2.0)
        assert look is not None and reader.error is None and look.seq >= 3
        assert look.taken < look.received and (look.cells == 1).any()
    finally:
        reader.close()
        srv.shutdown()
        srv.server_close()
    dead = BeamReader('http://127.0.0.1:9', timeout=0.3)
    time.sleep(0.5)
    assert dead.latest() is None and dead.error
    dead.close()


class StubBeam:
    """Stands for BeamReader: the D405 of a simulated rover, looking at a box only it sees."""

    def __init__(self, rover, box):
        self.rover, self.box = rover, box
        self.count, self.error, self.last_s = 0, None, 0.05

    def latest(self, max_age=None):
        self.count += 1
        p = self.rover.truth  # where the rover really is: the D405 sees the box from there
        c, s = math.cos(p.heading), math.sin(p.heading)
        bx0, by0, bx1, by1 = self.box
        corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        local = [
            ((x - p.x) * c + (y - p.y) * s - 0.10, -(x - p.x) * s + (y - p.y) * c)
            for x, y in corners
        ]
        xs, ys = [q[0] for q in local], [q[1] for q in local]
        d = look_json([(min(xs), min(ys), max(xs), max(ys))], age_s=0.0)
        return parse_look(d, self.rover.t, self.count)

    def close(self):
        pass


def test_a_box_only_the_d405_sees_is_driven_around() -> None:
    from guide_rover import GuideService
    from test_guidance import FakeRover
    from test_planner import drive

    rover = FakeRover(Pose2(-0.5, -0.6, math.radians(90)), scale=1.0, drift=0.0)
    s = GuideService('http://127.0.0.1:9', rover, finder=None, port=0,
                     bounds=(-1.5, -1.5, 1.5, 1.5), res=0.01, colour=False)  # fmt: skip
    s.stream.close()
    box = (-0.65, -0.05, -0.35, 0.10)  # on the straight line to the goal, unseen from the air
    s.beam = StubBeam(rover, box)
    try:
        drive(s, rover, 1.0, on_tick=lambda t: s._use_beam())
        s.set_goal({'x': -0.5, 'y': 0.6})
        statuses = drive(s, rover, 40.0, on_tick=lambda t: s._use_beam())
        assert statuses[-1].startswith('arrived'), statuses
        assert s.beam_events >= 1 and s.plans >= 2  # seen on the way: planned again
        trail = [(p.x, p.y) for _, p in rover.trail]
        clear = min(max(abs(x - (-0.5)) - 0.15, abs(y - 0.025) - 0.075, 0) for x, y in trail)
        inside = [(x, y) for x, y in trail if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
        assert not inside and clear >= 0  # it never drove into the box
    finally:
        s.server.server_close()
