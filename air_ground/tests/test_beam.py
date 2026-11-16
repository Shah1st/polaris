"""The rover's D405 as a second stream: the team's local grid (GET /local_grid, odometry
frame) placed on the floor and believed over the air map (add, confirm, clear); the
rover's guard stopping a path ('blocked') puts the obstacle on the map and replans."""

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

from beam import BeamReader, parse_grid  # noqa: E402
from floor_map import FloorMap  # noqa: E402
from guidance import OdomLink, Pose2  # noqa: E402

RES = 0.05


def grid_json(pose, boxes=(), frames=1, age_s=0.1, seen=1.0) -> dict:
    """What the team's rover answers on GET /local_grid: 2.5 m around the rover, odometry
    frame; floor in a 60 degree cone up to `seen` m ahead of it, and the boxes (odometry
    x0, y0, x1, y1) as obstacles where the cone reaches them."""
    n = 50
    x0, y0 = pose[0] - n * RES / 2, pose[1] - n * RES / 2
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    x, y = x0 + (jj + 0.5) * RES, y0 + (ii + 0.5) * RES
    dx, dy = x - pose[0], y - pose[1]
    ahead = dx * math.cos(pose[2]) + dy * math.sin(pose[2])
    side = -dx * math.sin(pose[2]) + dy * math.cos(pose[2])
    cone = (ahead > 0.07) & (ahead <= seen) & (np.abs(np.arctan2(side, ahead)) < math.radians(30))
    cells = np.where(cone, 0, -1).astype(np.int8)
    for bx0, by0, bx1, by1 in boxes:
        cells[cone & (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)] = 1
    return {'res': RES, 'x0': x0, 'y0': y0, 'cells': cells.tolist(), 'pose': list(pose),
            'age_s': age_s, 'frames': frames, 'error': None}  # fmt: skip


def turned_link(theta_deg=90.0, tx=1.0, ty=2.0) -> OdomLink:
    link = OdomLink()
    link.theta, link.tx, link.ty, link.ready = math.radians(theta_deg), tx, ty, True
    return link


def test_the_local_grid_lands_on_the_floor_through_the_link() -> None:
    # odometry: rover at (0, 0) facing +x, a box 0.35 m ahead; floor = odometry turned 90 deg
    g = parse_grid(grid_json((0.0, 0.0, 0.0), [(0.30, -0.05, 0.40, 0.05)]), time.monotonic())
    obstacle, floor = g.on_floor(turned_link())
    assert len(obstacle) >= 2
    assert obstacle.mean(axis=0) == pytest.approx((1.0, 2.35), abs=0.03)  # ahead = +y on the floor
    assert len(floor) > 10 * len(obstacle)
    near, _ = g.on_floor(turned_link(), max_dist=0.2)
    assert len(near) == 0  # beyond the range asked: left out
    assert parse_grid({'cells': [], 'age_s': None}, 0.0).cells.size == 0


def test_the_d405_adds_confirms_and_clears_cells() -> None:
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    link = turned_link(0.0, 0.0, 0.0)  # floor = odometry

    def obstacle_at(x, y):
        _, _, ob = fmap.obstacles()
        c, r = fmap._cells(np.asarray([[x, y]]))[0].astype(int)
        return bool(ob[r, c])

    k = fmap.colour_k
    for dx in range(4):  # the air thinks there is something at (0.5, 0.2): a false obstacle
        for dy in range(4):
            xy = np.asarray([[0.45 + 0.03 * dx, 0.15 + 0.03 * dy]])
            c, r = (fmap._cells(xy)[0] / k).astype(int)
            fmap.height_hits[r, c] = 3
    assert obstacle_at(0.5, 0.2)
    g = parse_grid(grid_json((0.0, 0.0, 0.0), [(0.30, -0.05, 0.40, 0.05)]), time.monotonic())
    out = fmap.add_beam(*g.on_floor(link))
    assert out['obstacle_cells'] > 0
    assert obstacle_at(0.35, 0.0)  # added: the air never saw it
    assert not obstacle_at(0.5, 0.2)  # cleared, and the air's verdict forgotten
    c, r = (fmap._cells(np.asarray([[0.5, 0.2]]))[0] / k).astype(int)
    assert fmap.height_hits[r, c] == 0
    g = parse_grid(grid_json((0.0, 0.0, 0.0), frames=2), time.monotonic())
    fmap.add_beam(*g.on_floor(link))
    assert not obstacle_at(0.35, 0.0)  # taken away: the next look sees floor there


def test_a_box_right_in_front_is_not_taken_for_the_rover() -> None:
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    fmap.set_rover(0.0, 0.0)
    g = parse_grid(grid_json((0.0, 0.0, 0.0), [(0.08, -0.06, 0.20, 0.06)]), time.monotonic())
    fmap.add_beam(*g.on_floor(turned_link(0.0, 0.0, 0.0)))
    _, _, ob = fmap.obstacles()
    c, r = fmap._cells(np.asarray([[0.15, 0.0]]))[0].astype(int)
    assert ob[r, c]


class FakeRoverServer(BaseHTTPRequestHandler):
    frames = 0

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        FakeRoverServer.frames += 1
        d = grid_json((0.0, 0.0, 0.0), [(0.30, -0.05, 0.40, 0.05)], frames=FakeRoverServer.frames)
        body = json.dumps(d).encode()
        self.send_response(200 if self.path == '/local_grid' else 404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_reader_keeps_the_newest_grid() -> None:
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeRoverServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reader = BeamReader(f'http://127.0.0.1:{srv.server_address[1]}', period=0.05)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and reader.count < 3:
            time.sleep(0.02)
        g = reader.latest(max_age=2.0)
        assert g is not None and reader.error is None and g.frames >= 3
        assert (g.cells == 1).any()
        assert reader.latest(max_age=0.01) is None  # its newest picture is 0.1 s old
    finally:
        reader.close()
        srv.shutdown()
        srv.server_close()
    dead = BeamReader('http://127.0.0.1:9', timeout=0.3)
    time.sleep(0.5)
    assert dead.latest() is None and dead.error
    dead.close()


class StubBeam:
    """Stands for BeamReader: the D405 of a simulated rover (its odometry frame), looking at
    a box only it sees."""

    def __init__(self, rover, box_floor):
        self.rover, self.box = rover, box_floor
        self.count, self.error, self.last_s = 0, None, 0.05

    def latest(self, max_age=None):
        self.count += 1
        o, t = self.rover.odom, self.rover.truth
        # the box in the odometry frame, through where odometry and truth differ
        dth = o.heading - t.heading
        c, s = math.cos(dth), math.sin(dth)
        bx0, by0, bx1, by1 = self.box
        pts = []
        for x, y in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)):
            dx, dy = x - t.x, y - t.y
            pts.append((o.x + c * dx - s * dy, o.y + s * dx + c * dy))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        d = grid_json(
            (o.x, o.y, o.heading),
            [(min(xs), min(ys), max(xs), max(ys))],
            frames=self.count,
            seen=0.4,
        )  # fmt: skip  (seen once near)
        return parse_grid(d, time.monotonic())

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
        inside = [(x, y) for x, y in trail if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
        assert not inside  # it never drove into the box
        kinds = {e['kind'] for e in s.events}
        assert {'goal', 'route', 'd405'} <= kinds
        assert rover.path_ids and all(i and i.startswith('air-') for i in rover.path_ids)
        assert len(set(rover.path_ids)) == len(rover.path_ids)  # a new id per path: taken over
    finally:
        s.server.server_close()


def test_a_guard_stop_becomes_an_obstacle_and_a_new_route() -> None:
    from guide_rover import GuideService
    from test_guidance import FakeRover

    rover = FakeRover(Pose2(0.0, 0.0, 0.0), scale=1.0, drift=0.0)
    s = GuideService('http://127.0.0.1:9', rover, finder=None, port=0, beam=False,
                     bounds=(-1.5, -1.5, 1.5, 1.5), res=0.01, colour=False)  # fmt: skip
    s.stream.close()
    try:
        g = s.guide
        g.link.theta, g.link.tx, g.link.ty, g.link.ready = 0.0, 0.0, 0.0, True
        state = rover.state()
        state['ctrl']['history'] = [{'id': 'air-1-1', 'status': 'blocked', 'duration': 2.0,
                                     'obstacle': [0.4, 0.0]}]  # fmt: skip
        g.on_state(state, 0.0)  # there before any goal of this guide: not its own
        assert g.blocked_at == []
        g.goal = (1.0, 0.0)
        state['ctrl']['history'] = [
            {'id': 'air-1-2', 'status': 'blocked', 'duration': 2.0, 'obstacle': [0.4, 0.0]},
            *state['ctrl']['history'],
        ]  # fmt: skip  (newest first)
        g.on_state(state, 0.1)
        assert g.blocked_at == [pytest.approx((0.4, 0.0))] and g.blocked_in_a_row == 1
        g.on_state(state, 0.2)  # the same entry read again: counted once
        assert g.blocked_in_a_row == 1
        s._next_plan = 5.0
        s._use_blocked()
        _, _, ob = s.mapper.fmap.obstacles()
        c, r = s.mapper.fmap._cells(np.asarray([[0.4, 0.0]]))[0].astype(int)
        assert ob[r, c] and s._next_plan == 0.0 and s.events[0]['kind'] == 'guard'
        assert g.blocked_at == []
    finally:
        s.server.server_close()
