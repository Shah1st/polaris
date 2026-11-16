"""d405_server.py GET /grid and POST /calibrate_floor, without the camera: the fake
pyrealsense2 of test_d405_server.py shows a synthetic scene, a floor 15 cm under the lens (the
rover's mount) and a box 20 cm wide and 15 cm high, 50 cm ahead."""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d405_server as ds  # noqa: E402
from test_d405_server import SCALE, get_json, intrinsics, wait_for, world  # noqa: E402, F401

HEIGHT = 0.15


def scene(w, h, box=True, height=HEIGHT, pitch_deg=0.0):
    """Raw depth (uint16) of a floor `height` under the lens, the camera pitched down by
    `pitch_deg`, with a box 0.5 m ahead."""
    k = intrinsics('depth', w, h)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x, y = (u - k.ppx) / k.fx, (v - k.ppy) / k.fy  # ray (x, y, 1), y down
    p = np.radians(pitch_deg)
    down = y * np.cos(p) + np.sin(p)  # the ray's downward part once the camera is pitched
    z = np.zeros((h, w), np.float32)
    floor = down > 0.02
    z[floor] = height / down[floor]
    z[z > 3.0] = 0
    if box:
        inside = (np.abs(x * 0.5) <= 0.10) & (y * 0.5 <= height) & (y * 0.5 >= height - 0.15)
        z[inside & ((z == 0) | (z > 0.5))] = 0.5
    return np.round(z / SCALE).astype(np.uint16)


def snap(depth, seq=1):
    h, w = depth.shape
    k = intrinsics('depth', w, h)
    info = {
        'depth_scale_m': SCALE,
        'depth': {'K': [[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]]},
    }
    return {'seq': seq, 't': 0.0, 'depth': depth, 'info': info}


def obstacles(grid):
    cells = np.asarray(grid['cells'])
    ii, jj = np.nonzero(cells == 1)
    return grid['x0'] + (jj + 0.5) * grid['res'], grid['y0'] + (ii + 0.5) * grid['res']


@pytest.mark.parametrize('stride', [1, 2, 4])
def test_grid_sees_the_box_and_the_floor(stride):
    maker = ds.GridMaker(SimpleNamespace(), hz=0, stride=stride)
    grid = maker.compute(snap(scene(640, 480)))
    assert grid['floor'] and grid['plane'] == 'fitted' and grid['res'] == 0.05
    xs, ys = obstacles(grid)
    assert len(xs) >= 3
    assert abs(float(np.median(xs)) - 0.5) < 0.06  # the box's front face, 50 cm ahead
    assert float(np.abs(ys).max()) < 0.20  # 20 cm wide, centred
    cells = np.asarray(grid['cells'])
    assert cells.shape == (50, 25)  # 2.5 m wide, 1.25 m ahead
    free = np.nonzero(cells == 0)
    free_x = grid['x0'] + (free[1] + 0.5) * grid['res']
    assert len(free_x) > 30 and float(free_x.min()) < 0.40  # the floor before the box
    # nothing is "free" behind the box: what it hides is not seen
    row = cells[cells.shape[0] // 2]
    assert not (row[12:] == 0).any()


def test_empty_floor_has_no_obstacle():
    maker = ds.GridMaker(SimpleNamespace(), hz=0)
    grid = maker.compute(snap(scene(640, 480, box=False)))
    assert grid['floor'] and len(obstacles(grid)[0]) == 0
    assert (np.asarray(grid['cells']) == 0).sum() > 100


def test_calibrated_floor_is_kept_and_survives_a_wall(tmp_path):
    path = str(tmp_path / 'floor.json')
    maker = ds.GridMaker(SimpleNamespace(), hz=0, floor_file=path)
    c = maker.calibrate(snap(scene(640, 480, box=False, pitch_deg=10.0)))
    assert c['height_m'] == pytest.approx(HEIGHT, abs=0.005)
    assert c['tilt_deg'] == pytest.approx(10.0, abs=0.5)
    assert json.load(open(path, encoding='utf-8'))['d'] == pytest.approx(HEIGHT, abs=0.005)

    # a wall 30 cm ahead fills the picture: no floor to fit, the calibrated plane still works
    wall = np.full((480, 640), round(0.30 / SCALE), np.uint16)
    again = ds.GridMaker(SimpleNamespace(), hz=0, floor_file=path)  # read back from the file
    grid = again.compute(snap(wall))
    assert grid['plane'] == 'calibrated' and grid['floor']
    xs, _ = obstacles(grid)
    assert len(xs) >= 5 and abs(float(np.median(xs)) - 0.30) < 0.06
    unfitted = ds.GridMaker(SimpleNamespace(), hz=0).compute(snap(wall))
    assert not unfitted['floor'] and len(obstacles(unfitted)[0]) == 0  # why it is calibrated


def test_calibration_refuses_a_wrong_height_or_no_floor(tmp_path):
    path = str(tmp_path / 'floor.json')
    maker = ds.GridMaker(SimpleNamespace(), hz=0, floor_file=path)
    with pytest.raises(ValueError, match='expected 0.150'):
        maker.calibrate(snap(scene(640, 480, box=False, height=0.30)))  # on a table?
    assert maker.plane is None and not os.path.exists(path)
    assert maker.calibrate(snap(scene(640, 480, box=False, height=0.30)), force=True)['height_m'] > 0.28
    with pytest.raises(ValueError, match='no floor'):
        ds.GridMaker(SimpleNamespace(), hz=0).calibrate(snap(np.zeros((480, 640), np.uint16)))


def scene_box_at(distance, w=640, h=480):
    """The floor and a box 20 cm wide, 15 cm high, `distance` m ahead."""
    k = intrinsics('depth', w, h)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x, y = (u - k.ppx) / k.fx, (v - k.ppy) / k.fy
    z = scene(w, h, box=False).astype(np.float32) * SCALE
    inside = (np.abs(x * distance) <= 0.10) & (y * distance <= HEIGHT)
    inside &= y * distance >= HEIGHT - 0.15
    z[inside & ((z == 0) | (z > distance))] = distance
    return np.round(z / SCALE).astype(np.uint16)


def test_no_obstacle_is_reported_far_away_but_the_floor_is():
    maker = ds.GridMaker(SimpleNamespace(), hz=0)
    far = maker.compute(snap(scene_box_at(1.0)))
    assert len(obstacles(far)[0]) == 0  # beyond 0.8 m: too far to trust, and early enough later
    cells = np.asarray(far['cells'])
    free_x = far['x0'] + (np.nonzero(cells == 0)[1] + 0.5) * far['res']
    assert float(free_x.max()) > 0.9  # the floor in front of it is still seen free
    near = maker.compute(snap(scene_box_at(0.7)))
    assert abs(float(np.median(obstacles(near)[0])) - 0.7) < 0.06
    everything = ds.GridMaker(SimpleNamespace(), hz=0, obstacle_range=None)
    assert abs(float(np.median(obstacles(everything.compute(snap(scene_box_at(1.0))))[0])) - 1.0) < 0.06


@pytest.mark.parametrize('pitch_deg', [3.0, 2.0, -2.0])
def test_a_rover_that_pitches_does_not_turn_the_floor_into_obstacles(tmp_path, pitch_deg):
    """The floor is calibrated standing still; braking tips the nose down by two or three degrees.
    With a fixed 4 cm threshold the far floor then read as obstacles (the old settings, kept
    here to show it); with a threshold that grows with the distance it does not."""
    path = str(tmp_path / 'floor.json')
    still = snap(scene(640, 480, box=False))
    tipped = snap(scene(640, 480, box=False, pitch_deg=pitch_deg))
    maker = ds.GridMaker(SimpleNamespace(), hz=0, floor_file=path)
    maker.calibrate(still)
    assert len(obstacles(maker.compute(tipped))[0]) == 0
    if pitch_deg == 3.0:
        old = ds.GridMaker(SimpleNamespace(), hz=0, floor_file=path, obstacle_range=None,
                           band_slope=0.0, min_points=3)  # fmt: skip
        assert len(obstacles(old.compute(tipped))[0]) > 5  # phantoms on an empty floor
    # a real box is still seen while pitching
    boxed = snap(scene(640, 480, box=True, pitch_deg=pitch_deg))
    xs, _ = obstacles(maker.compute(boxed))
    assert len(xs) >= 3 and abs(float(np.median(xs)) - 0.5) < 0.08


def post(url):
    req = urllib.request.Request(url, data=b'', method='POST')
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_grid_and_calibration_over_http(world, tmp_path):  # noqa: F811
    world.depth_fn = lambda n, w, h: scene(w, h)
    cap = ds.Capture(retry_s=0.1, timeout_ms=300, log=lambda *a, **k: None)
    maker = ds.GridMaker(cap, hz=20.0, floor_file=str(tmp_path / 'floor.json'))
    srv = ds.make_server('127.0.0.1', 0, cap, max_age=1.0, grid=maker)
    url = 'http://127.0.0.1:%d' % srv.server_address[1]
    cap.start()
    maker.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert wait_for(lambda: get_json(url + '/health')[1]['ok'])
        assert maker.grids == 0  # nothing is computed before someone asks
        code, grid = get_json(url + '/grid')
        assert code == 200 and grid['floor'] and grid['age_s'] < 1.0
        assert abs(float(np.median(obstacles(grid)[0])) - 0.5) < 0.06
        code, later = get_json(url + '/grid')
        assert wait_for(lambda: get_json(url + '/grid')[1]['seq'] > grid['seq'])
        health = get_json(url + '/health')[1]['grid']
        assert health['grids'] >= 2 and health['cpu_ms_per_grid'] > 0

        world.depth_fn = lambda n, w, h: scene(w, h, box=False)
        assert wait_for(lambda: len(obstacles(get_json(url + '/grid')[1])[0]) == 0)
        code, body = post(url + '/calibrate_floor')
        assert code == 200 and body['calibration']['height_m'] == pytest.approx(HEIGHT, abs=0.005)
        assert wait_for(lambda: get_json(url + '/grid')[1]['plane'] == 'calibrated')
        assert get_json(url + '/calibration')[1]['calibration']['seq'] > 0

        world.depth_fn = lambda n, w, h: scene(w, h, box=False, height=0.4)
        assert wait_for(lambda: post(url + '/calibrate_floor')[0] == 422)
        world.plugged = False
        assert wait_for(lambda: get_json(url + '/grid')[0] == 503)
    finally:
        srv.shutdown()
        srv.server_close()
        maker.stop()
        cap.stop()


def test_grid_off():
    cap = SimpleNamespace(device=True, last_error=None, latest=lambda: None, info=None, seq=0)
    srv = ds.make_server('127.0.0.1', 0, cap, grid=ds.GridMaker(cap, hz=0))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, body = get_json('http://127.0.0.1:%d/grid' % srv.server_address[1])
        assert code == 503 and 'grid-hz' in body['error']
    finally:
        srv.shutdown()
        srv.server_close()
