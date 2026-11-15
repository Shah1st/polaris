"""The whole chain on this computer: a fake phone that films the SIMULATED rover of
rover/rover_server.py from a camera that keeps moving, and the guide that drives it to a
point clicked on the top view. Real HTTP on both sides, real marker detection."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'rover'))

import rover_server  # noqa: E402
import synthetic as syn  # noqa: E402
from fake_phone import HEIGHT, SIDE, TURN, floor_pose, make_phone  # noqa: E402
from guidance import RoverClient  # noqa: E402
from guide_rover import GuideService, draw, to_pixel  # noqa: E402
from rover_core import Config  # noqa: E402
from rover_marker import RoverFinder  # noqa: E402


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.headers.get('Content-Type'), r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), method='POST')
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_click_on_the_top_view_and_the_simulated_rover_drives_there(tmp_path) -> None:
    args = SimpleNamespace(sim_drift=0.0, width=64, height=48, fps=5.0, quality=50, camera='0',
                           no_camera=True, d405='off')  # fmt: skip
    app = rover_server.App(Config(), str(tmp_path / 'cfg.json'), args, serial_port=None)
    app.set_mode('sim')
    rover_srv = ThreadingHTTPServer(('127.0.0.1', 0), rover_server.make_handler(app))
    threading.Thread(target=rover_srv.serve_forever, daemon=True).start()
    rover_url = f'http://127.0.0.1:{rover_srv.server_address[1]}'
    phone_srv = ThreadingHTTPServer(('127.0.0.1', 0), make_phone(app, syn.floor_texture()))
    phone_srv.daemon_threads = True
    threading.Thread(target=phone_srv.serve_forever, daemon=True).start()
    service = GuideService(
        f'http://127.0.0.1:{phone_srv.server_address[1]}',
        RoverClient(rover_url, name='air-guide-test'),
        RoverFinder(0, SIDE, HEIGHT),
        bounds=(-1.0, -1.0, 1.0, 1.0),
        colour=False,  # the test floor is painted with coloured squares: not obstacles
        res=0.005,
        port=0,
    )
    service.start()
    base = f'http://127.0.0.1:{service.port}'
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:  # the rover's gyro, and the first sighting
            state = json.loads(get(base + '/state')[1])
            ready = json.loads(get(rover_url + '/state')[1])['odom']['gyro_ready']
            if ready and state['link']['ready'] and state['rover']:
                break
            time.sleep(0.2)
        assert state['link']['ready'], state
        # the link found the hidden transform between the two frames
        assert state['link']['theta_deg'] == pytest.approx(math.degrees(TURN), abs=3)
        assert state['rover'][:2] == pytest.approx(floor_pose(app.sim_board)[:2], abs=0.03)

        assert post(base + '/goal', {'fx': 2})[0] == 400
        assert post(base + '/goal', {'x': 'far'})[0] == 400
        # a click at 70 % of the width, 30 % of the height of a 2 m view: (0.4, 0.4) m
        code, answer = post(base + '/goal', {'fx': 0.7, 'fy': 0.3})
        assert code == 200 and answer['goal'] == pytest.approx([0.4, 0.4])

        deadline = time.monotonic() + 45
        seen_driving = False
        while time.monotonic() < deadline:
            state = json.loads(get(base + '/state')[1])
            seen_driving |= state['mode'] == 'path'
            if state['status'].startswith(('arrived', 'stopped')):
                break
            time.sleep(0.2)
        assert state['status'].startswith('arrived'), state
        assert seen_driving
        x, y, _ = floor_pose(app.sim_board)
        assert math.hypot(x - 0.4, y - 0.4) < 0.05, (x, y)
        assert 'ok' in state['phone'] and state['latency_ms'] is not None

        ctype, jpg = get(base + '/top.jpg')
        top = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert ctype == 'image/jpeg' and top.shape == (400, 400, 3)
        assert b'Click on the map' in get(base + '/')[1]

        # what the dashboard needs besides the map: the cameras' addresses (the browser reads
        # them itself), the rover's own telemetry, what the last picture gave
        state = json.loads(get(base + '/state')[1])
        assert state['rover_url'] == rover_url and state['phone_url'].startswith('http://')
        assert state['rover_state']['odom']['gyro_ready'] and state['rover_state_age_s'] < 1
        assert state['rover_rtt_ms']['usual'] < 500 and state['trail_m'] > 0.3
        air = state['air']
        assert air['tracking'] == 'normal' and air['floor_known'] and air['seen_age_s'] < 1
        assert air['height_m'] == pytest.approx(1.9, abs=0.01) and air['marker']['side_px'] > 20
        # the rover's JSON goes through the guide (the rover sends no CORS headers)
        assert json.loads(get(base + '/rover/config')[1])['v_manual'] > 0
        assert urllib.request.urlopen(base + '/rover/d405/status', timeout=5).status == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            get(base + '/rover/record.csv')
        assert e.value.code == 404
        # manual driving from the dashboard: under its own name, refusals passed on, and a goal
        # gives way to it
        assert post(base + '/goal', {'x': -0.3, 'y': 0.0})[0] == 200
        assert post(base + '/rover/cmd', {'v': 'fast'})[0] == 400
        assert post(base + '/rover/move', {'distance': 1})[0] == 404
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and json.loads(get(base + '/state')[1])['goal']:
            assert post(base + '/rover/cmd', {'v': 0.1, 'omega': 0.0})[0] == 200  # a held key
            time.sleep(0.1)
        state = json.loads(get(base + '/state')[1])
        assert state['goal'] is None and 'someone else' in state['status']
        log = json.loads(get(base + '/rover/commands')[1])
        assert any(c['client'].endswith('dashboard') for c in log['clients'])
        # STOP: no goal left, and the rover got a stop whoever was driving
        stops = json.loads(get(rover_url + '/state')[1])['stop_count']
        assert post(base + '/goal', {'x': 0.0, 'y': 0.0})[0] == 200
        assert post(base + '/stop', {})[0] == 200
        state = json.loads(get(base + '/state')[1])
        assert state['goal'] is None and state['status'] == 'stopped from the dashboard'
        assert json.loads(get(rover_url + '/state')[1])['stop_count'] == stops + 1
        # the rover's command log names this guide: the team sees who drove
        clients = json.loads(get(rover_url + '/commands')[1])['clients']
        assert any(c['client'].endswith('air-guide-test') for c in clients)
        assert post(base + '/cancel', {})[0] == 200
    finally:
        service.close()
        phone_srv.shutdown()
        rover_srv.shutdown()
        rover_srv.server_close()
        app.close()


def test_drawing_puts_things_where_they_are() -> None:
    bounds, res = (-1.0, -1.0, 1.0, 1.0), 0.01
    assert to_pixel(-1.0, 1.0, bounds, res) == (0, 0)
    assert to_pixel(0.5, -0.5, bounds, res) == (150, 150)
    top = np.zeros((200, 200, 3), np.uint8)
    snap = {'rover': [0.0, 0.0, 0.0], 'goal': (0.5, 0.5), 'waypoints': [(0.5, 0.0)]}
    out = draw(top, bounds, res, snap, [(-0.5, 0.0), (0.0, 0.0)], None)
    assert out[100, 105, 1] > 200  # the rover's green arrow, pointing along +x
    assert out[50, 150, 2] > 200  # the goal's red cross at (0.5, 0.5)
    assert out[100, 75].any() and not top.any()  # the trail; the input is left untouched
