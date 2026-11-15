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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from guidance import RoverClient  # noqa: E402
from guide_rover import GuideService, draw, to_pixel  # noqa: E402
from pose_camera import Pose  # noqa: E402
from rover_core import Config  # noqa: E402
from rover_marker import RoverFinder  # noqa: E402

BOUNDARY = 'camstreamframe'
# The floor frame is NOT the simulation's: turned by 40 degrees and shifted. The guide must
# find that out by itself, from the pictures.
TURN, SHIFT = math.radians(40), (0.2, -0.3)
SIDE, HEIGHT = 0.10, 0.12


def floor_pose(board) -> tuple[float, float, float]:
    c, s = math.cos(TURN), math.sin(TURN)
    return (
        c * board.x - s * board.y + SHIFT[0],
        s * board.x + c * board.y + SHIFT[1],
        board.yaw + TURN,
    )


def make_phone(app, floor):
    printed = syn.marker_image(0)
    t0 = time.monotonic()

    class Phone(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.end_headers()
            try:
                while True:
                    t = time.monotonic() - t0
                    x, y, heading = floor_pose(app.sim_board)
                    # a hand-held phone: it circles above the area and never looks straight down
                    eye = (0.3 + 0.25 * math.cos(0.7 * t), 0.25 * math.sin(0.7 * t), 1.9)
                    d = syn.arkit_pose(eye, (0.3, 0.0, 0.0), roll_deg=20 * math.sin(0.4 * t), t=t)
                    d['sid'] = 7
                    marker = (printed, x, y, heading - math.pi / 2, SIDE * 1.5, HEIGHT)
                    image = syn.render(Pose.from_arkit(d), floor, marker)
                    ok, jpg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    body = jpg.tobytes()
                    head = (
                        f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                        f'Content-Length: {len(body)}\r\nX-Pose: {syn.header(d)}\r\n\r\n'
                    )
                    self.wfile.write(head.encode() + body + b'\r\n')
                    time.sleep(0.05)
            except OSError:
                pass

    return Phone


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
        assert b'Click on the view' in get(base + '/')[1]
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
