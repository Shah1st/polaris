"""The rover page's D405 view, against a fake capture server with the same HTTP contract as
rover_mapper/d405_server.py (/info, /health, /color.jpg, /depth.png)."""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rover_server  # noqa: E402
from d405_view import D405UnavailableError, D405View, colorize, depth_stats  # noqa: E402
from rover_core import Config  # noqa: E402

SCALE = 0.0001  # m per raw unit, the real D405's default


def scene() -> np.ndarray:
    """Raw depth: a wall at 0.60 m, a box at 0.25 m in the middle, no measure on the left."""
    raw = np.full((480, 640), round(0.60 / SCALE), np.uint16)
    raw[180:300, 260:380] = round(0.25 / SCALE)
    raw[:, :40] = 0
    return raw


class FakeCapture(BaseHTTPRequestHandler):
    healthy = True

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype, age=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if age is not None:
            self.send_header('X-Age-Ms', str(age))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if not self.healthy and self.path != '/health':
            return self._send(503, b'{"error": "pas de D405"}', 'application/json')
        if self.path == '/info':
            info = {
                'name': 'Intel RealSense D405',
                'serial': '123',
                'firmware': '5.15',
                'usb': '3.2',
                'mode': '640x480@15',
                'depth_scale_m': SCALE,
            }
            self._send(200, json.dumps(info).encode(), 'application/json')
        elif self.path == '/health':
            body = {'ok': self.healthy, 'device': self.healthy, 'age_s': 0.04, 'frames': 321}
            self._send(200, json.dumps(body).encode(), 'application/json')
        elif self.path == '/color.jpg':
            ok, jpg = cv2.imencode('.jpg', np.full((480, 640, 3), 120, np.uint8))
            self._send(200, jpg.tobytes(), 'image/jpeg', 40)
        elif self.path == '/depth.png':
            ok, png = cv2.imencode('.png', scene())
            self._send(200, png.tobytes(), 'image/png', 40)
        else:
            self._send(404, b'{}', 'application/json')


@pytest.fixture
def capture():
    FakeCapture.healthy = True
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeCapture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()
    srv.server_close()


def test_depth_in_metres_and_numbers(capture) -> None:
    view = D405View(capture)
    depth, age = view.depth_m()
    assert depth.shape == (480, 640) and age == '40'
    assert depth[240, 320] == pytest.approx(0.25, abs=1e-4)
    stats = view.status()
    assert stats['ok'] and stats['camera']['name'] == 'Intel RealSense D405'
    assert stats['depth']['centre_m'] == pytest.approx(0.25, abs=0.001)
    assert stats['depth']['nearest_m'] == pytest.approx(0.25, abs=0.001)
    assert stats['depth']['valid'] == pytest.approx(600 / 640, abs=0.01)


def test_colour_map_red_near_blue_far_black_unknown() -> None:
    depth = scene().astype(np.float32) * SCALE
    img = cv2.imdecode(np.frombuffer(colorize(depth, 1.0), np.uint8), cv2.IMREAD_COLOR)
    near, far, none = img[240, 320], img[400, 600], img[400, 20]
    assert near[2] > near[0]  # red > blue: near
    assert far[0] > far[2] or far[1] > far[2]  # bluish / greenish: farther
    assert none.max() < 40  # no measure: black
    assert depth_stats(np.zeros((10, 10), np.float32))['centre_m'] is None


def test_unreachable_or_unhealthy_server(capture) -> None:
    with pytest.raises(D405UnavailableError, match='not reachable'):
        D405View('http://127.0.0.1:9').color_jpg()
    FakeCapture.healthy = False
    status = D405View(capture).status()
    assert not status['ok'] and 'HTTP 503' in status['error']
    assert not D405View(None).status()['ok']


def test_rover_server_routes(capture, tmp_path) -> None:
    args = SimpleNamespace(sim_drift=0.0, width=64, height=48, fps=5.0, quality=50, camera='0',
                           no_camera=True, d405=capture)  # fmt: skip
    app = rover_server.App(Config(), str(tmp_path / 'cfg.json'), args, serial_port=None)
    app.set_mode('sim')
    srv = ThreadingHTTPServer(('127.0.0.1', 0), rover_server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    try:
        with urllib.request.urlopen(base + '/d405/depth.jpg?max=0.5', timeout=5) as r:
            assert r.headers['Content-Type'] == 'image/jpeg' and r.headers['X-Age-Ms'] == '40'
            assert cv2.imdecode(np.frombuffer(r.read(), np.uint8), 1).shape == (480, 640, 3)
        with urllib.request.urlopen(base + '/d405/color.jpg', timeout=5) as r:
            assert r.read()[:2] == b'\xff\xd8'
        status = json.loads(urllib.request.urlopen(base + '/d405/status', timeout=5).read())
        assert status['ok'] and status['depth']['centre_m'] == pytest.approx(0.25, abs=0.001)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + '/d405/depth.jpg?max=50', timeout=5)
        assert e.value.code == 400
        FakeCapture.healthy = False
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + '/d405/color.jpg', timeout=5)
        assert e.value.code == 503
    finally:
        srv.shutdown()
        srv.server_close()
        app.close()
