"""perception_service.py without the model: frame sources (MJPEG stream, snapshot URL) and the
HTTP API with a fake perceiver. Fake servers on 127.0.0.1:8132-8136.

Run: python -m pytest rover_mapper/test_perception_service.py -q -p no:cacheprovider
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import perception_service as ps  # noqa: E402
from rovermap import perception as per  # noqa: E402


def jpeg(value=128):
    img = np.full((48, 64, 3), value, np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


class Source:
    """'stream': MJPEG, one frame every `period` s for `frames` frames, then the connection
    stays open and silent; 'snapshot': one JPEG per GET."""

    def __init__(self, port, kind, period=0.05, frames=10**6):
        self.kind, self.period, self.frames = kind, period, frames
        src = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                if src.kind == 'snapshot':
                    b = jpeg()
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    for k in range(src.frames):
                        b = jpeg(k % 250)
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                            + str(len(b)).encode()
                            + b'\r\n\r\n'
                            + b
                            + b'\r\n'
                        )
                        self.wfile.flush()
                        time.sleep(src.period)
                    time.sleep(30)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.srv = ThreadingHTTPServer(('127.0.0.1', port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{port}/video'

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_stream_gives_fresh_frames():
    src = Source(8132, 'stream')
    cam = ps.Camera(src.url)
    try:
        f = cam.grab(time.time(), 3.0)
        assert f is not None and cam.mode == 'stream'
        t, seq, rgb = f
        assert rgb.shape == (48, 64, 3)
        f2 = cam.grab(t + 1e-6, 2.0)
        assert f2 is not None and f2[1] > seq
    finally:
        cam.close()
        src.close()


def test_stream_that_stops_gives_none():
    src = Source(8133, 'stream', frames=2)
    cam = ps.Camera(src.url)
    try:
        assert cam.grab(0.0, 3.0) is not None
        time.sleep(0.3)
        assert cam.grab(time.time(), 0.5) is None  # nothing new arrives
        assert cam.grab(time.time() - 2.0, 0.1) is not None  # the old one, if old is allowed
    finally:
        cam.close()
        src.close()


def test_snapshot_url_fetched_on_demand():
    src = Source(8134, 'snapshot')
    cam = ps.Camera(src.url)
    try:
        f = cam.grab(time.time(), 3.0)
        assert f is not None and cam.mode == 'snapshot'
        assert cam.grab(time.time(), 1.0)[1] == f[1] + 1
    finally:
        cam.close()
        src.close()


class FakePerceiver:
    def __init__(self):
        self.last = None
        self.calls = 0

    def run(self, frame):
        t, seq, rgb = frame
        if self.last is not None and self.last['seq'] == seq:
            return self.last
        self.calls += 1
        h, w = 120, 160
        K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1.0]])
        v = np.mgrid[0:h, 0:w][0].astype(np.float32)
        yc = (v - K[1, 2]) / K[1, 1]
        depth = np.where(yc > 0.01, 0.22 / np.maximum(yc, 1e-6), 0).astype(np.float32)
        r = per.perceive(depth, K, plane=(np.array([0.0, -1.0, 0.0]), 0.22), max_range=1.0)
        self.last = {'seq': seq, 't': t, 'result': r, 'image': np.zeros((h, w, 3), np.uint8),
                     'K': K, 'dt': 0.01}
        return self.last


def args(source):
    return SimpleNamespace(source=source, max_age=2.0, cam_height=0.22, range=1.0,
                           band_low=0.04, band_high=0.40, k=None, fov=None, model='fake')


def serve(cam, perceiver, a, port):
    srv = ThreadingHTTPServer(('127.0.0.1', port), ps.make_handler(cam, perceiver, a))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_api_obstacles_debug_health():
    src = Source(8135, 'stream')
    cam = ps.Camera(src.url)
    srv = serve(cam, FakePerceiver(), args(src.url), 8136)
    try:
        code, body = get('http://127.0.0.1:8136/obstacles?fresh=1')
        assert code == 200
        r = json.loads(body)
        for k in ('t', 'age_s', 'res', 'x0', 'y0', 'cells'):
            assert k in r
        assert r['res'] == 0.05 and r['age_s'] < 2.0
        m = np.array(r['cells'])
        assert set(np.unique(m)) <= {-1, 0, 1} and (m == 0).any()
        code, body = get('http://127.0.0.1:8136/debug.jpg')
        assert code == 200 and body[:2] == b'\xff\xd8'
        code, body = get('http://127.0.0.1:8136/health')
        assert code == 200 and json.loads(body)['source_mode'] == 'stream'
        assert get('http://127.0.0.1:8136/nope')[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()
        cam.close()
        src.close()


def test_api_503_without_fresh_frame():
    cam = ps.Camera('http://127.0.0.1:8139/nothing')  # nobody listens
    a = args(cam.source)
    a.max_age = 0.3
    srv = serve(cam, FakePerceiver(), a, 8137)
    try:
        code, body = get('http://127.0.0.1:8137/obstacles')
        assert code == 503
        assert 'error' in json.loads(body)
    finally:
        srv.shutdown()
        srv.server_close()
        cam.close()


@pytest.mark.parametrize('fov', [100.0])
def test_known_k_from_fov(fov):
    a = SimpleNamespace(k=None, fov=fov)
    K = ps.known_k(a, 640, 480)
    assert K[0, 0] == pytest.approx(320 / np.tan(np.radians(50)))
    assert (K[0, 2], K[1, 2]) == (320, 240)
    a = SimpleNamespace(k='300,301,310,250', fov=None)
    assert ps.known_k(a, 640, 480)[1, 1] == 301
    assert ps.known_k(SimpleNamespace(k=None, fov=None), 640, 480) is None
