"""perception_service.py without the model: frame sources (MJPEG stream, snapshot URL) and the
HTTP API with a fake perceiver. Fake servers on 127.0.0.1:8132-8136 and 8285-8289.

Run: python -m pytest rover_mapper/test_perception_service.py -q -p no:cacheprovider
"""

import importlib.util
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
    stays open and silent; 'snapshot': one JPEG per GET (`body` if given), answered `delay` s
    late (a slow link). `requests` counts the GETs."""

    def __init__(self, port, kind, period=0.05, frames=10**6, path='/video', body=None,
                 delay=0.0):
        self.kind, self.period, self.frames = kind, period, frames
        self.body, self.delay, self.requests = body, delay, 0
        src = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                src.requests += 1
                if src.kind == 'snapshot':
                    time.sleep(src.delay)
                    b = jpeg() if src.body is None else src.body
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
        self.url = f'http://127.0.0.1:{port}{path}'

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
        self.frames = []

    def run(self, frame):
        t, seq, rgb = frame
        self.frames.append(frame)
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


def args(source, **kw):
    a = SimpleNamespace(source=source, max_age=2.0, cam_height=0.22, range=1.0,
                        band_low=0.04, band_high=0.40, k=None, fov=None, model='fake',
                        calib=None, rect_fov=100.0)
    a.__dict__.update(kw)
    return a


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


# -- raw fisheye frames (the robot's /snapshot.jpg) rectified by the service itself ----------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(REPO, 'teleop', 'calib', 'plumbline.json')


def camera_hub_class():
    """teleop/server.py's CameraHub, imported (not copied): the reference rectification."""
    spec = importlib.util.spec_from_file_location('teleop_server_ref',
                                                  os.path.join(REPO, 'teleop', 'server.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CameraHub


def pinhole_scene(w=640, h=480):
    """A checkerboard with diagonal stripes: straight lines everywhere (BGR)."""
    v, u = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.where(((u // 40) + (v // 40)) % 2 == 0, 220, 30)
    img[..., 1] = np.where(((u + v) // 25) % 2 == 0, 200, 50)
    img[..., 2] = (u * 255 // w).astype(np.uint8)
    return img


def fisheye_of(scene, fov=100.0):
    """What the robot's lens would see of a centred pinhole `scene` of `fov` degrees, with the
    lens model of plumbline.json (theta_d = theta*(1+k1*theta^2), r = f*theta_d), written
    independently of OpenCV's fisheye code."""
    with open(CALIB) as f:
        c = json.load(f)
    w, h = c['size']
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    du, dv = u - c['cx'], v - c['cy']
    r = np.hypot(du, dv)
    td = r / c['f']
    th = td.copy()
    for _ in range(20):  # Newton: theta*(1+k1*theta^2) = theta_d
        th -= (th * (1 + c['k1'] * th**2) - td) / (1 + 3 * c['k1'] * th**2)
    rr = np.tan(th) / np.maximum(r, 1e-9)
    fo = (w / 2) / np.tan(np.radians(fov / 2))
    mx = (fo * du * rr + w / 2).astype(np.float32)
    my = (fo * dv * rr + h / 2).astype(np.float32)
    return cv2.remap(scene, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def test_rectifier_matches_teleop_camerahub():
    hub = camera_hub_class()('http://127.0.0.1:8289/video_feed', CALIB, 100.0, snapshots=False)
    rect = ps.Rectifier(CALIB, 100.0)
    assert np.array_equal(rect.maps[0], hub._maps[0]) and np.array_equal(rect.maps[1], hub._maps[1])
    assert np.allclose(rect.K, hub.rect_K)
    assert np.allclose(ps.known_k(SimpleNamespace(k=None, fov=100.0), 640, 480), hub.rect_K)
    scene = pinhole_scene()
    raw = cv2.imencode('.jpg', fisheye_of(scene), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    ours = rect(ps.decode_image(raw))  # RGB
    assert ours.shape == (480, 640, 3)
    # The same image as /video_rect: encoded as the dashboard does (quality 90), the very same
    # bytes.
    q90 = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    assert cv2.imencode('.jpg', ours[..., ::-1], q90)[1].tobytes() == hub.rectified(1, raw)
    # And the lens is really removed: the pinhole scene comes back (centre, away from the
    # black corners the fisheye does not cover).
    c = (slice(120, 360), slice(160, 480))
    assert np.abs(ours[c].astype(int) - scene[c][..., ::-1].astype(int)).mean() < 12
    with pytest.raises(ValueError):
        rect(np.zeros((240, 320, 3), np.uint8))  # not the lens model's size


def test_lens_options():
    a = SimpleNamespace(calib=CALIB, rect_fov=90.0, fov=None, k=None)
    assert isinstance(ps.lens(a), ps.Rectifier) and a.fov == 90.0  # --rect-fov is the field
    assert ps.lens(SimpleNamespace(calib=None, rect_fov=100.0, fov=None, k=None)) is None
    with pytest.raises(SystemExit):
        ps.lens(SimpleNamespace(calib=CALIB, rect_fov=100.0, fov=80.0, k=None))
    with pytest.raises(SystemExit):
        ps.lens(SimpleNamespace(calib=CALIB, rect_fov=100.0, fov=None, k='1,1,1,1'))
    with pytest.raises(SystemExit):
        ps.lens(SimpleNamespace(calib='/nonexistent.json', rect_fov=100.0, fov=None, k=None))


def raw_jpeg():
    return cv2.imencode('.jpg', fisheye_of(pinhole_scene()))[1].tobytes()


def test_snapshot_source_one_request_per_fresh_look():
    """The robot's /snapshot.jpg through a slow link: no probe request, one GET per fresh look,
    its cost in fetch_s, and nothing fetched again for a non-fresh request or /debug.jpg."""
    body = raw_jpeg()
    src = Source(8285, 'snapshot', path='/snapshot.jpg', body=body, delay=0.8)
    rect = ps.Rectifier(CALIB, 100.0)
    cam = ps.Camera(src.url, timeout=5.0, rectify=rect)
    a = args(src.url, max_age=0.5, calib=CALIB, fov=100.0)  # the link is slower than max_age
    perceiver = FakePerceiver()
    srv = serve(cam, perceiver, a, 8286)
    try:
        assert cam.mode == 'snapshot' and src.requests == 0  # known from the URL: no probe
        seen = []
        for k in range(3):
            t0 = time.time()
            code, b = get('http://127.0.0.1:8286/obstacles?fresh=1')
            assert code == 200, b
            r = json.loads(b)
            assert src.requests == k + 1
            assert 0.8 <= r['fetch_s'] < time.time() - t0 + 0.01
            assert r['t'] >= t0 + 0.8 and r['intrinsics'] == 'known'
            seen.append(r['frame'])
        assert seen == sorted(set(seen))
        assert np.array_equal(perceiver.frames[-1][2], rect(ps.decode_image(body)))
        code, b = get('http://127.0.0.1:8286/obstacles')  # newer than max_age: reused
        assert code == 200 and json.loads(b)['frame'] == seen[-1] and src.requests == 3
        assert get('http://127.0.0.1:8286/debug.jpg')[0] == 200 and src.requests == 3
        assert perceiver.calls == 3
    finally:
        srv.shutdown()
        srv.server_close()
        cam.close()
        src.close()


def post(url, body, timeout=5):
    req = urllib.request.Request(url, data=body, method='POST',
                                 headers={'Content-Type': 'image/jpeg'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_post_obstacles_with_the_callers_image():
    """The driver already downloaded the raw snapshot: it sends it, the service never opens a
    connection to the robot."""
    dead = Source(8287, 'snapshot', path='/snapshot.jpg')  # counts requests, must stay at 0
    rect = ps.Rectifier(CALIB, 100.0)
    cam = ps.Camera(dead.url, rectify=rect)
    perceiver = FakePerceiver()
    srv = serve(cam, perceiver, args(dead.url, calib=CALIB, fov=100.0), 8288)
    try:
        body = raw_jpeg()
        t0 = time.time()
        code, b = post('http://127.0.0.1:8288/obstacles', body)
        assert code == 200, b
        r = json.loads(b)
        for k in ('t', 'age_s', 'res', 'x0', 'y0', 'cells', 'frame', 'fetch_s'):
            assert k in r
        assert r['age_s'] == 0 and t0 <= r['t'] <= time.time() and r['fetch_s'] is None
        assert np.array_equal(perceiver.frames[-1][2], rect(ps.decode_image(body)))
        # /debug.jpg right after shows that same image and result, still without a fetch.
        assert get('http://127.0.0.1:8288/debug.jpg')[0] == 200 and perceiver.calls == 1
        assert dead.requests == 0
        assert post('http://127.0.0.1:8288/obstacles', b'not a jpeg')[0] == 400
        small = cv2.imencode('.jpg', np.zeros((240, 320, 3), np.uint8))[1].tobytes()
        code, b = post('http://127.0.0.1:8288/obstacles', small)
        assert code == 400 and '640x480' in json.loads(b)['error']
        assert post('http://127.0.0.1:8288/nope', body)[0] == 404
        assert dead.requests == 0
    finally:
        srv.shutdown()
        srv.server_close()
        cam.close()
        dead.close()
