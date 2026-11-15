"""D405Source (real D405 through d405_server.py) without the device and without the model:
an in-process fake D405 server with a non-trivial depth -> colour extrinsic, a stub robot
camera, and scripts/fake_d405.py against a fake simulator. Fake servers on 127.0.0.1:8245,
8248 and 8249.

Run: python -m pytest rover_mapper/test_d405_source.py -q -p no:cacheprovider
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rovermap.source import (MODE_BOTH, MODE_D405, MODE_RGB, D405Source, d405_in_rgb,  # noqa: E402
                             mode_settings, mount_pose, pinhole_K, sensor_inputs)

SCALE = 1e-4                                   # D405 default raw unit, metres
W, H = 160, 120                                # small D405 images: fast tests
K_DEPTH = np.array([[110.0, 0, 79.5], [0, 110.0, 60.5], [0, 0, 1]])
K_COLOR = np.array([[95.0, 0, 82.0], [0, 96.0, 58.0], [0, 0, 1]])


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


# Non-trivial factory extrinsics, X_color = R X_depth + t (a few degrees, a few cm).
R_DC = rot(1, 4.0) @ rot(0, -2.0) @ rot(2, 1.5)
T_DC = np.array([-0.018, 0.004, 0.002])
T_CD = np.eye(4)
T_CD[:3, :3], T_CD[:3, 3] = R_DC, T_DC

# One measured pixel of the depth sensor, at an exact raw depth.
UD, VD, RAW = 61, 47, 4000                     # 0.4 m


def depth_raw():
    d = np.zeros((H, W), np.uint16)
    d[VD, UD] = RAW
    return d


def point_depth_frame():
    z = RAW * SCALE
    return np.array([(UD - K_DEPTH[0, 2]) / K_DEPTH[0, 0] * z, (VD - K_DEPTH[1, 2]) / K_DEPTH[1, 1] * z, z])


def project(K, P):
    return int(round(K[0, 0] * P[0] / P[2] + K[0, 2])), int(round(K[1, 1] * P[1] / P[2] + K[1, 2]))


def frame_npz(seq, age_s, raw=None):
    bgr = np.zeros((H, W, 3), np.uint8)
    bgr[..., 2] = 255                          # red in OpenCV order: RGB must come out (255, 0, 0)
    buf = io.BytesIO()
    np.savez(buf, color_jpg=cv2.imencode(".jpg", bgr)[1].ravel(),
             depth_png=cv2.imencode(".png", depth_raw() if raw is None else raw,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].ravel(),
             depth_scale=np.float64(SCALE), K_color=K_COLOR, K_depth=K_DEPTH, R_dc=R_DC, t_dc=T_DC,
             age_s=np.float64(age_s), seq=np.int64(seq))
    return buf.getvalue()


class FakeD405:
    """GET /frame as d405_server.py; `age` (s) and `fail` (503 message) set by the test;
    `same_seq` serves the same frame again (polled faster than it captures). Also a rover
    GET /state with the pan-tilt."""

    def __init__(self, port=8245):
        self.age, self.fail, self.same_seq, self.seq, self.pan, self.requests = 0.02, None, False, 0, 0.0, 0
        me = self

        class Hd(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                if self.path == "/state":
                    body, ctype, code = json.dumps({"gimbal": {"pan": me.pan, "tilt": 0.0}}).encode(), \
                        "application/json", 200
                elif me.fail:
                    body, ctype, code = json.dumps({"error": me.fail}).encode(), "application/json", 503
                else:
                    me.requests += 1
                    if not me.same_seq:
                        me.seq += 1
                    body, ctype, code = frame_npz(me.seq, me.age), "application/x-npz", 200
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), Hd)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class StubCamera:
    """FrameSource stand-in: the robot camera's newest RGB frame (640x480)."""

    def __init__(self):
        self.seq, self.error, self.stopped = 0, None, False
        self.frame = np.full((480, 640, 3), 90, np.uint8)

    def tick(self):
        self.seq += 1

    def latest(self):
        return self.seq, (self.frame if self.seq else None)

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake():
    f = FakeD405()
    yield f
    f.close()


def wait(cond, timeout=5.0):
    t = time.time()
    while time.time() - t < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.02)
    return cond()


def first_snapshot(src):
    return wait(lambda: (lambda s: s if s[1] is not None else None)(src.snapshot()))


def test_mount_pitch_sign_and_pinhole_k():
    # Pitch > 0 = the D405 looks further down: a point straight ahead of the robot
    # camera is then ABOVE the D405's image centre (y < 0, OpenCV y points down).
    T = mount_pose(0, 0, 0, 10)
    P = np.linalg.inv(T) @ np.array([0, 0, 1.0, 1])
    assert P[1] < -0.1 and P[2] > 0.9
    # Yaw > 0 = looks further left: the same point is then to the right of its centre.
    P = np.linalg.inv(mount_pose(0, 0, 0, 0, 10)) @ np.array([0, 0, 1.0, 1])
    assert P[0] > 0.1
    # Roll > 0 = right side lower: a point to the robot camera's right shows up lower.
    # Roll > 0 = right side lower: the D405's x axis points right AND down, so a point to
    # the robot camera's right shows up higher (y < 0) in the D405.
    P = np.linalg.inv(mount_pose(0, 0, 0, 0, 0, 10)) @ np.array([1.0, 0, 1.0, 1])
    assert P[1] < -0.1
    assert np.allclose(mount_pose(0.1, 0.2, 0.3)[:3, 3], [0.1, 0.2, 0.3])
    # teleop/server.py rect_K: f = (w/2)/tan(fov/2), centre (w/2, h/2).
    K = pinhole_K(640, 480, 100.0)
    assert K[0, 0] == pytest.approx(320 / np.tan(np.radians(50)))
    assert (K[0, 2], K[1, 2]) == (320, 240)


def test_d405_mode_dict_and_pixel(fake):
    src = D405Source(fake.url, MODE_D405, max_hz=20)
    try:
        seq, snap = first_snapshot(src)
        img, sd, K, T, d_true = sensor_inputs(snap, MODE_D405)
        assert set(snap) == {"d405_rgb", "d405_depth", "d405_K", "d405_T", "d405_depth_true"}
        assert img.shape == (H, W, 3) and img[..., 0].mean() > 200 and img[..., 2].mean() < 50   # RGB order
        assert sd.dtype == np.float32 and sd.shape == (H, W)
        assert np.array_equal(K, K_COLOR) and np.array_equal(T, np.eye(4)) and d_true is None
        # The one measured point, moved into the colour camera with the factory extrinsics.
        Pc = R_DC @ point_depth_frame() + T_DC
        u, v = project(K_COLOR, Pc)
        assert (u, v) != (UD, VD)                    # the extrinsic really moves it
        assert sd[v, u] == pytest.approx(Pc[2], abs=1e-5)
        assert np.count_nonzero(sd) == 1
        assert src.error is None
        cam_h, rng = mode_settings(MODE_D405, 0.21, 0.06, (0.07, 0.8))
        assert cam_h == 0.06
        # Live panel: colour image over its depth picture.
        assert src.latest()[1].shape == (2 * H, W, 3)
    finally:
        src.stop()


def test_both_mode_lands_point_in_robot_camera(fake):
    cam = StubCamera()
    mount = mount_pose(0.01, 0.15, 0.08, 10.0, -3.0, 2.0)
    src = D405Source(fake.url, MODE_BOTH, rgb_source=cam, rgb_fov=100.0, mount=mount, max_hz=20)
    try:
        assert wait(lambda: src._d405 is not None)
        cam.tick()
        seq, snap = first_snapshot(src)
        img, sd, K, T, d_true = sensor_inputs(snap, MODE_BOTH)
        assert img is cam.frame and d_true is None and np.array_equal(T, np.eye(4))
        Kr = pinhole_K(640, 480, 100.0)
        assert np.allclose(K, Kr)
        assert np.array_equal(snap["d405_K"], K_DEPTH)            # raw depth stays in the depth frame
        assert sd.shape == (480, 640)
        # depth camera -> D405 colour camera -> robot camera.
        Pr = mount[:3, :3] @ (R_DC @ point_depth_frame() + T_DC) + mount[:3, 3]
        u, v = project(Kr, Pr)
        assert 0 <= u < 640 and 0 <= v < 480
        assert sd[v, u] == pytest.approx(Pr[2], abs=1e-5)
        assert np.count_nonzero(sd) == 1
        assert d405_in_rgb(snap) is sd                             # computed once per snapshot
        # One snapshot per robot camera frame, not per poll.
        time.sleep(0.2)
        assert src.snapshot()[0] == seq
        cam.tick()
        assert wait(lambda: src.snapshot()[0] == seq + 1)
        # Panel: robot image over the reprojected depth.
        assert src.latest()[1].shape == (960, 640, 3)
    finally:
        src.stop()
    assert cam.stopped


def test_stale_d405_frames_are_not_used(fake):
    fake.age = 1.5                                  # older than max_age at the server already
    src = D405Source(fake.url, MODE_D405, max_hz=20, max_age=1.0)
    cam = StubCamera()
    both = D405Source(fake.url, MODE_BOTH, rgb_source=cam, mount=np.eye(4), max_hz=20, max_age=1.0)
    try:
        assert wait(lambda: fake.requests >= 4)
        cam.tick()
        assert src.snapshot() == (0, None) and both.snapshot() == (0, None)
        assert "trop ancienne" in (src.error or "") and "trop ancienne" in (both.error or "")
        fake.age = 0.02                             # fresh again: used
        assert first_snapshot(src)[0] >= 1
        assert first_snapshot(both)[0] == 1
    finally:
        src.stop()
        both.stop()


def test_frame_received_then_too_old_is_skipped(fake):
    # A frame young on arrival but kept too long (the D405 server stopped) is not paired.
    cam = StubCamera()
    src = D405Source(fake.url, MODE_BOTH, rgb_source=cam, mount=np.eye(4), max_hz=20, max_age=0.3)
    try:
        assert wait(lambda: src._d405 is not None)
        fake.fail = "pas de D405"
        time.sleep(0.5)
        cam.tick()
        assert src.snapshot() == (0, None)
        assert src.error
    finally:
        src.stop()


def test_same_server_frame_gives_one_snapshot(fake):
    fake.same_seq, fake.seq = True, 7
    src = D405Source(fake.url, MODE_D405, max_hz=20)
    try:
        assert first_snapshot(src)[0] == 1
        assert wait(lambda: fake.requests >= 5)
        assert src.snapshot()[0] == 1
    finally:
        src.stop()


def test_errors_reported(fake):
    fake.fail = "pas de D405 branchée"
    src = D405Source(fake.url + "/frame", MODE_D405, max_hz=20)
    try:
        assert wait(lambda: src.error and "pas de D405 branchée" in src.error)
        assert src.snapshot() == (0, None)
    finally:
        src.stop()
    down = D405Source("http://127.0.0.1:8248", MODE_D405, max_hz=20)   # nothing listens there
    try:
        assert wait(lambda: down.error and down.error.startswith("D405 :"))
    finally:
        down.stop()
    # Modes that need what was not given say so.
    alone = D405Source(fake.url, MODE_BOTH, max_hz=20)
    nomount = D405Source(fake.url, MODE_BOTH, rgb_source=StubCamera(), max_hz=20)
    try:
        assert "--source" in alone.error and "--d405-mount" in nomount.error
        assert alone.snapshot() == (0, None)
    finally:
        alone.stop()
        nomount.stop()


def test_mode_switch_drops_old_dict_and_rgb_mode(fake):
    cam = StubCamera()
    src = D405Source(fake.url, MODE_D405, rgb_source=cam, mount=np.eye(4), max_hz=20)
    try:
        first_snapshot(src)
        src.mode = MODE_RGB
        assert src.snapshot()[1] is None            # D405 dict never fed as an RGB one
        n = fake.requests
        cam.tick()
        seq, snap = first_snapshot(src)
        img, sd, K, T, _ = sensor_inputs(snap, MODE_RGB)
        assert img is cam.frame and sd is None
        time.sleep(0.4)
        assert fake.requests - n <= 1               # robot camera alone: the D405 is not polled
    finally:
        src.stop()


def test_gimbal_pan_warning(fake):
    fake.pan = 30.0
    src = D405Source(fake.url, MODE_BOTH, rgb_source=StubCamera(), mount=np.eye(4), max_hz=20,
                     rover_url=fake.url)
    try:
        assert wait(lambda: src.gimbal_warning and "pan +30" in src.gimbal_warning)
    finally:
        src.stop()


def test_fake_d405_script_serves_contract():
    """scripts/fake_d405.py on a fake simulator /snapshot (8248), read back by D405Source (8249)."""
    rng = np.random.default_rng(0)
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 0] = 250                                   # strong red: catches a channel swap
    rgb[..., 1] = np.linspace(0, 120, 64, dtype=np.uint8)[None, :]
    depth = np.zeros((48, 64), np.float32)
    depth[10:30, 20:40] = 0.35 + 0.001 * rng.random((20, 20), np.float32)
    K = np.array([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]])

    class Hs(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            buf = io.BytesIO()
            np.savez(buf, d405_rgb=rgb, d405_depth=depth, d405_K=K)
            self.send_response(200)
            self.send_header("Content-Length", str(len(buf.getvalue())))
            self.end_headers()
            self.wfile.write(buf.getvalue())

    sim = ThreadingHTTPServer(("127.0.0.1", 8248), Hs)
    sim.daemon_threads = True
    threading.Thread(target=sim.serve_forever, daemon=True).start()
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "scripts", "fake_d405.py"),
                          "--sim", "http://127.0.0.1:8248", "--port", "8249"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    src = None
    try:
        def info():
            try:
                with urllib.request.urlopen("http://127.0.0.1:8249/info", timeout=2) as r:
                    return json.loads(r.read())
            except Exception:
                return None
        i = wait(info, 15.0)
        assert i and i["depth_scale_m"] == 1e-4 and i["depth_to_color"]["R"] == np.eye(3).tolist()
        with urllib.request.urlopen("http://127.0.0.1:8249/health", timeout=2) as r:
            h = json.loads(r.read())
        assert h["device"] and h["mode"].startswith("64x48@")
        with urllib.request.urlopen("http://127.0.0.1:8249/frame?fresh=1", timeout=5) as r:
            assert r.headers["Content-Type"] == "application/x-npz" and int(r.headers["X-Age-Ms"]) >= 0
        src = D405Source("http://127.0.0.1:8249", MODE_D405, max_hz=4)
        _, snap = first_snapshot(src)
        sd = snap["d405_depth"]
        assert np.abs(sd - depth).max() <= 0.6e-4      # identity extrinsics, 0.1 mm units
        assert np.abs(snap["d405_rgb"].astype(int) - rgb).mean() < 25 and snap["d405_rgb"][..., 0].mean() > 200
        assert np.allclose(snap["d405_K"], K)
    finally:
        if src is not None:
            src.stop()
        p.terminate()
        p.wait(5)
        sim.shutdown()
        sim.server_close()
