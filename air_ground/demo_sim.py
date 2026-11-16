"""The control dashboard without any hardware, on this computer.

    python air_ground/demo_sim.py        then open http://127.0.0.1:8097

Starts the simulated rover of rover/rover_server.py (with its simulated camera), a fake phone
that films it from a camera that keeps moving (the pictures and poses CamStream sends in ARKit
mode, and LiDAR depth maps on /depth as a LiDAR iPhone does), and the guide with its
dashboard. The scene: a concrete floor with boxes of several heights, and one more box that
the air camera cannot see (as if under a table) but the rover's D405 does, once it comes
near: the route then changes. Click on the map: the simulated rover drives there. Nothing
here talks to the real rover or to a real phone.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rover'))

import rover_server  # noqa: E402
import synthetic as syn  # noqa: E402
from beam import parse_grid  # noqa: E402
from fake_phone import BOUNDARY, HEIGHT, SHIFT, SIDE, TURN, floor_pose  # noqa: E402
from guidance import RoverClient  # noqa: E402
from guide_rover import GuideService  # noqa: E402
from pose_camera import Pose  # noqa: E402
from rover_core import Config  # noqa: E402
from rover_marker import RoverFinder  # noqa: E402

# boxes seen from the air: centre x, y (floor frame, m), side, height
BOXES = [
    (0.80, 0.30, 0.22, 0.18),
    (-0.40, 0.45, 0.18, 0.10),
    (0.55, -0.55, 0.16, 0.25),
    (-0.50, -0.40, 0.24, 0.06),
    (0.15, 0.70, 0.14, 0.14),
]
HIDDEN = (-0.15, 0.0, 0.14, 0.12)  # only the rover's D405 sees it: try a goal at (-0.6, 0.05)


def concrete(seed: int = 5) -> np.ndarray:
    """A grey concrete floor: blotches, grain, faint seams every 0.5 m (texture enough for
    the plane sweep, no colours that could pass for objects)."""
    rng = np.random.default_rng(seed)
    n = round((syn.FLOOR_BOUNDS[2] - syn.FLOOR_BOUNDS[0]) / syn.FLOOR_RES)
    low = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 40)
    low = low / (np.abs(low).max() + 1e-6)
    mid = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 6)
    mid = mid / (np.abs(mid).max() + 1e-6)
    grain = rng.normal(0, 1, (n, n)).astype(np.float32)
    g = 118 + 16 * low + 9 * mid + 7 * grain
    step = round(0.5 / syn.FLOOR_RES)
    g[::step, :] -= 22
    g[:, ::step] -= 22
    out = np.clip(np.stack([g + 4, g + 2, g], -1), 0, 255).astype(np.uint8)
    return out


def corners(cx, cy, side, height):
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, height)


def phone_pose(t: float) -> dict:
    """A hand-held phone: it circles slowly 2 m above the area, never looking straight down."""
    eye = (0.15 + 0.35 * math.cos(0.45 * t), 0.30 * math.sin(0.45 * t), 2.0)
    d = syn.arkit_pose(eye, (0.15, 0.05, 0.0), roll_deg=15 * math.sin(0.3 * t), t=t)
    d['sid'] = 7
    return d


def make_phone(app, floor):
    """Like tests/fake_phone.py, with boxes, and /depth: the LiDAR depth of the same frame."""
    printed = syn.marker_image(0)
    tops = [(syn.box_top(i + 11), x, y, 0.0, s, h) for i, (x, y, s, h) in enumerate(BOXES)]
    solid = [corners(*b) for b in BOXES]
    t0 = time.monotonic()

    class Phone(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            if self.path.split('?')[0] == '/depth':
                self._depth()
                return
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.end_headers()
            try:
                while True:
                    t = time.monotonic() - t0
                    x, y, heading = floor_pose(app.sim_board)
                    d = phone_pose(t)
                    marker = (printed, x, y, heading - math.pi / 2, SIDE * 1.5, HEIGHT)
                    image = syn.render(Pose.from_arkit(d), floor, marker, tops)
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

        def _depth(self):
            t = time.monotonic() - t0
            d = phone_pose(t)
            x, y, _ = floor_pose(app.sim_board)
            metres, k = syn.render_depth(Pose.from_arkit(d), solid, rover=(x, y, 0.12, HEIGHT))
            body = metres.astype('<f4').tobytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Depth-Width', str(metres.shape[1]))
            self.send_header('X-Depth-Height', str(metres.shape[0]))
            self.send_header('X-Depth-Intrinsics', ','.join(f'{v:.3f}' for v in k))
            self.send_header('X-Pose', syn.header(d))
            self.send_header('X-Age-Ms', '30')
            self.end_headers()
            self.wfile.write(body)

    return Phone


class DemoD405:
    """Stands for the rover's D405 and its local grid (GET /local_grid): what a 60 degree
    cone 0.7 m ahead of the simulated rover sees, every box included, in its odometry frame."""

    def __init__(self, app, period=0.2):
        self.app, self.period = app, period
        self.reach = 0.7  # m ahead: the hidden box shows up only once the rover is near
        self.count, self.error, self.last_s = 0, None, 0.02
        self._grid, self._at = None, 0.0
        self.boxes = [corners(*b) for b in [*BOXES, HIDDEN]]

    def latest(self, max_age=None):
        now = time.monotonic()
        if self._grid is not None and now - self._at < self.period:
            return self._grid
        b = self.app.sim_board
        o = self.app.rover.snapshot().get('odom') or {}
        ox, oy = float(o.get('x', b.x)), float(o.get('y', b.y))
        oh = math.radians(float(o.get('heading_deg', math.degrees(b.yaw))))
        # odometry -> the board's frame -> the floor (fake_phone's turn and shift)
        dth = b.yaw - oh
        cd, sd = math.cos(dth), math.sin(dth)
        c, s = math.cos(TURN), math.sin(TURN)

        def to_floor(x, y):
            dx, dy = x - ox, y - oy
            bx, by = b.x + cd * dx - sd * dy, b.y + sd * dx + cd * dy
            return c * bx - s * by + SHIFT[0], s * bx + c * by + SHIFT[1]

        n, res = 50, 0.05
        x0, y0 = ox - n * res / 2, oy - n * res / 2
        jj, ii = np.meshgrid(np.arange(n), np.arange(n))
        x, y = x0 + (jj + 0.5) * res, y0 + (ii + 0.5) * res
        dx, dy = x - ox, y - oy
        ahead = dx * math.cos(oh) + dy * math.sin(oh)
        side = -dx * math.sin(oh) + dy * math.cos(oh)
        cone = (
            (ahead > 0.07)
            & (ahead <= self.reach)
            & (np.abs(np.arctan2(side, ahead)) < math.radians(30))
        )
        cells = np.where(cone, 0, -1).astype(np.int8)
        fxs, fys = to_floor(x, y)  # each cell's centre on the floor
        for bx0, by0, bx1, by1, _ in self.boxes:
            inside = (fxs >= bx0) & (fxs <= bx1) & (fys >= by0) & (fys <= by1)
            cells[cone & inside] = 1
        self.count += 1
        self._grid = parse_grid({'res': res, 'x0': x0, 'y0': y0, 'cells': cells.tolist(),
                                 'pose': [ox, oy, oh], 'age_s': 0.05, 'frames': self.count},
                                now)  # fmt: skip
        self._at = now
        return self._grid

    def close(self):
        pass


def serve(handler, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(('127.0.0.1', port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8097, help='the dashboard (8097)')
    ap.add_argument('--rover-port', type=int, default=8098, help='the simulated rover (8098)')
    ap.add_argument('--phone-port', type=int, default=8099, help='the fake phone (8099)')
    a = ap.parse_args()
    args = SimpleNamespace(sim_drift=0.0, width=640, height=480, fps=15.0, quality=70, camera='0',
                           no_camera=False, d405='off')  # fmt: skip
    config = os.path.join(tempfile.mkdtemp(prefix='polaris-demo-'), 'rover_config.json')
    app = rover_server.App(Config(), config, args, serial_port=None)
    app.set_mode('sim')
    rover = serve(rover_server.make_handler(app), a.rover_port)
    phone = serve(make_phone(app, concrete()), a.phone_port)
    service = GuideService(
        f'http://127.0.0.1:{a.phone_port}',
        RoverClient(f'http://127.0.0.1:{a.rover_port}'),
        RoverFinder(0, SIDE, HEIGHT),
        bounds=(-1.2, -1.2, 1.4, 1.2),
        res=0.01,  # as guide_rover.py's default
        colour=False,
        port=a.port,
    )
    service.beam.close()
    service.beam = DemoD405(app)  # the simulated rover has no D405 of its own
    service.start()
    print(f'SIMULATION, no hardware: dashboard http://127.0.0.1:{service.port}', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        phone.shutdown()
        rover.shutdown()
        app.close()


if __name__ == '__main__':
    main()
