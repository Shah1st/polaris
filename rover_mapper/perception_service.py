"""Rover perception service: obstacle cells in front of the rover from its single camera.

On each request it takes the newest camera frame, runs Depth Anything 3 on that one image
(no tracking: the mapper's tracking was unreliable on the real rover), fits the floor, sets the
metric scale from the known camera height and classifies 5 cm cells (rovermap/perception.py).

    GET /obstacles    {"t", "age_s", "res", "x0", "y0", "cells", ...}: -1/0/1 grid in the ROVER
                      frame (x forward, y left, metres; cell [i][j] at x0 + j*res, y0 + i*res),
                      or 503 when no frame is newer than --max-age s.
                      ?fresh=1: wait (up to --max-age s) for a frame that arrives after the
                      request, so a picture taken while the rover was still moving is not used.
                      With a snapshot source: ONE new GET of it per fresh request, its duration
                      in "fetch_s" (the link's cost); without fresh=1 a snapshot newer than
                      --max-age is reused (no request).
    POST /obstacles   body: one JPEG the caller already has (raw fisheye with --calib, else
                      already rectified); same JSON, "age_s" 0, "t" = reception time. The
                      service opens no connection of its own.
    GET /debug.jpg    the image with obstacle (red) and free (green) cells drawn on it
    GET /health       model, source, last frame age, last result summary

Real rover on a slow link (one request per look, no dashboard in between: the robot's raw
fisheye snapshot is rectified here, as teleop/server.py does for /video_rect):
    /home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \\
        --source http://127.0.0.1:18080/snapshot.jpg --calib teleop/calib/plumbline.json \\
        --rect-fov 100 --fov 100 --cam-height 0.22 --range 1.5 --port 8095

--source: MJPEG stream URL, single-image (snapshot) URL — JPEG/PNG, or the simulator's .npz
snapshot (its 'rgb') — fetched on demand, or a still image file (for trying it offline).
A URL whose path ends in .jpg/.jpeg/.png/.npz is a snapshot without a probe request.
Binds 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rovermap import perception as per  # noqa: E402


def decode_image(body: bytes) -> np.ndarray | None:
    """RGB frame from a JPEG/PNG, or from the simulator's snapshot (.npz, key 'rgb')."""
    if body[:2] == b'PK':
        with np.load(io.BytesIO(body)) as z:
            return np.ascontiguousarray(z['rgb']) if 'rgb' in z.files else None
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Rectifier:
    """The robot's fisheye lens model (teleop/calib/plumbline.json) -> a centred pinhole image
    of `rect_fov` degrees horizontally: the same K and remap tables as teleop/server.py
    CameraHub (its /video_rect), applied to a decoded frame instead of a JPEG."""

    def __init__(self, calib: str, rect_fov: float = 100.0):
        with open(calib) as f:
            c = json.load(f)
        w, h = c['size']
        self.size = (w, h)
        K = np.array([[c['f'], 0, c['cx']], [0, c['f'], c['cy']], [0, 0, 1.0]])
        fo = (w / 2) / np.tan(np.radians(rect_fov / 2))
        self.K = np.array([[fo, 0, w / 2], [0, fo, h / 2], [0, 0, 1.0]])
        self.maps = cv2.fisheye.initUndistortRectifyMap(
            K, np.array([c['k1'], 0, 0, 0.0]), np.eye(3), self.K, (w, h), cv2.CV_16SC2)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        w, h = self.size
        if img.shape[:2] != (h, w):  # another camera or size: the lens model does not apply
            raise ValueError(f'image {img.shape[1]}x{img.shape[0]}, lens model is for {w}x{h}')
        return cv2.remap(img, self.maps[0], self.maps[1], cv2.INTER_LINEAR)


def lens(args) -> Rectifier | None:
    """--calib/--rect-fov: the Rectifier, and --rect-fov becomes the known field of view."""
    if not args.calib:
        return None
    if args.k:
        raise SystemExit('--k with --calib: the rectified image is given by --rect-fov')
    if args.fov is not None and args.fov != args.rect_fov:
        raise SystemExit(f'--fov {args.fov:g} differs from --rect-fov {args.rect_fov:g}')
    if not os.path.exists(args.calib):
        raise SystemExit(f'cannot read {args.calib}')
    args.fov = args.rect_fov
    return Rectifier(args.calib, args.rect_fov)


SNAPSHOT_EXT = ('.jpg', '.jpeg', '.png', '.npz')
MAX_POST = 16 << 20  # bytes: a POSTed image larger than this is refused


class Camera:
    """Newest frame of a source, with its arrival time (time.time()).

    MJPEG stream: read continuously in a thread, only the last frame kept. Anything else from
    an http URL: one GET per grab that needs a newer frame than the last one (a snapshot is
    only taken when needed); a URL ending in an image extension is known to be one, without a
    probe request. File: read once, "arrives" at every grab. `rectify` (a Rectifier) is applied
    to every decoded frame."""

    def __init__(self, source: str, timeout: float = 3.0, rectify=None):
        self.source, self.timeout, self.rectify = source, timeout, rectify
        self.mode = 'file' if '://' not in source else None  # None: not known yet
        if self.mode is None and urlparse(source).path.lower().endswith(SNAPSHOT_EXT):
            self.mode = 'snapshot'
        self.error: str | None = None
        self._cond = threading.Condition()
        self._frame: tuple[float, int, np.ndarray] | None = None
        self._seq = 0
        self._fetch_s: dict[int, float] = {}  # sequence number -> duration of its GET (s)
        self._stop = False
        self._detected = threading.Event()
        if self.mode == 'file':
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f'cannot read {source}')
            self._still = self.prepare(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif self.mode is None:
            threading.Thread(target=self._run, daemon=True).start()

    def prepare(self, rgb: np.ndarray | None) -> np.ndarray | None:
        """The decoded frame, rectified when a lens model is given (ValueError: wrong size)."""
        return rgb if rgb is None or self.rectify is None else self.rectify(rgb)

    def _decode(self, body: bytes) -> np.ndarray | None:
        try:
            rgb = self.prepare(decode_image(body))
        except ValueError as e:
            self.error = str(e)
            return None
        if rgb is None:
            self.error = 'undecodable image'
        return rgb

    def put(self, rgb: np.ndarray) -> tuple[float, int, np.ndarray]:
        """A frame given by a caller (POST /obstacles): newest frame from now on."""
        return self._publish(rgb)

    def fetch_s(self, seq: int) -> float | None:
        """How long the GET of snapshot `seq` took; None for other frames."""
        return self._fetch_s.get(seq)

    def _publish(self, rgb: np.ndarray) -> tuple[float, int, np.ndarray]:
        with self._cond:
            self._seq += 1
            self._frame = (time.time(), self._seq, rgb)
            self._cond.notify_all()
            return self._frame

    def _run(self) -> None:
        while not self._stop:
            try:
                with urllib.request.urlopen(self.source, timeout=5) as r:
                    if not r.headers.get('Content-Type', '').startswith('multipart'):
                        self.mode = 'snapshot'
                        self.error = None
                        self._detected.set()
                        return
                    self.mode = 'stream'
                    self.error = None
                    self._detected.set()
                    buf = b''
                    while not self._stop:
                        chunk = r.read1(16384)
                        if not chunk:
                            break  # stream closed by the server: reconnect
                        buf = (buf + chunk)[-(4 << 20) :]
                        end = buf.rfind(b'\xff\xd9')
                        if end < 0:
                            continue
                        start = buf.rfind(b'\xff\xd8', 0, end)
                        if start < 0:
                            buf = buf[end + 2 :]
                            continue
                        jpg, buf = buf[start : end + 2], buf[end + 2 :]
                        rgb = self._decode(jpg)
                        if rgb is not None:
                            self._publish(rgb)
            except Exception as e:  # network hiccup or source not up yet: retry
                self.error = f'{type(e).__name__}: {e}'
            time.sleep(1.0)

    def grab(self, not_before: float, wait: float) -> tuple[float, int, np.ndarray] | None:
        """(arrival time, sequence number, RGB) of a frame that arrived at or after
        `not_before`, waiting up to `wait` s for it; None if there is none."""
        if self.mode == 'file':
            return self._publish(self._still)
        if self.mode is None:
            self._detected.wait(wait)
        if self.mode == 'snapshot':
            f = self._frame
            if f is not None and f[0] >= not_before:  # recent enough: no request on the link
                return f
            t0 = time.time()
            try:
                with urllib.request.urlopen(self.source, timeout=self.timeout) as r:
                    body = r.read()
            except Exception as e:
                self.error = f'{type(e).__name__}: {e}'
                return None
            rgb = self._decode(body)
            if rgb is None:
                return None
            self.error = None
            frame = self._publish(rgb)
            self._fetch_s[frame[1]] = frame[0] - t0
            while len(self._fetch_s) > 16:
                self._fetch_s.pop(min(self._fetch_s))
            return frame
        deadline = time.time() + wait
        with self._cond:
            while self._frame is None or self._frame[0] < not_before:
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._cond.wait(left)
            return self._frame

    def last_age(self) -> float | None:
        f = self._frame
        return None if f is None else time.time() - f[0]

    def close(self) -> None:
        self._stop = True


def known_k(args, w: int, h: int) -> np.ndarray | None:
    """Intrinsics of the source image (pixels) from --K or --fov; None: use DA3's own."""
    if args.k:
        fx, fy, cx, cy = (float(v) for v in args.k.split(','))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    if args.fov:
        f = (w / 2) / np.tan(np.radians(args.fov / 2))
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return None


class Perceiver:
    """The model, loaded once, and the last result (a request on the same frame reuses it)."""

    def __init__(self, args):
        from rovermap.mapper import Mapper, net_intrinsics

        self.args = args
        self._net_intrinsics = net_intrinsics
        # Only its model loading and _infer are used; no tracking, no moving-object mask.
        self.mapper = Mapper(model_name=args.model, process_res=args.res,
                             cam_height=args.cam_height, mask_moving=False,
                             cpu_threads=args.cpu_threads, device=args.device)
        self.lock = threading.Lock()
        self.last: dict | None = None  # {'seq', 't', 'result', 'image', 'K', 'dt'}

    def run(self, frame) -> dict:
        t, seq, rgb = frame
        with self.lock:
            if self.last is not None and self.last['seq'] == seq and self.last['t'] == t:
                return self.last
            t0 = time.time()
            depth, conf, _, K_net, imgs = self.mapper._infer([rgb])
            depth, conf, image = depth[0], conf[0], imgs[0]
            K = known_k(self.args, rgb.shape[1], rgb.shape[0])
            K = K_net[0] if K is None else self._net_intrinsics(K, rgb.shape[:2], depth.shape)
            a = self.args
            result = per.perceive(depth, K, cam_height=a.cam_height, conf=conf,
                                  max_range=a.range, min_range=a.min_range,
                                  band=(a.band_low, a.band_high))
            self.last = {'seq': seq, 't': t, 'result': result, 'image': image, 'K': K,
                         'dt': time.time() - t0}
            return self.last


def make_handler(cam: Camera, perceiver: Perceiver, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # one line per request is noise at 1-3 Hz
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), 'application/json')

        def _perceive(self, fresh: bool):
            now = time.time()
            frame = cam.grab(now if fresh else now - args.max_age, args.max_age)
            if frame is None:
                self._json({'error': 'no frame newer than %.1f s' % args.max_age,
                            'source_error': cam.error}, 503)
                return None
            return perceiver.run(frame)

        def _obstacles(self, out: dict, age_s: float, fetch_s: float | None) -> None:
            body = out['result'].to_json()
            body.update({
                't': out['t'],
                'age_s': age_s,
                'fetch_s': None if fetch_s is None else round(fetch_s, 3),
                'dt_s': round(out['dt'], 3),
                'frame': out['seq'],
                'cam_height': args.cam_height,
                'range': args.range,
                'band': [args.band_low, args.band_high],
                'intrinsics': 'known' if (args.k or args.fov) else 'da3',
            })
            self._json(body)

        def do_POST(self) -> None:  # noqa: N802
            """POST /obstacles: the caller's own JPEG (it already paid the link for it)."""
            try:
                if urlparse(self.path).path != '/obstacles':
                    self._json({'error': 'not found'}, 404)
                    return
                n = int(self.headers.get('Content-Length') or 0)
                if not 0 < n <= MAX_POST:
                    self._json({'error': f'JPEG body of 1 to {MAX_POST} bytes expected'}, 400)
                    return
                data = self.rfile.read(n)
                try:
                    rgb = cam.prepare(decode_image(data))
                except ValueError as e:  # not the lens model's size
                    self._json({'error': str(e)}, 400)
                    return
                if rgb is None:
                    self._json({'error': 'undecodable image'}, 400)
                    return
                self._obstacles(perceiver.run(cam.put(rgb)), 0.0, None)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self._json({'error': f'{type(e).__name__}: {e}'}, 500)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            fresh = q.get('fresh', ['0'])[0] not in ('0', '', 'false')
            try:
                if url.path == '/obstacles':
                    out = self._perceive(fresh)
                    if out is None:
                        return
                    self._obstacles(out, round(time.time() - out['t'], 3),
                                    cam.fetch_s(out['seq']))
                elif url.path == '/debug.jpg':
                    out = self._perceive(fresh)
                    if out is None:
                        return
                    bgr = cv2.cvtColor(np.asarray(out['image'], np.uint8), cv2.COLOR_RGB2BGR)
                    img = per.draw(bgr, out['result'], out['K'])
                    self._send(200, cv2.imencode('.jpg', img)[1].tobytes(), 'image/jpeg')
                elif url.path == '/health':
                    last = perceiver.last
                    self._json({
                        'ok': True,
                        'source': args.source,
                        'source_mode': cam.mode,
                        'calib': args.calib,
                        'source_error': cam.error,
                        'frame_age_s': cam.last_age(),
                        'model': args.model,
                        'last': None if last is None else {
                            't': last['t'], 'dt_s': round(last['dt'], 3),
                            'floor': last['result'].floor, 'scale': last['result'].scale,
                            'stats': last['result'].stats,
                        },
                    })
                else:
                    self._json({'error': 'not found'}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # keep serving; the client treats any error as "nothing seen"
                self._json({'error': f'{type(e).__name__}: {e}'}, 500)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='MJPEG/snapshot URL or image file')
    ap.add_argument('--cam-height', type=float, default=0.22,
                    help='camera height above the floor (m): sets the metric scale')
    ap.add_argument('--range', type=float, default=1.5,
                    help='cells kept up to this distance (m); single-view depth is only '
                    'trustworthy near')
    ap.add_argument('--min-range', type=float, default=0.0,
                    help='points nearer than this (m) are ignored (the rover\'s own front)')
    ap.add_argument('--band-low', type=float, default=0.04, help='obstacle band bottom (m)')
    ap.add_argument('--band-high', type=float, default=0.40, help='obstacle band top (m)')
    ap.add_argument('--fov', type=float, default=None,
                    help='known horizontal field of view (deg) of a centred pinhole image, '
                    'e.g. 100 for the dashboard\'s /video_rect; default: DA3\'s own intrinsics')
    ap.add_argument('--k', default=None, help='known intrinsics fx,fy,cx,cy (source pixels)')
    ap.add_argument('--calib', default=None,
                    help='fisheye lens model (teleop/calib/plumbline.json): the source gives RAW '
                    'frames (e.g. the robot\'s /snapshot.jpg), rectified here as the dashboard\'s '
                    '/video_rect')
    ap.add_argument('--rect-fov', type=float, default=100.0,
                    help='horizontal field (deg) of the image rectified with --calib; it is then '
                    'the known field of view')
    ap.add_argument('--fetch-timeout', type=float, default=10.0,
                    help='timeout (s) of one snapshot GET; the robot link takes 1.3-4 s')
    ap.add_argument('--max-age', type=float, default=2.0,
                    help='503 when no frame is newer than this (s)')
    ap.add_argument('--model', default='depth-anything/DA3-BASE')
    ap.add_argument('--res', type=int, default=504, help='DA3 processing resolution')
    ap.add_argument('--cpu-threads', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--port', type=int, default=8095)
    args = ap.parse_args()

    rectify = lens(args)
    cam = Camera(args.source, args.fetch_timeout, rectify)
    print(f'Chargement du modele {args.model}...', flush=True)
    perceiver = Perceiver(args)
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(cam, perceiver, args))
    print(f'Perception : http://127.0.0.1:{args.port}/obstacles  (source {args.source})',
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == '__main__':
    main()
