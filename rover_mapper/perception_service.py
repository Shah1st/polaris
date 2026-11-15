"""Rover perception service: obstacle cells in front of the rover from its single camera.

On each request it takes the newest camera frame, runs Depth Anything 3 on that one image
(no tracking: the mapper's tracking was unreliable on the real rover), fits the floor, sets the
metric scale from the known camera height and classifies 5 cm cells (rovermap/perception.py).

    GET /obstacles    {"t", "age_s", "res", "x0", "y0", "cells", ...}: -1/0/1 grid in the ROVER
                      frame (x forward, y left, metres; cell [i][j] at x0 + j*res, y0 + i*res),
                      or 503 when no frame is newer than --max-age s.
                      ?fresh=1: wait (up to --max-age s) for a frame that arrives after the
                      request, so a picture taken while the rover was still moving is not used.
    GET /debug.jpg    the image with obstacle (red) and free (green) cells drawn on it
    GET /health       model, source, last frame age, last result summary

Real rover (the teleop dashboard serves the lens-corrected stream):
    /home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \\
        --source http://127.0.0.1:8082/video_rect --cam-height 0.22 --port 8095

--source: MJPEG stream URL, single-image (snapshot) URL — JPEG/PNG, or the simulator's .npz
snapshot (its 'rgb') — fetched on demand, or a still image file (for trying it offline).
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


class Camera:
    """Newest frame of a source, with its arrival time (time.time()).

    MJPEG stream: read continuously in a thread, only the last frame kept. Anything else from
    an http URL: one GET per grab (a snapshot is only taken when needed). File: read once,
    "arrives" at every grab."""

    def __init__(self, source: str, timeout: float = 3.0):
        self.source, self.timeout = source, timeout
        self.mode = 'file' if '://' not in source else None  # None: not known yet
        self.error: str | None = None
        self._cond = threading.Condition()
        self._frame: tuple[float, int, np.ndarray] | None = None
        self._seq = 0
        self._stop = False
        if self.mode == 'file':
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f'cannot read {source}')
            self._still = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            threading.Thread(target=self._run, daemon=True).start()

    def _publish(self, rgb: np.ndarray) -> None:
        with self._cond:
            self._seq += 1
            self._frame = (time.time(), self._seq, rgb)
            self._cond.notify_all()

    def _run(self) -> None:
        import requests

        while not self._stop:
            try:
                with requests.get(self.source, stream=True, timeout=5) as r:
                    r.raise_for_status()
                    if not r.headers.get('Content-Type', '').startswith('multipart'):
                        self.mode = 'snapshot'
                        self.error = None
                        return
                    self.mode = 'stream'
                    self.error = None
                    buf = b''
                    for chunk in r.iter_content(chunk_size=16384):
                        if self._stop:
                            return
                        buf += chunk
                        end = buf.rfind(b'\xff\xd9')
                        if end < 0:
                            continue
                        start = buf.rfind(b'\xff\xd8', 0, end)
                        if start < 0:
                            buf = buf[end + 2 :]
                            continue
                        jpg, buf = buf[start : end + 2], buf[end + 2 :]
                        rgb = decode_image(jpg)
                        if rgb is not None:
                            self._publish(rgb)
            except Exception as e:  # network hiccup or source not up yet: retry
                self.error = f'{type(e).__name__}: {e}'
                time.sleep(1.0)

    def grab(self, not_before: float, wait: float) -> tuple[float, int, np.ndarray] | None:
        """(arrival time, sequence number, RGB) of a frame that arrived at or after
        `not_before`, waiting up to `wait` s for it; None if there is none."""
        if self.mode == 'file':
            self._seq += 1
            return time.time(), self._seq, self._still
        if self.mode == 'snapshot':
            import requests

            try:
                r = requests.get(self.source, timeout=self.timeout)
                r.raise_for_status()
                rgb = decode_image(r.content)
            except Exception as e:
                self.error = f'{type(e).__name__}: {e}'
                return None
            if rgb is None:
                self.error = 'undecodable image'
                return None
            self._publish(rgb)
            return self._frame
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

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            fresh = q.get('fresh', ['0'])[0] not in ('0', '', 'false')
            try:
                if url.path == '/obstacles':
                    out = self._perceive(fresh)
                    if out is None:
                        return
                    r = out['result']
                    body = r.to_json()
                    body.update({
                        't': out['t'],
                        'age_s': round(time.time() - out['t'], 3),
                        'dt_s': round(out['dt'], 3),
                        'frame': out['seq'],
                        'cam_height': args.cam_height,
                        'range': args.range,
                        'band': [args.band_low, args.band_high],
                        'intrinsics': 'known' if (args.k or args.fov) else 'da3',
                    })
                    self._json(body)
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
    ap.add_argument('--max-age', type=float, default=2.0,
                    help='503 when no frame is newer than this (s)')
    ap.add_argument('--model', default='depth-anything/DA3-BASE')
    ap.add_argument('--res', type=int, default=504, help='DA3 processing resolution')
    ap.add_argument('--cpu-threads', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--port', type=int, default=8095)
    args = ap.parse_args()

    cam = Camera(args.source)
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
