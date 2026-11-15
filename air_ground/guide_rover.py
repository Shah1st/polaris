"""Click on the view from the air, the rover drives there.

    python air_ground/guide_rover.py --phone http://<iPhone>:8080 --rover http://<rover>:8080

then open http://127.0.0.1:8097 : the top view of the floor (stable while the phone moves),
the rover as the air camera sees it, a click sets the goal, "Stop" cancels it.

    GET  /            the page
    GET  /top.jpg     the top view with the rover, the goal and the path drawn on it
    GET  /state       where things stand (JSON): status, rover, goal, link, phone, latency
    POST /goal        {"x": m, "y": m} in the floor frame, or {"fx": 0..1, "fy": 0..1}: the
                      fractions of the top view's width and height, from its top left corner
    POST /cancel      forget the goal and stop the rover

The phone must be CamStream in ARKit mode (iphone-camera/README.md). Nothing here plans around
obstacles yet: the rover goes straight to the point.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from camstream import PoseStream
from guidance import ClockSync, Guide, GuideConfig, RoverClient, RoverError
from pose_camera import PoseError
from rover_marker import RoverFinder

PAGE = """<!doctype html><meta charset="utf-8"><title>Air guide</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#eee}
img{max-width:100%;border:1px solid #444;cursor:crosshair;image-rendering:pixelated}
button{font:inherit;padding:6px 14px;margin-right:8px}#s{margin:8px 0;min-height:3em}</style>
<div id="s">…</div><img id="top" alt="top view from the air">
<p><button id="stop">Stop</button>Click on the view: the rover drives there.</p>
<script>
const top_ = document.getElementById('top'), s = document.getElementById('s');
async function view(){ for(;;){ try{ const r = await fetch('/top.jpg?'+Date.now());
  if(r.ok){ const u = URL.createObjectURL(await r.blob()); const old = top_.src;
    top_.src = u; if(old.startsWith('blob:')) URL.revokeObjectURL(old); } }catch(e){}
  await new Promise(f=>setTimeout(f,150)); } }
async function state(){ for(;;){ try{ const d = await (await fetch('/state')).json();
  s.textContent = d.status + ' · phone: ' + d.phone + (d.dropped ? ' · ' + d.dropped : '')
    + (d.latency_ms != null ? ' · picture ' + d.latency_ms + ' ms old' : ''); }catch(e){
  s.textContent = 'guide not reachable'; } await new Promise(f=>setTimeout(f,300)); } }
const post = (p, b) => fetch(p, {method:'POST', body: JSON.stringify(b)});
top_.onclick = e => { const r = top_.getBoundingClientRect();
  post('/goal', {fx:(e.clientX-r.left)/r.width, fy:(e.clientY-r.top)/r.height}); };
document.getElementById('stop').onclick = () => post('/cancel', {});
view(); state();
</script>"""


def to_pixel(x: float, y: float, bounds, res: float) -> tuple[int, int]:
    """A floor point (metres) in the top view of `bounds` at `res` m per pixel."""
    return round((x - bounds[0]) / res), round((bounds[3] - y) / res)


def draw(top: np.ndarray, bounds, res: float, snap: dict, trail, seen) -> np.ndarray:
    """The top view with what the guide knows: the rover's trail, the rover as the link says
    (green arrow), as the last picture saw it (yellow ring), the path and the goal (red)."""
    out = top.copy()
    if len(trail) > 1:
        pts = np.array([to_pixel(x, y, bounds, res) for x, y in trail], np.int32)
        cv2.polylines(out, [pts], False, (255, 180, 0), 1, cv2.LINE_AA)
    rover = snap.get('rover')
    goal = snap.get('goal')
    if goal is not None:
        chain = ([rover[:2]] if rover else []) + [*snap.get('waypoints', []), goal]
        pts = np.array([to_pixel(x, y, bounds, res) for x, y in chain], np.int32)
        cv2.polylines(out, [pts], False, (60, 60, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            out, to_pixel(*goal, bounds, res), (60, 60, 255), cv2.MARKER_TILTED_CROSS, 14, 2
        )
    if seen is not None:
        cv2.circle(out, to_pixel(seen.x, seen.y, bounds, res), 9, (0, 230, 255), 1, cv2.LINE_AA)
    if rover is not None:
        x, y, heading = rover[0], rover[1], math.radians(rover[2])
        tip = (x + 0.15 * math.cos(heading), y + 0.15 * math.sin(heading))
        cv2.arrowedLine(
            out,
            to_pixel(x, y, bounds, res),
            to_pixel(*tip, bounds, res),
            (80, 255, 80),
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )
    return out


class GuideService:
    """The loop (phone -> rover finder -> guide -> rover) and its small HTTP server."""

    def __init__(
        self,
        phone_url,
        rover,
        finder,
        cfg=None,
        bounds=(-1.5, -1.5, 1.5, 1.5),
        res=0.005,
        port=8097,
        host='127.0.0.1',
    ):
        self.stream = PoseStream(phone_url)
        self.rover = rover
        self.finder = finder
        self.guide = Guide(rover, cfg or GuideConfig())
        self.bounds, self.res = bounds, res
        self.sync = ClockSync()
        self.lock = threading.Lock()
        self.top_jpg: bytes | None = None
        self.phone = 'no picture yet'
        self.rover_error: str | None = None
        self.latency_ms: int | None = None
        self.trail: deque[tuple[float, float]] = deque(maxlen=2000)
        self._session: int | None = None
        self._seen = None
        self._stop = threading.Event()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    # -- the loop -------------------------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self.stream.close()
        self.server.shutdown()
        self.server.server_close()

    def _loop(self) -> None:
        seq, next_state, next_draw = -1, 0.0, 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_state:
                next_state = now + 0.1
                self._read_rover()
            frame = self.stream.latest(max_age=2.0)
            fresh = frame is not None and frame.seq != seq
            if fresh:
                seq = frame.seq
                self._look(frame)
            elif frame is None:
                self.phone = self.stream.error or 'no fresh picture from the phone'
            with self.lock:
                try:
                    self.guide.step(time.monotonic())
                except RoverError as e:
                    self.rover_error = str(e)
            if fresh and now >= next_draw:
                next_draw = now + 0.15
                self._draw(frame)
            self._stop.wait(0.01 if fresh else 0.03)

    def _read_rover(self) -> None:
        try:
            t0 = time.monotonic()
            state = self.rover.state()
            t1 = time.monotonic()
        except RoverError as e:
            self.rover_error = str(e)
            return
        self.rover_error = None
        with self.lock:
            self.guide.on_state(state, (t0 + t1) / 2)
            here = self.guide.rover_on_floor()
        if here and (not self.trail or math.dist(self.trail[-1], (here.x, here.y)) > 0.01):
            self.trail.append((here.x, here.y))

    def _look(self, frame) -> None:
        """One new picture: find the rover in it and tell the guide."""
        pose = frame.pose
        self._seen = None
        if pose is None:
            self.phone = frame.pose_error or 'no pose'
            return
        if pose.session != self._session:  # the phone's floor frame started again
            with self.lock:
                if self._session is not None:
                    self.guide.cancel(
                        'stopped: the phone restarted its tracking (new floor frame)'
                    )
                self.guide.link.reset()
            self.trail.clear()
            self._session = pose.session
        t_picture = self.sync.local_time(pose.t, frame.received)
        self.latency_ms = round(self.sync.latency(pose.t, frame.received) * 1000)
        if not pose.floor_known:
            self.phone = 'floor not found yet: move the phone slowly above the floor'
            return
        if pose.tracking != 'normal':
            self.phone = f'tracking {pose.tracking}: sightings ignored'
            return
        image = frame.image
        if image is None:
            self.phone = 'picture not decodable'
            return
        seen = self.finder(image, pose)
        self._seen = seen
        self.phone = f'ok, {self.stream.fps:.0f} fps, {pose.c[2]:.2f} m above the floor, ' + (
            'rover seen' if seen else 'rover NOT in the picture'
        )
        with self.lock:
            self.guide.on_sighting(seen, t_picture)

    def _draw(self, frame) -> None:
        pose, image = frame.pose, frame.image
        if pose is None or image is None or not pose.floor_known:
            return
        try:
            top = pose.floor_view(image, self.bounds, self.res, fill=(40, 40, 40))
        except PoseError:
            return
        with self.lock:
            snap = self.guide.snapshot()
        out = draw(top, self.bounds, self.res, snap, list(self.trail), self._seen)
        ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.top_jpg = jpg.tobytes()

    # -- commands -------------------------------------------------------------------------

    def set_goal(self, body: dict) -> tuple[float, float]:
        if 'fx' in body or 'fy' in body:
            fx, fy = float(body['fx']), float(body['fy'])
            if not (0 <= fx <= 1 and 0 <= fy <= 1):
                raise ValueError('fx and fy are fractions of the view, 0 to 1')
            x0, y0, x1, y1 = self.bounds
            x, y = x0 + fx * (x1 - x0), y1 - fy * (y1 - y0)
        else:
            x, y = float(body['x']), float(body['y'])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('the goal must be finite numbers')
        with self.lock:
            self.guide.set_goal(x, y)
        return x, y

    def cancel(self) -> None:
        with self.lock:
            try:
                self.guide.cancel('cancelled')
            except RoverError as e:
                self.rover_error = str(e)

    def state(self) -> dict:
        with self.lock:
            snap = self.guide.snapshot()
        snap.update(
            phone=self.phone,
            rover_error=self.rover_error,
            latency_ms=self.latency_ms,
            bounds=list(self.bounds),
            res=self.res,
        )
        if self.rover_error:
            snap['status'] = f'rover not reachable: {self.rover_error}'
        return snap

    # -- HTTP -----------------------------------------------------------------------------

    def _handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200) -> None:
                self._send(code, json.dumps(obj).encode(), 'application/json')

            def do_GET(self):  # noqa: N802
                path = self.path.split('?')[0]
                if path == '/':
                    self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
                elif path == '/top.jpg':
                    if service.top_jpg is None:
                        self._json({'error': service.phone}, 503)
                    else:
                        self._send(200, service.top_jpg, 'image/jpeg')
                elif path == '/state':
                    self._json(service.state())
                else:
                    self._json({'error': 'unknown path'}, 404)

            def do_POST(self):  # noqa: N802
                path = self.path.split('?')[0]
                try:
                    length = int(self.headers.get('Content-Length') or 0)
                    body = json.loads(self.rfile.read(length) or b'{}')
                    if not isinstance(body, dict):
                        raise ValueError('the body must be a JSON object')
                    if path == '/goal':
                        x, y = service.set_goal(body)
                        self._json({'ok': True, 'goal': [x, y]})
                    elif path == '/cancel':
                        service.cancel()
                        self._json({'ok': True})
                    else:
                        self._json({'error': 'unknown path'}, 404)
                except (KeyError, TypeError, ValueError) as e:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 400)

        return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phone', required=True, help='CamStream, e.g. http://172.20.10.6:8080')
    ap.add_argument(
        '--rover', required=True, help="the rover's server, e.g. http://172.20.10.2:8080"
    )
    ap.add_argument('--port', type=int, default=8097)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--marker-id', type=int, default=0, help='ArUco 4x4 id on the rover (0)')
    ap.add_argument('--marker-size', type=float, default=0.10, help='printed side, m (0.10)')
    ap.add_argument('--marker-height', type=float, default=0.12, help='above the floor, m (0.12)')
    ap.add_argument(
        '--marker-forward',
        type=float,
        default=90.0,
        help="degrees from the marker's top edge to the rover's front (90: the top "
        'of the print points forward)',
    )
    ap.add_argument(
        '--marker-offset',
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=('X', 'Y'),
        help="the marker's centre in the rover's frame, m (forward, left)",
    )
    ap.add_argument('--speed', type=float, default=0.15, help='m/s (0.15)')
    ap.add_argument(
        '--bounds',
        type=float,
        nargs=4,
        default=(-1.5, -1.5, 1.5, 1.5),
        metavar=('X0', 'Y0', 'X1', 'Y1'),
        help='the top view, floor frame, m',
    )
    ap.add_argument('--res', type=float, default=0.005, help='m per pixel of the top view')
    a = ap.parse_args()
    finder = RoverFinder(a.marker_id, a.marker_size, a.marker_height, a.marker_forward)
    cfg = GuideConfig(speed=a.speed, marker_offset=tuple(a.marker_offset))
    service = GuideService(
        a.phone, RoverClient(a.rover), finder, cfg, tuple(a.bounds), a.res, a.port, a.host
    )
    service.start()
    print(
        f'air guide: http://{a.host}:{service.port}  phone: {a.phone}  rover: {a.rover}',
        flush=True,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.cancel()
        service.close()


if __name__ == '__main__':
    main()
