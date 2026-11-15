"""Click on the view from the air, the rover drives there.

    python air_ground/guide_rover.py --phone http://<iPhone>:8080 --rover http://<rover>:8080

then open http://127.0.0.1:8097 : the control dashboard (web/index.html). The map is the floor
seen from the air, built up as the phone moves (floor_map.py): the floor's picture, obstacles
in red from their height, with the rover as the air camera sees it; a click sets the goal.
Without --rover it only maps: move the phone above the scene and watch the map grow.
Next to it: the two cameras read by the browser straight from the phone and from the rover
(no relay, no processing in between), manual driving, the rover's telemetry and the
commands it received.

    GET  /            the dashboard
    GET  /top.jpg     the map with the rover, the goal and the path drawn on it; with
                      ?view=grid, the 5 cm grid the planner uses instead of the photos
    GET  /view.jpg    the last picture alone, laid flat on the floor (no memory)
    GET  /map.json    the obstacle grid: 5 cm cells, 1 obstacle / 0 floor / -1 unknown,
                      row 0 = south, and the height of each cell
    POST /map/reset   start the map again
    GET  /state       where things stand (JSON): status, rover, goal, link, phone, latency;
                      `air`: what the last picture gave; `rover_state`: the rover's last
                      GET /state as it is, its age and the round trip to the rover
    POST /goal        {"x": m, "y": m} in the floor frame, or {"fx": 0..1, "fy": 0..1}: the
                      fractions of the top view's width and height, from its top left corner
    POST /cancel      forget the goal and stop the rover if this guide was driving it
    POST /stop        forget the goal and stop the rover, whoever was driving it
    GET  /rover/...   /commands, /config, /d405/status of the rover, passed on as they are
    POST /rover/...   /cmd, /gimbal of the rover (manual driving from the dashboard); the
                      rover's command log shows them as "dashboard"

The rover's server sends no CORS headers: its JSON goes through this server, its pictures do
not (the page's <img> read them from the rover itself).

The phone must be CamStream in ARKit mode (iphone-camera/README.md). Nothing here plans around
obstacles yet: the rover goes straight to the point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from camstream import PoseStream
from floor_map import FloorMap
from guidance import ClockSync, Guide, GuideConfig, RoverClient, RoverError
from planner import RoutePlanner
from pose_camera import PoseError
from rover_marker import RoverFinder

INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'index.html')
# What the dashboard may ask the rover through this server.
ROVER_GET = ('/commands', '/config', '/d405/status')
ROVER_POST = ('/cmd', '/gimbal')


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


def shade_blocked(picture: np.ndarray, blocked: np.ndarray, k: int) -> np.ndarray:
    """The map picture with the cells the rover's centre may not enter (the obstacles grown
    by its radius; `blocked` row 0 = south, `k` picture pixels per cell) shaded violet: the
    route stays out of them."""
    rows, cols = picture.shape[:2]
    mask = np.kron(blocked[::-1].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    full = np.zeros((rows, cols), bool)
    h, w = min(rows, mask.shape[0]), min(cols, mask.shape[1])
    full[:h, :w] = mask[:h, :w]
    out = picture.copy()
    out[full] = (0.7 * out[full] + 0.3 * np.array([200, 60, 160])).astype(np.uint8)
    return out


class NoRover:
    """No rover yet: the guide only maps. Every order is refused, nothing is sent."""

    base = None

    def state(self) -> dict:
        raise RoverError('no rover: mapping only (start with --rover to drive)')

    def path(self, points, speed) -> None:
        raise RoverError('no rover: mapping only')

    def stop(self) -> None:
        raise RoverError('no rover: mapping only')


class MapWorker:
    """Builds the floor map in its own thread, from the newest picture only: a slow update
    never holds the guide's loop, it skips pictures instead."""

    def __init__(self, fmap: FloorMap) -> None:
        self.fmap = fmap
        self.lock = threading.Lock()
        self.picture: np.ndarray | None = None
        self.grid_picture: np.ndarray | None = None  # the planner's grid, drawn
        self.error: str | None = None
        self.last_s: float | None = None  # seconds the last update took
        # s between two runs of the team's colour detection (~0.1 s each); None: never
        self.colour_every: float | None = 1.0
        self._colour_at = 0.0
        self._rover_xy = None  # where the rover was last seen: left out of the colours
        self._job = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def offer(self, image, pose, rover_xy) -> None:
        self._job = (image, pose, rover_xy)
        self._wake.set()

    def reset(self) -> None:
        with self.lock:
            self.fmap.reset()
            self.picture = None
            self.grid_picture = None

    def close(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            job, self._job = self._job, None
            if job is None:
                continue
            t0 = time.monotonic()
            try:
                if job[2] is not None:
                    self._rover_xy = job[2]
                with self.lock:
                    self.fmap.add(*job)
                    if self.colour_every is not None and t0 - self._colour_at >= self.colour_every:
                        self._colour_at = t0
                        self.fmap.update_colour(self._rover_xy)
                    self.picture = self.fmap.render()
                    self.grid_picture = self.fmap.render_grid()
                self.error = None
            except Exception as e:  # noqa: BLE001 - the map must never stop the guide
                self.error = f'{type(e).__name__}: {e}'
            self.last_s = time.monotonic() - t0

    def grid(self) -> dict:
        with self.lock:
            return self.fmap.grid(0.05)

    def stats(self) -> dict:
        with self.lock:
            s = self.fmap.stats()
        s['update_s'] = None if self.last_s is None else round(self.last_s, 2)
        s['error'] = self.error
        return s


class GuideService:
    """The loop (phone -> rover finder -> guide -> rover) and its small HTTP server."""

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The map's place on the floor (it may move when the map centres itself)."""
        return self.mapper.fmap.bounds

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
        follow=False,
        radius=0.15,
        colour=True,
    ):
        """`follow`: the map centres itself where the phone first looks (`bounds` then only
        gives its size); otherwise it covers `bounds` of the floor frame. `radius`: the
        circle around the whole rover, by which obstacles are grown for the route.
        `colour`: the team's colour detection adds its obstacles to the heights' (it cannot
        tell a flat colour on the floor from an object: off for a painted floor)."""
        self.planner = RoutePlanner(radius=radius)
        self.plans = 0  # routes computed for the current goal and before
        self.plan_note: str | None = None
        self._next_plan = 0.0
        self.stream = PoseStream(phone_url)
        rover = rover if rover is not None else NoRover()
        self.rover = rover
        self.mapping_only = isinstance(rover, NoRover)
        self.mapper = MapWorker(FloorMap(bounds, res, follow=follow))
        if not colour:
            self.mapper.colour_every = None
        self.view_jpg: bytes | None = None
        self.finder = finder
        self.guide = Guide(rover, cfg or GuideConfig())
        self.res = res
        self.sync = ClockSync()
        self.lock = threading.Lock()
        self.top_jpg: bytes | None = None
        self.grid_jpg: bytes | None = None  # the same, on the planner's grid
        self.phone = 'no picture yet'
        self.rover_error: str | None = None
        self.latency_ms: int | None = None
        self.trail: deque[tuple[float, float]] = deque(maxlen=2000)
        self.phone_url = phone_url.rstrip('/')
        self.rover_url = getattr(rover, 'base', None)
        # the dashboard's own orders (manual driving) carry another name in the rover's log
        self.manual = RoverClient(self.rover_url, name='dashboard') if self.rover_url else None
        self.rover_state: dict | None = None
        self.rover_state_at: float | None = None
        self.rover_rtt: deque[float] = deque(maxlen=100)
        self.air: dict = {}
        self.marker: dict | None = None  # the rover's marker in the last picture that had it
        self.seen_at: float | None = None
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
        self.mapper.close()
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
            if now >= self._next_plan:
                self._next_plan = now + 0.5
                self._replan()
            with self.lock:
                try:
                    self.guide.step(time.monotonic())
                except RoverError as e:
                    self.rover_error = str(e)
            if fresh and now >= next_draw:
                next_draw = now + 0.15
                self._draw(frame)
            self._stop.wait(0.01 if fresh else 0.03)

    def _replan(self, force: bool = False) -> None:
        """The route around the obstacles, from where the rover is now: computed for a new
        goal, and again whenever the map blocks the way being driven (an obstacle appeared,
        or the rover drifted towards one). None found: the rover stops."""
        with self.lock:
            goal = self.guide.goal
            here = self.guide.rover_on_floor()
            route = [*self.guide.waypoints, goal] if goal is not None else None
            sent = self.guide.sent
        if goal is None or here is None:
            return
        grid = self.mapper.grid()
        start = (here.x, here.y)
        if not force and sent is not None and self.planner.blocked(grid, route, start) is None:
            return  # the way being driven is still clear
        new = self.planner.route(grid, start, goal)
        with self.lock:
            if self.guide.goal != goal:
                return  # the goal changed meanwhile
            self.plans += 1
            if new is None:
                self.plan_note = 'no way to the goal around the obstacles'
                try:
                    self.guide.cancel('stopped: no way to the goal around the obstacles')
                except RoverError as e:
                    self.rover_error = str(e)
                return
            self.plan_note = f'route of {len(new)} point(s) around the obstacles'
            self.guide.set_route(new[:-1])

    def _read_rover(self) -> None:
        try:
            t0 = time.monotonic()
            state = self.rover.state()
            t1 = time.monotonic()
        except RoverError as e:
            self.rover_error = str(e)
            return
        self.rover_error = None
        self.rover_state, self.rover_state_at = state, t1
        self.rover_rtt.append(t1 - t0)
        with self.lock:
            self.guide.on_state(state, (t0 + t1) / 2)
            here = self.guide.rover_on_floor()
        if here and (not self.trail or math.dist(self.trail[-1], (here.x, here.y)) > 0.01):
            self.trail.append((here.x, here.y))

    def _look(self, frame) -> None:
        """One new picture: find the rover in it and tell the guide."""
        air = {'fps': round(self.stream.fps, 1)}
        try:
            self._look_at(frame, air)
        finally:
            self.air = air  # whole, never half filled, for a /state read meanwhile

    def _look_at(self, frame, air: dict) -> None:
        pose = frame.pose
        self._seen = None
        if pose is None:
            self.phone = frame.pose_error or 'no pose'
            return
        air.update(
            tracking=pose.tracking,
            floor_known=pose.floor_known,
            height_m=round(float(pose.c[2]), 3),
            session=pose.session,
            size=[pose.width, pose.height],
        )
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
        self.mapper.offer(image, pose, None if seen is None else (seen.x, seen.y))
        if seen is not None:
            self.seen_at = time.monotonic()
            self.marker = {
                'x': round(seen.x, 3),
                'y': round(seen.y, 3),
                'heading_deg': round(math.degrees(seen.heading), 1),
                'side_px': round(seen.side_px, 1),
                'scale_error': round(seen.scale_error, 3),
            }
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
            view = pose.floor_view(image, self.bounds, self.res, fill=(40, 40, 40))
        except PoseError:
            return
        with self.lock:
            snap = self.guide.snapshot()
        trail = list(self.trail)
        ok, jpg = cv2.imencode('.jpg', draw(view, self.bounds, self.res, snap, trail, self._seen))
        if ok:
            self.view_jpg = jpg.tobytes()
        grid = self.planner.last_grid
        for name, picture in (
            ('top_jpg', self.mapper.picture),
            ('grid_jpg', self.mapper.grid_picture),
        ):
            if picture is not None and grid is not None and grid['x0'] == self.bounds[0]:
                k = max(1, round(grid['res'] / self.res))
                picture = shade_blocked(picture, self.planner.blocked_cells(grid), k)
            out = draw(picture if picture is not None else view, self.bounds, self.res, snap,
                       trail, self._seen)  # fmt: skip
            ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                setattr(self, name, jpg.tobytes())

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
            self.guide.route_ready = False  # nothing sent before the route is planned
            self.plan_note = None
        self._replan(force=True)  # the route before anything is sent to the rover
        if self.guide.goal is None and self.plan_note:
            raise ValueError(f'{self.plan_note}: pick another point')
        return x, y

    def cancel(self) -> None:
        with self.lock:
            try:
                self.guide.cancel('cancelled')
            except RoverError as e:
                self.rover_error = str(e)

    def stop(self) -> None:
        """The dashboard's STOP: forget the goal and stop the rover, whoever drives it."""
        with self.lock:
            self.guide.goal, self.guide.sent, self.guide.waypoints = None, None, []
            self.guide.status = 'stopped from the dashboard'
        try:
            (self.manual or self.rover).stop()
        except RoverError as e:
            self.rover_error = str(e)
            raise

    def state(self) -> dict:
        now = time.monotonic()
        with self.lock:
            snap = self.guide.snapshot()
        rtt = sorted(self.rover_rtt)
        trail = list(self.trail)
        snap.update(
            phone=self.phone,
            rover_error=self.rover_error,
            latency_ms=self.latency_ms,
            bounds=list(self.bounds),
            res=self.res,
            phone_url=self.phone_url,
            rover_url=self.rover_url,
            air=dict(
                self.air,
                marker=self.marker,
                seen_age_s=None if self.seen_at is None else round(now - self.seen_at, 2),
            ),
            rover_state=self.rover_state,
            rover_state_age_s=(
                None if self.rover_state_at is None else round(now - self.rover_state_at, 2)
            ),
            rover_rtt_ms=(
                {'usual': round(rtt[len(rtt) // 2] * 1000), 'worst': round(rtt[-1] * 1000)}
                if rtt
                else None
            ),
            trail_m=round(sum(math.dist(a, b) for a, b in zip(trail, trail[1:], strict=False)), 2),
        )
        snap['map'] = self.mapper.stats()
        snap['plan'] = {
            'radius_m': self.planner.radius,
            'routes': self.plans,
            'note': self.plan_note,
        }
        snap['mapping_only'] = self.mapping_only
        if self.mapping_only:
            snap['status'] = 'mapping only (no rover): move the phone above the scene'
        elif self.rover_error:
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
                    with open(INDEX_HTML, 'rb') as f:
                        self._send(200, f.read(), 'text/html; charset=utf-8')
                elif path == '/top.jpg':  # ?view=grid: the planner's grid instead of photos
                    grid = 'view=grid' in self.path
                    jpg = service.grid_jpg if grid else service.top_jpg
                    if jpg is None:
                        self._json({'error': service.phone}, 503)
                    else:
                        self._send(200, jpg, 'image/jpeg')
                elif path == '/view.jpg':
                    if service.view_jpg is None:
                        self._json({'error': service.phone}, 503)
                    else:
                        self._send(200, service.view_jpg, 'image/jpeg')
                elif path == '/map.json':
                    self._json(service.mapper.grid())
                elif path == '/state':
                    self._json(service.state())
                elif path.startswith('/rover/') and path[6:] in ROVER_GET:
                    self._rover(path[6:], None)
                else:
                    self._json({'error': 'unknown path'}, 404)

            def _rover(self, path: str, body: dict | None) -> None:
                """Pass one request on to the rover, and its answer or its refusal back."""
                if service.manual is None:
                    self._json({'error': 'this guide has no rover address'}, 503)
                    return
                try:
                    self._json(service.manual._call(path, body))
                except RoverError as e:
                    refused = ': HTTP 4' in str(e)  # the rover answered: its own refusal
                    self._json({'error': str(e)}, 400 if refused else 502)

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
                    elif path == '/map/reset':
                        service.mapper.reset()
                        self._json({'ok': True})
                    elif path == '/stop':
                        try:
                            service.stop()
                            self._json({'ok': True})
                        except RoverError as e:
                            self._json({'error': f'the rover did not get the STOP: {e}'}, 502)
                    elif path.startswith('/rover/') and path[6:] in ROVER_POST:
                        self._rover(path[6:], body)
                    else:
                        self._json({'error': 'unknown path'}, 404)
                except (KeyError, TypeError, ValueError) as e:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 400)

        return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phone', required=True, help='CamStream, e.g. http://172.20.10.6:8080')
    ap.add_argument(
        '--rover',
        default=None,
        help="the rover's server, e.g. http://192.168.137.20:8080; without it: mapping only",
    )
    ap.add_argument('--port', type=int, default=8097)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--marker-id', type=int, default=10, help='AprilTag 25h9 id on the rover (10)')
    ap.add_argument(
        '--marker-size', type=float, default=0.14, help="the marker's black square, m (0.14)"
    )
    ap.add_argument(
        '--marker-height', type=float, default=0.26, help='marker above the floor, m (0.26)'
    )
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
        '--radius',
        type=float,
        default=0.15,
        help='m: the circle around the whole rover; obstacles are grown by it for the route '
        '(0.15, the value tested in the small arena)',
    )
    ap.add_argument(
        '--bounds',
        type=float,
        nargs=4,
        default=(-3.0, -3.0, 3.0, 3.0),
        metavar=('X0', 'Y0', 'X1', 'Y1'),
        help='the map, floor frame, m (6 m wide); centred where the phone first looks unless '
        '--fixed-bounds: then only its size counts',
    )
    ap.add_argument(
        '--no-colour',
        action='store_true',
        help="obstacles from the heights only, without the team's colour detection (for a "
        'floor with flat colours on it: a rug, tape, paint)',
    )
    ap.add_argument(
        '--fixed-bounds',
        action='store_true',
        help='keep the map on --bounds instead of centring it where the phone first looks',
    )
    ap.add_argument('--res', type=float, default=0.01, help='m per cell of the map (0.01)')
    a = ap.parse_args()
    finder = RoverFinder(a.marker_id, a.marker_size, a.marker_height, a.marker_forward)
    cfg = GuideConfig(speed=a.speed, marker_offset=tuple(a.marker_offset))
    rover = RoverClient(a.rover) if a.rover else None
    service = GuideService(
        a.phone, rover, finder, cfg, tuple(a.bounds), a.res, a.port, a.host,
        follow=not a.fixed_bounds, radius=a.radius, colour=not a.no_colour,
    )  # fmt: skip
    service.start()
    print(
        f'air guide: http://{a.host}:{service.port}  phone: {a.phone}  '
        f'rover: {a.rover or "none (mapping only)"}',
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
