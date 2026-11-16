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
                      ?view=grid, the grid the planner uses instead of the photos; with
                      ?view=heights, the heights (LiDAR and plane sweep) in colour
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

The phone must be CamStream in ARKit mode (iphone-camera/README.md); with a LiDAR iPhone its
depth maps measure the heights at once. The route goes around the obstacles (planner.py, the
team's A*); the rover's D405 (the team's GET /local_grid) adds, confirms and clears what is
right ahead of it, and its guard stopping a path puts that obstacle on the map.
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
from beam import BeamReader
from camstream import DepthPoller, PoseStream
from floor_map import LINE_BGR, OBSTACLE_BGR, FloorMap
from guidance import ClockSync, Guide, GuideConfig, RoverClient, RoverError
from planner import RoutePlanner
from pose_camera import PoseError
from rover_marker import RoverFinder
from run_log import RunLog

INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'index.html')
# What the dashboard may ask the rover through this server.
ROVER_GET = ('/commands', '/config', '/d405/status')
ROVER_POST = ('/cmd', '/gimbal')
# the dashboard's guard switch: POST /guard {"on": true|false} sets this one rover setting only
GUARD_SETTING = 'local_guard'


def to_pixel(x: float, y: float, bounds, res: float) -> tuple[int, int]:
    """A floor point (metres) in the top view of `bounds` at `res` m per pixel."""
    return round((x - bounds[0]) / res), round((bounds[3] - y) / res)


def draw(top: np.ndarray, bounds, res: float, snap: dict, trail, seen) -> np.ndarray:
    """The top view with what the guide knows: the rover's trail, the rover as the link says
    (green arrow), as the last picture saw it (yellow ring), the path and the goal (red)."""
    out = top.copy()
    if len(trail) > 1:
        pts = np.array([to_pixel(x, y, bounds, res) for x, y in trail], np.int32)
        cv2.polylines(out, [pts], False, (200, 170, 60), 1, cv2.LINE_AA)
    rover = snap.get('rover')
    goal = snap.get('goal')
    if goal is not None:
        chain = ([rover[:2]] if rover else []) + [*snap.get('waypoints', []), goal]
        pts = np.array([to_pixel(x, y, bounds, res) for x, y in chain], np.int32)
        cv2.polylines(out, [pts], False, LINE_BGR, 2, cv2.LINE_AA)
        cv2.drawMarker(out, to_pixel(*goal, bounds, res), LINE_BGR, cv2.MARKER_TILTED_CROSS, 14, 2)
    if seen is not None:
        cv2.circle(out, to_pixel(seen.x, seen.y, bounds, res), 9, (71, 181, 255), 1, cv2.LINE_AA)
    if rover is not None:
        x, y, heading = rover[0], rover[1], math.radians(rover[2])
        tip = (x + 0.15 * math.cos(heading), y + 0.15 * math.sin(heading))
        cv2.arrowedLine(
            out,
            to_pixel(x, y, bounds, res),
            to_pixel(*tip, bounds, res),
            (165, 255, 75),
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )
    return out


def shade_blocked(picture: np.ndarray, blocked: np.ndarray, k: int) -> np.ndarray:
    """The map picture with the cells the rover's centre may not enter (the obstacles grown
    by its radius; `blocked` row 0 = south, `k` picture pixels per cell) shaded coral: the
    route stays out of them."""
    rows, cols = picture.shape[:2]
    mask = np.kron(blocked[::-1].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    full = np.zeros((rows, cols), bool)
    h, w = min(rows, mask.shape[0]), min(cols, mask.shape[1])
    full[:h, :w] = mask[:h, :w]
    out = picture.copy()
    out[full] = (0.82 * out[full] + 0.18 * np.array(OBSTACLE_BGR)).astype(np.uint8)
    return out


def on_the_way(points, route, radius: float, ahead: float) -> bool:
    """Is any of `points` (N x 2) within `radius` of the first `ahead` metres of `route`
    (a polyline from the rover's place)?"""
    pts = np.asarray(points, float).reshape(-1, 2)
    left = ahead
    for a, b in zip(route, route[1:], strict=False):
        length = math.dist(a, b)
        if length < 1e-6:
            continue
        span = min(length, left)
        d = np.subtract(b, a) / length
        t = np.clip((pts - a) @ d, 0.0, span)
        if (np.hypot(*(pts - (np.asarray(a) + t[:, None] * d)).T) <= radius).any():
            return True
        left -= span
        if left <= 0:
            break
    return False


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
        self.heights_picture: np.ndarray | None = None  # the heights, in colour
        self.error: str | None = None
        self.last_s: float | None = None  # seconds the last update took
        self.depth_said: dict = {}  # what the last LiDAR depth map did
        self._job = None
        self._depth_job = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def offer(self, image, pose, rover_xy) -> None:
        self._job = (image, pose, rover_xy)
        self._wake.set()

    def offer_depth(self, depth, rover_xy) -> None:
        """A LiDAR depth map (camstream.DepthMap, with its pose): the newest one only."""
        self._depth_job = (depth, rover_xy)
        self._wake.set()

    def reset(self) -> None:
        with self.lock:
            self.fmap.reset()
            self.picture = None
            self.grid_picture = None
            self.heights_picture = None

    def close(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            job, self._job = self._job, None
            depth, self._depth_job = self._depth_job, None
            if job is None and depth is None:
                continue
            t0 = time.monotonic()
            try:
                with self.lock:
                    if depth is not None:  # heights measured at once, before the picture's
                        d, rover_xy = depth
                        self.depth_said = self.fmap.add_depth(d.metres, d.k, d.pose, rover_xy)
                    if job is not None:
                        self.fmap.add(*job)  # with the team's colour detection on this view
                    self.picture = self.fmap.render()
                    self.grid_picture = self.fmap.render_grid()
                    self.heights_picture = self.fmap.render_heights()
                self.error = None
            except Exception as e:  # noqa: BLE001 - the map must never stop the guide
                self.error = f'{type(e).__name__}: {e}'
            self.last_s = time.monotonic() - t0

    def grid(self, cell: float | None = None) -> dict:
        """The grid the route is planned on (the map's own decision cells), or `cell` m."""
        with self.lock:
            return self.fmap.grid(cell or self.fmap.cell)

    def set_rover(self, x: float, y: float) -> None:
        with self.lock:
            self.fmap.set_rover(x, y)

    def add_beam(self, obstacle_xy, floor_xy) -> dict:
        with self.lock:
            return self.fmap.add_beam(obstacle_xy, floor_xy)

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
        log_dir=None,
        beam=True,
        beam_range=1.5,
        lidar=True,
    ):
        """`follow`: the map centres itself where the phone first looks (`bounds` then only
        gives its size); otherwise it covers `bounds` of the floor frame. `radius`: the
        circle around the whole rover, by which obstacles are grown for the route.
        `colour`: the team's colour detection adds its obstacles to the heights' (it cannot
        tell a flat colour on the floor from an object: off for a painted floor). `lidar`:
        ask the phone for its LiDAR depth (/depth; a phone without LiDAR says so, asked
        again every few seconds only)."""
        self.planner = RoutePlanner(radius=radius)
        # the rover's D405 (beam.py): a second stream, right ahead of the rover
        rover_url = getattr(rover, 'base', None)
        self.beam = BeamReader(rover_url) if beam and rover_url else None
        self.beam_range = beam_range
        self.beam_info: dict = {}
        self.events: deque[dict] = deque(maxlen=40)  # for the dashboard's event list
        self._goal_seen = None
        self.beam_events = 0
        self._beam_seq = 0
        self._beam_said = 0.0  # no new D405 event before then
        # the phone's LiDAR (ARKit sceneDepth with the pose of its frame): heights at once
        self.depth = DepthPoller(phone_url) if lidar else None
        self.depth_info: dict = {}
        self._depth_seq = 0
        # a record of each drive (run_log.py): None = none
        self.log = RunLog(log_dir) if log_dir else None
        self.clock = time.monotonic  # the time of the record's lines
        self._next_log = 0.0
        self._logged = (None, 0, None)  # goal, paths sent, status: what was last written
        self.plans = 0  # routes computed for the current goal and before
        self.plan_note: str | None = None
        self._next_plan = 0.0
        self.stream = PoseStream(phone_url)
        rover = rover if rover is not None else NoRover()
        self.rover = rover
        self.mapping_only = isinstance(rover, NoRover)
        self.mapper = MapWorker(FloorMap(bounds, res, follow=follow, colour=colour))
        self.view_jpg: bytes | None = None
        self.finder = finder
        self.guide = Guide(rover, cfg or GuideConfig())
        self.res = res
        self.sync = ClockSync()
        self.lock = threading.Lock()
        self.top_jpg: bytes | None = None
        self.grid_jpg: bytes | None = None  # the same, on the planner's grid
        self.heights_jpg: bytes | None = None  # the same, the heights in colour
        self.bare: dict[str, bytes] = {}  # the same three without the guide's drawings
        self.beam_overlay: dict | None = None  # the D405's last look on the floor, for the page
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
        if self.beam is not None:
            self.beam.close()
        if self.depth is not None:
            self.depth.close()
        self.server.shutdown()
        self.server.server_close()

    def _loop(self) -> None:
        seq, next_state, next_draw = -1, 0.0, 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_state:
                next_state = now + 0.1
                self._read_rover()
            if self.beam is not None:
                self._use_beam()
            if self.depth is not None:
                self._use_depth()
            if self.guide.blocked_at:
                self._use_blocked()
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
            if self.log is not None:
                self._write_log(now)
            if self._goal_seen is not None and self.guide.goal is None:
                self.event('end', self.guide.status)  # arrived, stopped, cancelled
            self._goal_seen = self.guide.goal
            if fresh and now >= next_draw:
                next_draw = now + 0.15
                self._draw(frame)
            self._stop.wait(0.01 if fresh else 0.03)

    def _use_blocked(self) -> None:
        """The rover's guard stopped it before an obstacle: that place becomes an obstacle on
        the map (as if the D405 stream had seen it) and the route is planned again at once."""
        with self.lock:
            places, self.guide.blocked_at = list(self.guide.blocked_at), []
        for x, y in places:
            a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            disc = np.column_stack([x + 0.05 * np.cos(a), y + 0.05 * np.sin(a)])
            self.mapper.add_beam(np.vstack([[x, y], disc]), np.zeros((0, 2)))
            self.event('guard', f'rover guard: stopped before an obstacle at {x:.2f}, {y:.2f}')
        self._next_plan = 0.0

    def event(self, kind: str, text: str) -> None:
        """Something worth showing on the dashboard's event list (and in the drive's record)."""
        self.events.appendleft({'t': round(time.time(), 1), 'kind': kind, 'text': text})
        if self.log is not None:
            self.log.write('event', at=self.clock(), what=kind, text=text)

    def _use_beam(self) -> None:
        """A new look of the rover's D405 (its local grid, odometry frame): placed on the floor
        through the air-ground link, then believed over the air map (FloorMap.add_beam).
        Something on the way, close ahead: the route is planned again at once, not at the next
        half second."""
        grid = self.beam.latest(max_age=2.0)
        if grid is None:
            self.beam_info = {'error': self.beam.error or 'no fresh D405 picture',
                              'looks': self.beam.count}  # fmt: skip
            return
        if grid.frames == self._beam_seq:
            return
        self._beam_seq = grid.frames
        with self.lock:
            link = self.guide.link
            ready = link.ready
            here = self.guide.rover_on_floor()
            goal, waypoints = self.guide.goal, list(self.guide.waypoints)
        if not ready or here is None:
            self.beam_info = {
                'error': 'the rover is not placed on the map yet',
                'looks': grid.frames,
            }
            return
        obstacle, floor = grid.on_floor(link, self.beam_range)
        said = self.mapper.add_beam(obstacle, floor)
        ob1, fl1 = grid.on_floor(link, self.beam_range, sub=1)  # one point per cell: the page
        self.beam_overlay = {
            'obstacle': np.round(ob1, 3).tolist(),
            'floor': np.round(fl1, 3).tolist(),
            'cell': grid.res,
            'at': time.monotonic(),
        }
        if goal is not None and len(obstacle):
            route = [(here.x, here.y), *waypoints, goal]
            # inside the band the rover sweeps (a route planned around it passes at about
            # its radius: not on the way any more), at most once a second and a half
            now = time.monotonic()
            if now >= self._beam_said and on_the_way(
                obstacle, route, self.planner.radius - 0.02, 0.8
            ):
                self._next_plan = 0.0  # plan again now
                self._beam_said = now + 1.5
                self.beam_events += 1
                self.event('d405', 'D405: an obstacle on the way, planning again')
        self.beam_info = {
            'looks': grid.frames,
            'age_ms': None if grid.age_s is None else round(grid.age_s * 1000),
            'obstacle_cells': said['obstacle_cells'],
            'floor_cells': said['floor_cells'],
            'events': self.beam_events,
            'request_ms': None if self.beam.last_s is None else round(self.beam.last_s * 1000),
        }

    def _use_depth(self) -> None:
        """A new LiDAR depth map of the phone, with the pose of its frame: to the map worker,
        which places every point it measured on the floor (FloorMap.add_depth)."""
        d = self.depth.latest(max_age=1.5)
        maps = self.mapper.fmap.depth_maps
        if d is None:
            self.depth_info = {'error': self.depth.error or 'no fresh depth map', 'maps': maps}
            return
        if self.depth.count == self._depth_seq:
            return
        self._depth_seq = self.depth.count
        if d.pose is None:
            self.depth_info = {'error': d.pose_error or 'depth without a pose (ARKit mode off)',
                               'maps': maps}  # fmt: skip
            return
        with self.lock:
            est = self.guide.rover_on_floor()
        self.mapper.offer_depth(d, None if est is None else (est.x, est.y))
        said = self.mapper.depth_said
        self.depth_info = {
            'maps': maps,
            'size': [int(d.metres.shape[1]), int(d.metres.shape[0])],
            'age_ms': d.age_ms,
            'points': said.get('points'),
            'error': None if said.get('used', True) else said.get('why'),
            'floor_offset_cm': round(self.mapper.fmap.floor_offset * 100, 1),
        }

    def _write_log(self, now: float) -> None:
        """The drive's record: a line per path sent and per end of a goal, four a second
        while a goal is set."""
        g = self.guide
        goal, sent, status = self._logged
        if g.paths_sent != sent:
            self.log.write('send', at=now, n=g.paths_sent, blind_m=round(g.blind_m, 3))
        if goal is not None and g.goal is None:  # the goal ended: arrived, stopped, cancelled
            self.log.write('end', at=now, status=g.status)
        if g.goal is not None and now >= self._next_log:
            self._next_log = now + 0.25
            est = g.rover_on_floor()
            seen = None if g.last_seen is None else round(now - g.last_seen, 2)
            self.log.write(
                'tick',
                at=now,
                rover=None if est is None else [round(est.x, 3), round(est.y, 3),
                                                 round(math.degrees(est.heading), 1)],
                odom=None if g.odom is None else [round(g.odom.x, 3), round(g.odom.y, 3)],
                mode=g.mode,
                status=g.status,
                seen_age=seen,
                blind_m=round(g.blind_m, 3),
                link_error_cm=round(g.link.last_error_m * 100, 1),
            )  # fmt: skip
        self._logged = (g.goal, g.paths_sent, g.status)

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
        # on odometry since the rover was last seen: more room, its place is less sure
        self.planner.set_margin(self.guide.blind_uncertainty())
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
        length = sum(math.dist(a, b) for a, b in zip([start, *new], new, strict=False))
        self.event('route', f'route planned: {length:.2f} m, {len(new) - 1} turn(s)')
        if self.log is not None:
            points = [[round(x, 3), round(y, 3)] for x, y in new]
            margin = self.planner.carte.marge_securite
            self.log.write('route', at=self.clock(), points=points, margin_m=margin)

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
        if here is not None:
            self.mapper.set_rover(here.x, here.y)
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
        # where the rover is, left out of the map: seen in this picture, or else where the
        # guide puts it (a blurred marker must not turn the rover into an obstacle)
        if seen is not None:
            rover_xy = (seen.x, seen.y)
        else:
            with self.lock:
                est = self.guide.rover_on_floor()
            rover_xy = None if est is None else (est.x, est.y)
        self.mapper.offer(image, pose, rover_xy)
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
            ('heights_jpg', self.mapper.heights_picture),
        ):
            if picture is not None and grid is not None and grid['x0'] == self.bounds[0]:
                k = max(1, round(grid['res'] / self.res))
                picture = shade_blocked(picture, self.planner.blocked_cells(grid), k)
            base = picture if picture is not None else view
            out = draw(base, self.bounds, self.res, snap, trail, self._seen)
            ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                setattr(self, name, jpg.tobytes())
            ok, jpg = cv2.imencode('.jpg', base, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.bare[name] = jpg.tobytes()

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
        if self.log is not None:
            here = self.guide.rover_on_floor()
            start = None if here is None else [round(here.x, 3), round(here.y, 3)]
            self.log.write('goal', at=self.clock(), goal=[round(x, 3), round(y, 3)], start=start)
        with self.lock:
            self.plan_note = None
        self.event('goal', f'goal set at {x:.2f}, {y:.2f}')
        self._replan(force=True)  # the route before anything is sent to the rover
        if self.guide.goal is None and self.plan_note:
            if self.log is not None:  # refused at once: the loop never saw this goal
                self.log.write('end', at=self.clock(), status=f'refused: {self.plan_note}')
                self._logged = (None, self.guide.paths_sent, self.guide.status)
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
        snap['beam'] = self.beam_info if self.beam is not None else None
        snap['events'] = list(self.events)[:20]
        snap['blind_m'] = round(self.guide.blind_m, 3)  # driven on odometry since last seen
        step = max(1, len(trail) // 300)
        snap['trail'] = [[round(x, 3), round(y, 3)] for x, y in trail[::step]]
        ov = self.beam_overlay
        snap['beam_points'] = (
            {k: v for k, v in ov.items() if k != 'at'}
            if ov is not None and now - ov['at'] < 2.0
            else None
        )
        snap['lidar'] = (
            dict(self.depth_info, maps=self.mapper.fmap.depth_maps)
            if self.depth is not None
            else None
        )
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
                elif path == '/top.jpg':  # ?view=grid: the planner's grid; ?view=heights
                    query = self.path.partition('?')[2]
                    args = dict(p.partition('=')[::2] for p in query.split('&') if p)
                    name = {'grid': 'grid_jpg', 'heights': 'heights_jpg'}.get(
                        args.get('view'), 'top_jpg'
                    )  # &bare=1: without the drawings (the dashboard draws them itself)
                    bare = args.get('bare') == '1'
                    jpg = service.bare.get(name) if bare else getattr(service, name)
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
                    self._json(service.mapper.grid(0.05))  # the team's 5 cm format
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
                    elif path == '/guard':  # the rover's own guard (its D405), on or off
                        on = 1 if body.get('on') else 0
                        self._rover('/config', {GUARD_SETTING: on})
                        service.event('guard', f'rover guard {"on" if on else "off"}')
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
        '--no-d405',
        action='store_true',
        help="without the rover's D405 (its GET /local_grid): the air camera's map only",
    )
    ap.add_argument(
        '--no-lidar',
        action='store_true',
        help="without the phone's LiDAR depth (/depth): heights from the moving picture only",
    )
    ap.add_argument(
        '--log',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        help="folder of the drives' records (run_log.py); '' for none",
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
        log_dir=a.log or None, beam=not a.no_d405, lidar=not a.no_lidar,
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
