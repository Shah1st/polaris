"""Drive the rover to a point chosen in the air camera's picture.

The air camera sees the rover in the floor frame (pose_camera.py, rover_marker.py); the rover
only knows its own odometry frame, which drifts and starts anywhere. This module ties the two
and steers:

- `OdomLink`: the transform floor <- odometry, corrected a little at every sighting of the
  rover. Between two sightings the rover's odometry carries the estimate.
- `Guide`: a goal in the floor frame becomes a path in the odometry frame, sent to the rover's
  own path follower (POST /path, 20 Hz on the robot). The path is sent again only when the
  correction has moved its end by more than a few centimetres, so the network stays out of the
  fast loop.

No ROS, no network here: `Guide` talks to a rover object with state(), path() and stop()
(`RoverClient` for the real server, a fake one in the tests).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

MOVING = ('move', 'turn', 'path', 'settle')


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    heading: float  # radians


class OdomHistory:
    """The rover's recent odometry poses, to know where it was when a picture was taken."""

    def __init__(self, seconds: float = 3.0) -> None:
        self.seconds = seconds
        self._items: deque[tuple[float, Pose2]] = deque()

    def add(self, t: float, pose: Pose2) -> None:
        if self._items and t <= self._items[-1][0]:
            return
        self._items.append((t, pose))
        while self._items and t - self._items[0][0] > self.seconds:
            self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    def at(self, t: float, tolerance: float = 0.5) -> Pose2 | None:
        """The pose at time t, between the two nearest readings; None when t is more than
        `tolerance` seconds outside what is known."""
        items = self._items
        if not items:
            return None
        if t <= items[0][0]:
            return items[0][1] if items[0][0] - t <= tolerance else None
        if t >= items[-1][0]:
            return items[-1][1] if t - items[-1][0] <= tolerance else None
        for (t0, a), (t1, b) in zip(items, list(items)[1:], strict=False):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0)
                return Pose2(
                    a.x + k * (b.x - a.x),
                    a.y + k * (b.y - a.y),
                    a.heading + k * wrap(b.heading - a.heading),
                )
        return None


class ClockSync:
    """Turns the phone's picture times into this computer's clock: the offset is the smallest
    (arrival - picture time) seen lately, that is the fastest trip, minus `min_latency`."""

    def __init__(self, min_latency: float = 0.05, window: int = 300) -> None:
        self.min_latency = min_latency
        self._gaps: deque[float] = deque(maxlen=window)

    def local_time(self, remote_t: float, arrival: float) -> float:
        self._gaps.append(arrival - remote_t)
        return remote_t + min(self._gaps) - self.min_latency

    def latency(self, remote_t: float, arrival: float) -> float:
        """How old the picture was when it arrived (s), by the same estimate."""
        return arrival - (remote_t + min(self._gaps or [arrival - remote_t]) - self.min_latency)


class OdomLink:
    """floor <- odometry: p_floor = R(theta) p_odom + (tx, ty), heading_floor = heading + theta.

    Each sighting of the rover from the air gives the whole transform (the marker gives the
    heading too). A sighting moves the estimate by `alpha` of the way, around the rover's
    current place, so that the rover's estimated position does not jump. A sighting far from
    the estimate is ignored, unless `reset_after` of them agree in a row: the odometry was
    reset (server restart, POST /reset_odom) or the floor frame started again.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        jump_m: float = 0.25,
        jump_deg: float = 25.0,
        reset_after: int = 5,
    ) -> None:
        self.alpha = alpha
        self.jump_m = jump_m
        self.jump = math.radians(jump_deg)
        self.reset_after = reset_after
        self.theta = 0.0
        self.tx = self.ty = 0.0
        self.ready = False
        self.updates = 0
        self.resets = 0
        self.last_error_m = 0.0  # how far the last sighting was from the estimate
        self.last_error_deg = 0.0
        self._outliers = 0

    def reset(self) -> None:
        self.ready = False
        self._outliers = 0

    def to_floor(self, odom: Pose2) -> Pose2:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2(
            c * odom.x - s * odom.y + self.tx,
            s * odom.x + c * odom.y + self.ty,
            wrap(odom.heading + self.theta),
        )

    def to_odom(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dy = x - self.tx, y - self.ty
        return c * dx + s * dy, -s * dx + c * dy

    def _set(self, theta: float, at_odom: Pose2, at_floor_x: float, at_floor_y: float) -> None:
        """The transform with this rotation that puts `at_odom` on that floor point."""
        self.theta = wrap(theta)
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.tx = at_floor_x - (c * at_odom.x - s * at_odom.y)
        self.ty = at_floor_y - (s * at_odom.x + c * at_odom.y)

    def update(self, seen: Pose2, odom: Pose2) -> bool:
        """One sighting: the rover seen at `seen` (floor frame) while its odometry said `odom`.
        True when it was used."""
        theta_seen = wrap(seen.heading - odom.heading)
        if not self.ready:
            self._set(theta_seen, odom, seen.x, seen.y)
            self.ready, self._outliers = True, 0
            self.updates += 1
            return True
        est = self.to_floor(odom)
        d_theta = wrap(theta_seen - self.theta)
        dx, dy = seen.x - est.x, seen.y - est.y
        self.last_error_m = math.hypot(dx, dy)
        self.last_error_deg = math.degrees(d_theta)
        if self.last_error_m > self.jump_m or abs(d_theta) > self.jump:
            self._outliers += 1
            if self._outliers >= self.reset_after:
                self._set(theta_seen, odom, seen.x, seen.y)
                self._outliers = 0
                self.resets += 1
                return True
            return False
        self._outliers = 0
        a = self.alpha
        self._set(self.theta + a * d_theta, odom, est.x + a * dx, est.y + a * dy)
        self.updates += 1
        return True


class RoverError(RuntimeError):
    """The rover's server did not answer, or refused the command."""


class RoverClient:
    """The rover's HTTP server (rover/rover_server.py): GET /state, POST /path, POST /stop."""

    def __init__(self, base_url: str, timeout: float = 2.0, name: str = 'air-guide') -> None:
        self.base = base_url.rstrip('/')
        self.timeout = timeout
        self.name = name

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data, method='GET' if data is None else 'POST'
        )
        req.add_header('X-Rover-Client', self.name)
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b'{}')
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', 'replace')[:200]
            raise RoverError(f'{path}: HTTP {e.code} {detail}') from e
        except (OSError, ValueError) as e:
            raise RoverError(f'{path}: {type(e).__name__}: {e}') from e

    def state(self) -> dict:
        return self._call('/state')

    def path(
        self, points: list[tuple[float, float]], speed: float, path_id: str | None = None
    ) -> None:
        """With an id, a path replacing a running one takes over at speed, without a stop
        (the team's rover, rover_core.py): the guide's corrections never stop the rover."""
        body = {'points': [list(p) for p in points], 'frame': 'odom', 'speed': speed}
        if path_id:
            body['id'] = path_id
        self._call('/path', body)

    def stop(self) -> None:
        self._call('/stop', {})


@dataclass
class GuideConfig:
    speed: float = 0.15  # m/s
    tolerance: float = 0.04  # m: arrived within this distance of the goal
    # While driving, the path is sent again for a correction of the rover's place only when
    # it moved the path's end by more than `resend` m, and not twice within `min_resend_s`:
    # the rover drives on its odometry, as in the team's segments (sim/fusion_reelle.py
    # --segment); a new route (an obstacle on the way) is sent at once.
    resend: float = 0.08
    min_resend_s: float = 2.0
    # Rover not seen from the air (picture lost, rover out of view): after `blind_s` s it goes
    # on along the route on its odometry; it stops after `blind_max_m` m driven or
    # `blind_max_s` s that way, and goes on when it is seen again. The planner's margin grows
    # by `blind_margin` of the distance driven blind (the odometry's uncertainty), as the
    # team's `marge_drone` does without a drone (merging_map/fusion_carte.py).
    blind_s: float = 1.0
    blind_max_m: float = 1.5
    blind_max_s: float = 30.0
    blind_margin: float = 0.05
    near: float = 0.10  # m: this close to the goal, at most `near_tries` more paths
    near_tries: int = 3
    marker_offset: tuple[float, float] = (0.0, 0.0)  # marker centre in the rover's own frame
    max_scale_error: float = 0.2  # a sighting whose marker is this much off its size is dropped
    min_side_px: float = 14.0  # a marker smaller than this in the picture is dropped


@dataclass
class Guide:
    """Call `on_state` with each GET /state of the rover, `on_sighting` with each picture's
    result, then `step`: it sends what is needed to the rover and says where things stand."""

    rover: object  # state() is not used here: path(points, speed), stop()
    cfg: GuideConfig = field(default_factory=GuideConfig)
    link: OdomLink = field(default_factory=OdomLink)
    history: OdomHistory = field(default_factory=OdomHistory)

    goal: tuple[float, float] | None = None
    waypoints: list[tuple[float, float]] = field(default_factory=list)  # before the goal
    status: str = 'no goal'
    odom: Pose2 | None = None
    mode: str = 'idle'
    last_seen: float | None = None
    sent: tuple[float, float] | None = None  # end of the path the rover is following (odom)
    paths_sent: int = 0
    goals: int = 0  # goals set so far: part of each path's id
    # the rover's guard stopped a path of this guide before an obstacle (its place on the
    # floor), not yet handled by whoever plans the route
    blocked_at: list = field(default_factory=list)
    blocked_in_a_row: int = 0
    _history_seen: object = None
    dropped: str | None = None  # why the last sighting was not used
    _boot_id: object = None
    _stop_count: int | None = None
    _near_tries: int = 0
    _blind_stopped: bool = False
    _state_since_command: bool = True  # False until the rover's state is read after a command
    # False while a planner owes a route for the goal: nothing is sent before (not even the
    # straight line). Without a planner it stays True.
    route_ready: bool = True
    blind_m: float = 0.0  # m driven on odometry since the rover was last seen
    _sent_at: float = -1e9
    _prev_odom: Pose2 | None = None

    # -- inputs ---------------------------------------------------------------------------

    def set_goal(self, x: float, y: float, waypoints=()) -> None:
        """Go to (x, y) in the floor frame, through `waypoints` first (a planner's path)."""
        self.goals += 1
        self.blocked_in_a_row = 0
        self.goal = (float(x), float(y))
        self.waypoints = [(float(a), float(b)) for a, b in waypoints]
        self.sent = None
        self._near_tries = 0
        self._blind_stopped = False
        self._stop_count = None
        self.status = 'goal set'

    def set_route(self, waypoints) -> None:
        """A new way to the same goal (the planner went around something new): the rover
        gets it at the next step, without starting the goal over."""
        self.waypoints = [(float(a), float(b)) for a, b in waypoints]
        self.sent = None  # sent again at the next step
        self.route_ready = True

    def cancel(self, why: str = 'cancelled') -> None:
        """Forget the goal and stop the rover if this guide was driving it."""
        driving = self.goal is not None and self.sent is not None
        self.goal, self.sent, self.waypoints = None, None, []
        self.status = why
        if driving:
            self._stop()

    def on_state(self, state: dict, t: float) -> None:
        """One GET /state of the rover, received at time t (this computer's clock)."""
        o = state['odom']
        self.odom = Pose2(float(o['x']), float(o['y']), math.radians(float(o['heading_deg'])))
        ctrl = state.get('ctrl', {})
        self.mode = str(ctrl.get('mode', 'idle'))
        if self.mode == 'idle' and ctrl.get('queued'):
            self.mode = 'path'  # received, not started yet: the rover's loop takes it next
        self._state_since_command = True
        if self._prev_odom is not None and (self.last_seen is None or t - self.last_seen > 0.5):
            self.blind_m += math.dist(
                (self._prev_odom.x, self._prev_odom.y), (self.odom.x, self.odom.y)
            )
        self._prev_odom = self.odom
        boot = state.get('boot_id')
        if boot != self._boot_id:  # the server restarted: its odometry starts again from zero
            if self._boot_id is not None:
                self.link.reset()
                self.history.clear()
                self.sent = None
            self._boot_id = boot
            self._stop_count = None
        self.history.add(t, self.odom)
        self._read_guard(ctrl.get('history') or [])
        stops = state.get('stop_count')
        if self.goal is not None:
            # Someone drives by hand, or pressed STOP while this guide was driving.
            someone_else = self.mode == 'manual' or (
                self.sent is not None
                and stops is not None
                and self._stop_count is not None
                and stops > self._stop_count
            )
            if someone_else:
                self.goal, self.sent, self.waypoints = None, None, []
                self.status = 'stopped: someone else took the rover over'
        self._stop_count = stops

    def _read_guard(self, history: list) -> None:
        """The rover's own guard (its D405, local_guard = 1) ended one of this guide's paths
        as 'blocked' before an obstacle: its place (odometry) goes to `blocked_at`, on the
        floor, for the planner; three in a row without the rover moving end the goal."""
        if not history:
            return
        last = history[0]
        key = (last.get('id'), last.get('status'), last.get('duration'))
        if key == self._history_seen:
            return
        first = self._history_seen is None
        self._history_seen = key
        if first or not str(last.get('id') or '').startswith('air-'):
            return
        if last.get('status') != 'blocked':
            self.blocked_in_a_row = 0
            return
        ob = last.get('obstacle')
        if ob is not None and self.link.ready:
            p = self.link.to_floor(Pose2(float(ob[0]), float(ob[1]), 0.0))
            self.blocked_at.append((p.x, p.y))
        self.blocked_in_a_row += 1
        self.sent = None  # sent again, around it, once the route is planned again
        if self.goal is not None and self.blocked_in_a_row >= 3:
            self.goal, self.waypoints = None, []
            self.status = "stopped: the rover's guard blocked it three times in a row"

    def on_sighting(self, seen, t_picture: float) -> bool:
        """The rover as one picture saw it (a rover_marker.RoverSeen), taken at `t_picture` on
        this computer's clock. True when it corrected the link."""
        self.dropped = None
        if seen is None:
            return False
        if seen.side_px < self.cfg.min_side_px:
            self.dropped = f'marker too small in the picture ({seen.side_px:.0f} px)'
            return False
        if abs(seen.scale_error) > self.cfg.max_scale_error:
            self.dropped = (
                f'marker {seen.scale_error:+.0%} off its size: pose or floor height wrong'
            )
            return False
        odom = self.history.at(t_picture)
        if odom is None:
            self.dropped = 'no odometry at the time of the picture'
            return False
        ox, oy = self.cfg.marker_offset
        c, s = math.cos(seen.heading), math.sin(seen.heading)
        base = Pose2(seen.x - (c * ox - s * oy), seen.y - (s * ox + c * oy), seen.heading)
        used = self.link.update(base, odom)
        if used:
            self.last_seen = t_picture
            self.blind_m = 0.0
        else:
            self.dropped = (
                f'sighting {self.link.last_error_m * 100:.0f} cm / '
                f'{self.link.last_error_deg:+.0f} deg from the estimate: ignored'
            )
        return used

    # -- the estimate ---------------------------------------------------------------------

    def rover_on_floor(self) -> Pose2 | None:
        """Where the rover is now in the floor frame: its odometry through the link."""
        if self.odom is None or not self.link.ready:
            return None
        return self.link.to_floor(self.odom)

    # -- steering -------------------------------------------------------------------------

    def _stop(self) -> None:
        self._state_since_command = False
        self.rover.stop()
        if self._stop_count is not None:
            self._stop_count += 1  # our own stop, not an operator's

    def step(self, now: float) -> str:
        """Send what is needed to the rover; returns the status line."""
        if self.goal is None:
            return self.status
        est = self.rover_on_floor()
        if est is None:
            self.status = 'waiting: the rover has not been seen from the air yet'
            return self.status
        if not self._state_since_command:
            return self.status  # what the rover does with the last command is not known yet
        if not self.route_ready:
            self.status = 'planning the route'
            return self.status
        blind = None if self.last_seen is None else now - self.last_seen
        if blind is None:
            self.status = 'waiting: the rover has not been seen from the air yet'
            return self.status
        blind_note = ''
        if blind > self.cfg.blind_s:
            if self.blind_m > self.cfg.blind_max_m or blind > self.cfg.blind_max_s:
                if self.mode in MOVING and self.sent is not None and not self._blind_stopped:
                    self._stop()
                    self._blind_stopped = True
                self.sent = None
                self.status = (
                    f'stopped: rover not seen from the air for {blind:.0f} s, '
                    f'{self.blind_m:.1f} m on odometry: waiting to see it again'
                )
                return self.status
            blind_note = f' on odometry (not seen for {blind:.0f} s, {self.blind_m:.2f} m)'
        self._blind_stopped = False

        # Waypoints already passed (closer than the tolerance) are dropped.
        while (
            self.waypoints
            and math.dist((est.x, est.y), self.waypoints[0]) < 2 * self.cfg.tolerance
        ):
            self.waypoints.pop(0)
        distance = math.dist((est.x, est.y), self.goal)
        moving = self.mode in MOVING
        if not moving and not self.waypoints and distance <= self.cfg.tolerance:
            self.goal, self.sent = None, None
            self.status = f'arrived, {distance * 100:.1f} cm from the goal'
            return self.status

        route = [*self.waypoints, self.goal]
        end = self.link.to_odom(*self.goal)
        if (
            moving
            and self.sent is not None
            and (
                math.dist(end, self.sent) <= self.cfg.resend
                or now - self._sent_at < self.cfg.min_resend_s
            )
        ):
            self.status = f'driving{blind_note}, {distance * 100:.0f} cm to go'
            return self.status
        if not moving and self.sent is not None and distance < self.cfg.near:
            self._near_tries += 1
            if self._near_tries > self.cfg.near_tries:
                self.goal, self.sent = None, None
                self.status = f'stopped {distance * 100:.1f} cm from the goal'
                return self.status
        self._state_since_command = False
        self.rover.path([self.link.to_odom(x, y) for x, y in route], self.cfg.speed,
                        f'air-{self.goals}-{self.paths_sent + 1}')  # fmt: skip
        self.sent = end
        self._sent_at = now
        self.paths_sent += 1
        self.status = f'path sent ({self.paths_sent}){blind_note}, {distance * 100:.0f} cm to go'
        return self.status

    def blind_uncertainty(self) -> float:
        """How far (m) the rover may be from where its odometry puts it, since it was last
        seen: the planner keeps that much more room around the obstacles."""
        return self.cfg.blind_margin * self.blind_m

    def snapshot(self) -> dict:
        est = self.rover_on_floor()
        return {
            'status': self.status,
            'goal': self.goal,
            'waypoints': self.waypoints,
            'rover': None if est is None else [est.x, est.y, math.degrees(est.heading)],
            'mode': self.mode,
            'link': {
                'ready': self.link.ready,
                'theta_deg': math.degrees(self.link.theta),
                'offset': [self.link.tx, self.link.ty],
                'updates': self.link.updates,
                'resets': self.link.resets,
                'last_error_cm': self.link.last_error_m * 100,
                'last_error_deg': self.link.last_error_deg,
            },
            'dropped': self.dropped,
            'paths_sent': self.paths_sent,
        }
