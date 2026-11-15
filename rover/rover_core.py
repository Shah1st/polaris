"""Core of the RaspRover test driver: ESP32 link, odometry and motion controller.

No ROS and no web code here, so it can be tested on its own (see tests/). Runs on the
rover's onboard computer (RDK X5, Python 3.10) and on a laptop with the simulated ESP32.

Conventions (REP 103): x forward, y left, heading in radians, positive = turning left.

Protocol: Waveshare "UGV" ESP32 firmware, as measured on the robot (2026-11-14). T:1001
arrives on its own at ~20 Hz with L, R (wheel speeds, m/s, in steps of 0.06 m/s), raw IMU
values (gz: 16.4 per deg/s, az ~8192 per g), odl/odr (wheel counters, cm) and v (battery,
centivolts). Opening the USB port reboots the board: about 7 s, IMU calibration included.

Position estimate:
* distance from the wheel encoders, (L + R) / 2 integrated over time;
* heading from the gyro (gz), with its bias measured while the rover stands still.
  With four fixed wheels the rover skids when it turns, so the encoders overestimate
  rotations; their heading is still computed and shown, as a slip indicator.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable

BAUDRATE = 115200


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Tunable settings. Saved as JSON next to the server, editable from the web page."""

    # Arrow keys
    v_manual: float = 0.15  # m/s
    v_manual_fast: float = 0.30  # m/s, with Shift
    w_manual: float = 1.2  # rad/s
    w_manual_fast: float = 2.0  # rad/s, with Shift
    manual_timeout: float = 0.5  # s without a manual command before the rover stops

    # Measured moves ("forward 50 cm")
    v_max: float = 0.30  # m/s
    v_min: float = 0.06  # m/s, slowest speed that still moves the rover
    accel: float = 0.6  # m/s^2, ramp at start to limit wheel slip
    k_dist: float = 1.5  # speed = k_dist * remaining distance
    k_head: float = 3.0  # heading hold while moving straight
    dist_tol: float = 0.005  # m

    # Measured turns ("turn left 90 deg")
    w_min: float = 0.8  # rad/s, slowest rotation that still turns the rover
    w_max: float = 2.0  # rad/s
    k_turn: float = 2.5  # rotation speed = k_turn * remaining angle
    angle_tol_deg: float = 1.0
    corrections: int = 2  # extra small moves/turns if the rover stopped off the goal

    lag: float = 0.15  # s, stop this early to absorb the rover's inertia
    settle: float = 0.4  # s, pause after each move/turn before measuring the result
    task_timeout: float = 20.0  # s, give up a move/turn after this long
    stall_time: float = 1.5  # s, give up if the wheels do not move while commanded

    # Calibration
    dist_scale: float = 1.0  # real distance / distance measured by the encoders
    yaw_sign: int = 1  # -1 if the heading decreases when the rover turns left
    # raw gz -> rad/s. UGV firmware: 16.4 raw per deg/s (+-2000 deg/s range), checked by
    # turning the rover by hand: 16.2 to 17.0 on 2026-11-14.
    gyro_scale: float = math.radians(1.0) / 16.4
    track: float = 0.125  # m, wheel track used by the ESP32 firmware for the RaspRover
    bias_calib_time: float = 2.0  # s of stillness to measure the gyro bias

    # Gyro steering assist for manual driving (a software differential): when the input is
    # "straight", hold the heading; when it is a turn, make the measured rotation follow it.
    # The correction is a rotation command, which the ESP32 turns into a left/right speed
    # difference. All of it runs on the onboard computer: the ESP32 receives the same 20
    # commands per second as without it.
    assist: int = 1  # 1 = on, 0 = off
    assist_hz: float = 20.0  # corrections per second (at most the 20 Hz control loop)
    assist_deadband_deg: float = 1.0  # heading errors smaller than this are ignored
    assist_k: float = 3.0  # heading hold: rad/s of correction per rad of heading error
    assist_ki: float = 1.0  # heading hold: integral gain (1/s), removes a steady drift
    assist_k_rate: float = 0.6  # turns: extra rad/s per rad/s of rotation-speed error
    assist_ki_rate: float = 1.5  # turns: integral gain (1/s), makes up for skidding
    assist_w_max: float = 0.8  # rad/s, largest correction
    assist_settle: float = 0.3  # s after a turn ends before the new heading is held
    assist_min_speed: float = 0.02  # m/s, no help while standing still

    # Path following ("follow this trajectory"): pure pursuit on the odometry pose
    path_speed: float = 0.2  # m/s, cruise speed (a path step can ask for another one)
    path_lookahead: float = 0.2  # m, smallest look-ahead distance (shorter: tighter, may wobble)
    path_lookahead_time: float = 1.0  # s, look-ahead grows with speed: max(above, time * v)
    path_w_max: float = 1.5  # rad/s, sharpest rotation while following
    path_pivot_deg: float = 60.0  # turn on the spot when the path is this far off the nose
    path_pivot_exit_deg: float = 10.0  # ... until it is back within this angle
    path_goal_tol: float = 0.02  # m, the end point counts as reached within this distance
    # A waypoint where the path bends more than this is a corner: the rover slows down,
    # stops on it and pivots instead of cutting it (0 = never, cut every corner).
    path_corner_deg: float = 45.0

    # Camera pan-tilt (UGV firmware T:133: pan -180 left .. 180 right, tilt -30 down .. 90 up)
    gimbal_speed: float = 90.0  # deg/s while a ZQSD key is held
    gimbal_hz: float = 20.0  # at most this many pan-tilt commands per second
    # s without a key repeat before the camera stops. Longer than manual_timeout: a camera
    # that keeps turning is harmless, while stopping on every late packet makes it stutter
    # on a busy Wi-Fi.
    gimbal_timeout: float = 1.0
    gimbal_spd: int = 0  # firmware servo speed (0 = fastest)
    gimbal_acc: int = 0  # firmware servo acceleration (0 = fastest)
    pan_min: float = -180.0
    pan_max: float = 180.0
    tilt_min: float = -30.0
    tilt_max: float = 90.0
    pan_sign: int = 1  # -1 if the camera turns the wrong way
    tilt_sign: int = 1

    @classmethod
    def load(cls, path: str) -> Config:
        cfg = cls()
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                cfg.update(json.load(f))
        return cfg

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    def update(self, values: dict) -> list[str]:
        """Set the known fields from `values` (with type conversion). Returns the ones set."""
        changed = []
        for fld in dataclasses.fields(self):
            if fld.name in values:
                kind = int if fld.type in (int, 'int') else float
                setattr(self, fld.name, kind(values[fld.name]))
                changed.append(fld.name)
        return changed


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------------------
# Odometry
# --------------------------------------------------------------------------------------


class Odometry:
    """Rover pose from the wheel encoders (distance) and the gyro (heading)."""

    GYRO_STALE = 0.5  # s, after this long without gyro data the encoders give the heading

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bias = 0.0
        self.bias_ready = False
        self._bias_samples: list[float] = []
        self._bias_start: float | None = None
        self.L = self.R = 0.0  # measured wheel speeds, m/s
        self.v = 0.0  # m/s
        self.yaw_rate = 0.0  # rad/s, gyro minus bias
        self.gz_raw: float | None = None
        self._last_wheels: float | None = None
        self._last_gyro: float | None = None
        self._still_since: float | None = None
        self.commanded = False  # set by the rover loop: a non-zero speed is commanded
        self.counters: tuple[float, float] | None = None  # firmware wheel counters, cm
        self._counters0: tuple[float, float] | None = None
        self.reset()

    def reset(self) -> None:
        """Back to (0, 0), heading 0. Keeps the gyro bias."""
        self.x = self.y = 0.0
        self.heading_gyro = 0.0  # rad, not wrapped: +2*pi after a full turn left
        self.heading_enc = 0.0
        self.distance = 0.0  # m, signed (backwards counts negative)
        self.odometer = 0.0  # m, total path length
        self._counters0 = self.counters

    # -- inputs ---------------------------------------------------------------------

    def on_wheels(self, left: float, right: float, t: float) -> None:
        dt = 0.0 if self._last_wheels is None else clamp(t - self._last_wheels, 0.0, 0.2)
        self._last_wheels = t
        s = self.cfg.dist_scale
        self.L, self.R = left * s, right * s
        self.v = (self.L + self.R) / 2.0
        ds = self.v * dt
        self.heading_enc += (self.R - self.L) / self.cfg.track * dt
        h = self.heading
        self.x += ds * math.cos(h)
        self.y += ds * math.sin(h)
        self.distance += ds
        self.odometer += abs(ds)

        still = abs(left) < 0.02 and abs(right) < 0.02 and not self.commanded
        if not still:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = t

    def on_counters(self, odl: float, odr: float) -> None:
        """Firmware wheel counters (cm): a 1 cm resolution cross-check of the distance."""
        self.counters = (odl, odr)
        if self._counters0 is None:
            self._counters0 = self.counters

    @property
    def counter_distance(self) -> float | None:
        """Distance since the last reset from the firmware counters, m (dist_scale applied)."""
        if self.counters is None or self._counters0 is None:
            return None
        dl = self.counters[0] - self._counters0[0]
        dr = self.counters[1] - self._counters0[1]
        return (dl + dr) / 2 / 100 * self.cfg.dist_scale

    def on_gyro(self, gz_raw: float, t: float) -> None:
        self.gz_raw = gz_raw
        gz = gz_raw * self.cfg.gyro_scale * self.cfg.yaw_sign
        dt = 0.0 if self._last_gyro is None else clamp(t - self._last_gyro, 0.0, 0.2)
        self._last_gyro = t

        if not self.bias_ready:
            self._calibrate_bias(gz, t)
            return

        # Slowly re-learn the bias whenever the rover has been still for a while.
        if self._still_since is not None and t - self._still_since > 0.5:
            self.bias += 0.02 * (gz - self.bias)

        self.yaw_rate = gz - self.bias
        self.heading_gyro += self.yaw_rate * dt

    def _calibrate_bias(self, gz: float, t: float) -> None:
        if self._still_since is None:  # moving: start over
            self._bias_samples.clear()
            self._bias_start = None
            return
        if self._bias_start is None:
            self._bias_start = t
        self._bias_samples.append(gz)
        if t - self._bias_start >= self.cfg.bias_calib_time and len(self._bias_samples) >= 5:
            self.bias = sum(self._bias_samples) / len(self._bias_samples)
            self.bias_ready = True
            self._bias_samples.clear()

    def recalibrate_gyro(self) -> None:
        self.bias_ready = False
        self._bias_samples.clear()
        self._bias_start = None

    # -- outputs --------------------------------------------------------------------

    def gyro_alive(self, now: float) -> bool:
        last = self._last_gyro
        return self.bias_ready and last is not None and now - last < self.GYRO_STALE

    @property
    def heading(self) -> float:
        """Best heading: the gyro when it works, else the encoders."""
        if self.bias_ready and self._last_gyro is not None:
            if self._last_wheels is None or self._last_wheels - self._last_gyro < self.GYRO_STALE:
                return self.heading_gyro
        return self.heading_enc

    def snapshot(self, now: float) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'heading_deg': math.degrees(self.heading),
            'heading_gyro_deg': math.degrees(self.heading_gyro),
            'heading_enc_deg': math.degrees(self.heading_enc),
            'heading_source': 'gyro' if self.gyro_alive(now) else 'encoders',
            'distance': self.distance,
            'counter_distance': self.counter_distance,
            'odometer': self.odometer,
            'v': self.v,
            'L': self.L,
            'R': self.R,
            'yaw_rate_dps': math.degrees(self.yaw_rate),
            'gz_raw': self.gz_raw,
            'gyro_bias': self.bias,
            'gyro_ready': self.bias_ready,
        }


# --------------------------------------------------------------------------------------
# Motion controller
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Step:
    kind: str  # 'move' (metres, + = forward), 'turn' (degrees, + = left) or 'path'
    value: float  # path: its length, m
    # Path only: waypoints [[x, y], ...] in metres. frame 'local' = relative to the rover
    # when the step starts (x forward, y left); 'odom' = the odometry frame (pose in /state).
    points: list[list[float]] | None = None
    frame: str = 'local'
    speed: float | None = None  # path cruise speed, m/s (default: path_speed)

    @classmethod
    def path(cls, points, frame: str = 'local', speed: float | None = None) -> Step:
        pts = [[float(p[0]), float(p[1])] for p in points]
        if not pts:
            raise ValueError('a path needs at least one point')
        if frame not in ('local', 'odom'):
            raise ValueError(f'path frame must be "local" or "odom", not {frame!r}')
        if speed is not None and not 0.0 < float(speed) <= 1.0:
            raise ValueError('path speed must be in ]0, 1] m/s')
        start = [[0.0, 0.0]] if frame == 'local' else []
        chain = start + pts
        length = sum(math.dist(a, b) for a, b in zip(chain, chain[1:], strict=False))
        return cls('path', round(length, 3), pts, frame, None if speed is None else float(speed))

    @classmethod
    def parse(cls, d: dict) -> Step:
        if 'move' in d:
            return cls('move', float(d['move']))
        if 'turn' in d:
            return cls('turn', float(d['turn']))
        if 'path' in d:
            return cls.path(d['path'], d.get('frame', 'local'), d.get('speed'))
        raise ValueError(
            f'unknown step {d!r}, expected {{"move": m}}, {{"turn": deg}} '
            'or {"path": [[x, y], ...]}'
        )


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    """Pure pursuit along a polyline given in the odometry frame.

    The path is resampled every 2 cm. Each update finds the closest path point ahead of the
    last one (progress only goes forward, so a path may cross itself), aims at the point one
    look-ahead distance further, and returns the (v, w) that drives an arc through it. When
    that point is far off the nose (a sharp corner, or the start), the rover turns on the spot
    first.
    """

    RESOLUTION = 0.02  # m
    SEARCH = 0.5  # m, how far ahead of the last progress the closest point is looked for

    def __init__(
        self,
        points: list[tuple[float, float]],
        start: tuple[float, float],
        corner_deg: float = 0.0,
    ) -> None:
        chain = [tuple(start)] if math.dist(start, points[0]) > self.RESOLUTION else []
        chain += [tuple(p) for p in points]
        chain = [p for k, p in enumerate(chain) if k == 0 or math.dist(p, chain[k - 1]) > 1e-6]
        self.pts: list[tuple[float, float]] = [chain[0]]
        self.corners: list[int] = []  # indices in pts of the waypoints to stop and pivot on
        for k, (a, b) in enumerate(zip(chain, chain[1:], strict=False)):
            if 0 < k and corner_deg > 0:
                before = math.atan2(a[1] - chain[k - 1][1], a[0] - chain[k - 1][0])
                after = math.atan2(b[1] - a[1], b[0] - a[0])
                if abs(wrap_angle(after - before)) > math.radians(corner_deg):
                    self.corners.append(len(self.pts) - 1)
            n = max(1, math.ceil(math.dist(a, b) / self.RESOLUTION))
            self.pts += [
                (a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n)
                for j in range(1, n + 1)
            ]
        self.s = [0.0]
        for a, b in zip(self.pts, self.pts[1:], strict=False):
            self.s.append(self.s[-1] + math.dist(a, b))
        self.length = self.s[-1]
        self.i = 0  # progress: index of the closest path point
        self.target = self.pts[-1]
        self.cte = 0.0  # m, distance from the rover to the path (cross-track error)
        self.max_cte = 0.0
        # Start by facing the path (the first update ends this at once if already aligned).
        self.pivoting = True

    @property
    def end(self) -> tuple[float, float]:
        return self.pts[-1]

    @property
    def remaining(self) -> float:
        return self.length - self.s[self.i]

    def update(
        self, x: float, y: float, heading: float, v_now: float, v_cap: float, cfg: Config
    ) -> tuple[float, float, bool]:
        """(v, w, done) for the rover at (x, y, heading); v_cap = speed allowed by the ramp."""
        c = cfg
        pos = (x, y)
        last = min(len(self.pts) - 1, self.i + int(self.SEARCH / self.RESOLUTION))
        self.i = min(range(self.i, last + 1), key=lambda j: math.dist(pos, self.pts[j]))
        self.cte = math.dist(pos, self.pts[self.i])
        self.max_cte = max(self.max_cte, self.cte)

        # End: close enough to the last point, or just past it.
        d_end = math.dist(pos, self.end)
        lookahead = max(c.path_lookahead, c.path_lookahead_time * abs(v_now))
        if self.remaining < lookahead:
            ahead = (self.end[0] - x) * math.cos(heading) + (self.end[1] - y) * math.sin(heading)
            if d_end - abs(v_now) * c.lag <= c.path_goal_tol or (ahead < 0 and d_end < lookahead):
                return 0.0, 0.0, True

        # Next corner still ahead: it is the look-ahead's limit and the place to slow down for.
        while self.corners and (
            self.i >= self.corners[0]
            or (math.dist(pos, self.pts[self.corners[0]]) <= c.path_goal_tol + abs(v_now) * c.lag)
        ):
            self.i = max(self.i, self.corners.pop(0))  # corner reached: go on to the next leg
            self.pivoting = True
        stop_at = self.corners[0] if self.corners else len(self.pts) - 1
        j = self.i
        while j < stop_at and self.s[j] - self.s[self.i] < lookahead:
            j += 1
        self.target = self.pts[j]
        to_stop = self.s[stop_at] - self.s[self.i] + self.cte
        dx, dy = self.target[0] - x, self.target[1] - y
        alpha = wrap_angle(math.atan2(dy, dx) - heading)

        if self.pivoting:
            self.pivoting = abs(alpha) > math.radians(c.path_pivot_exit_deg)
        else:
            self.pivoting = abs(alpha) > math.radians(c.path_pivot_deg)
        if self.pivoting:
            w = math.copysign(clamp(c.k_turn * abs(alpha), c.w_min, c.w_max), alpha)
            return 0.0, w, False

        curvature = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1e-3)
        v = min(v_cap, c.k_dist * (to_stop if self.corners else (self.remaining + d_end) / 2))
        if curvature:
            v = min(v, c.path_w_max / abs(curvature))
        v = max(v, c.v_min)
        return v, clamp(v * curvature, -c.path_w_max, c.path_w_max), False

    def snapshot(self) -> dict:
        step = max(1, len(self.pts) // 150)  # at most ~150 points for the page
        pts = self.pts[::step] + ([self.pts[-1]] if (len(self.pts) - 1) % step else [])
        return {
            'points': [[round(px, 3), round(py, 3)] for px, py in pts],
            'target': [round(self.target[0], 3), round(self.target[1], 3)],
            'length': round(self.length, 3),
            'done_m': round(self.s[self.i], 3),
            'cte': round(self.cte, 3),
            'max_cte': round(self.max_cte, 3),
            'pivoting': self.pivoting,
        }


class SteeringAssist:
    """Gyro steering help for manual driving: a software differential.

    * Straight input (w = 0 while moving): hold the heading the rover had when the straight
      input started (or `assist_settle` s after a turn ended), with a PI correction on the
      heading error.
    * Turn input: make the rotation speed measured by the gyro follow the input (the rover
      skids in turns and on loose ground, so it often turns slower than asked).

    Only rotations show on the gyro: a sideways slide on a slope, without rotation, is not
    seen and not corrected.
    """

    def __init__(self, cfg: Config, odom: Odometry) -> None:
        self.cfg = cfg
        self.odom = odom
        self.mode = 'off'  # off | idle | hold | settling | rate
        self.reset()

    def reset(self, mode: str = 'idle') -> None:
        self.mode = mode
        self.ref: float | None = None  # held heading, rad
        self.integral = 0.0
        self.w_corr = 0.0  # rad/s added to the input
        self.err = 0.0  # rad (hold) or rad/s (rate)
        self._released_at: float | None = None
        self._next_update = 0.0

    def apply(self, v: float, w: float, t: float) -> float:
        """Rotation command to send for the input (v, w)."""
        c = self.cfg
        if not c.assist or not self.odom.gyro_alive(t):
            if self.mode != 'off':
                self.reset('off')
            return w
        if abs(v) < c.assist_min_speed and w == 0:
            if self.mode != 'idle':
                self.reset('idle')
            return w
        if t + 1e-3 >= self._next_update:  # 1 ms margin: loop ticks are not exact
            period = 1.0 / clamp(c.assist_hz, 0.5, 50.0)
            self._next_update = t + period
            self._update(w, t, period)
        return w + self.w_corr

    def _update(self, w: float, t: float, period: float) -> None:
        c, o = self.cfg, self.odom
        if w != 0:
            if self.mode != 'rate':
                self.integral = 0.0
            self.mode = 'rate'
            self.ref = None
            self._released_at = None
            self.err = w - o.yaw_rate
            if c.assist_ki_rate > 0:
                limit = c.assist_w_max / c.assist_ki_rate
                self.integral = clamp(self.integral + self.err * period, -limit, limit)
            else:
                self.integral = 0.0
            self.w_corr = clamp(
                c.assist_k_rate * self.err + c.assist_ki_rate * self.integral,
                -c.assist_w_max,
                c.assist_w_max,
            )
            return
        if self.ref is None:
            after_turn = self.mode in ('rate', 'settling')
            if after_turn:
                if self._released_at is None:
                    self._released_at = t
                if t - self._released_at + 1e-3 < c.assist_settle:
                    self.mode = 'settling'  # let the rotation die out first
                    self.w_corr = 0.0
                    return
            self.ref = o.heading
            self.integral = 0.0
        self.mode = 'hold'
        self.err = self.ref - o.heading
        e = self.err if abs(self.err) > math.radians(c.assist_deadband_deg) else 0.0
        if c.assist_ki > 0:
            limit = c.assist_w_max / c.assist_ki
            self.integral = clamp(self.integral + e * period, -limit, limit)
        else:
            self.integral = 0.0
        self.w_corr = clamp(
            c.assist_k * e + c.assist_ki * self.integral, -c.assist_w_max, c.assist_w_max
        )

    def snapshot(self) -> dict:
        unit = 'deg' if self.mode == 'hold' else 'deg/s'
        return {
            'mode': self.mode,
            'ref_deg': None if self.ref is None else math.degrees(self.ref),
            'err': math.degrees(self.err),
            'err_unit': unit,
            'w_corr': self.w_corr,
        }


class Gimbal:
    """Camera pan-tilt with the UGV firmware's absolute command T:133.

    The firmware does not report the servo angles, so this keeps the commanded ones. Held
    ZQSD keys set a rate (-1, 0 or 1 per axis); the angles are integrated here and sent at
    most `gimbal_hz` times per second, only when they changed.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pan = 0.0  # deg, + = right
        self.tilt = 0.0  # deg, + = up
        self._rate = (0.0, 0.0)
        self._rate_t = 0.0
        self._sent: tuple[float, float] | None = None
        self._next_send = 0.0

    def set_rate(self, pan_rate: float, tilt_rate: float, t: float) -> None:
        self._rate = (clamp(pan_rate, -1.0, 1.0), clamp(tilt_rate, -1.0, 1.0))
        self._rate_t = t

    def set_angles(self, pan: float, tilt: float) -> None:
        c = self.cfg
        self._rate = (0.0, 0.0)
        self.pan = clamp(pan, c.pan_min, c.pan_max)
        self.tilt = clamp(tilt, c.tilt_min, c.tilt_max)

    def resend(self) -> None:
        """Send the angles again at the next step (e.g. after the ESP32 rebooted)."""
        self._sent = None

    def step(self, t: float, dt: float) -> dict | None:
        """The T:133 command to send now, or None."""
        c = self.cfg
        pr, tr = self._rate
        if t - self._rate_t > c.gimbal_timeout:  # key repeats stopped (connection lost)
            pr = tr = 0.0
            self._rate = (0.0, 0.0)
        if pr or tr:
            self.pan = clamp(self.pan + pr * c.gimbal_speed * dt, c.pan_min, c.pan_max)
            self.tilt = clamp(self.tilt + tr * c.gimbal_speed * dt, c.tilt_min, c.tilt_max)
        target = (round(self.pan, 1), round(self.tilt, 1))
        if target == self._sent or t < self._next_send:
            return None
        self._sent = target
        self._next_send = t + 1.0 / clamp(c.gimbal_hz, 0.5, 50.0)
        return {
            'T': 133,
            'X': round(target[0] * c.pan_sign, 1),
            'Y': round(target[1] * c.tilt_sign, 1),
            'SPD': c.gimbal_spd,
            'ACC': c.gimbal_acc,
        }

    def snapshot(self) -> dict:
        return {'pan': self.pan, 'tilt': self.tilt, 'moving': self._rate != (0.0, 0.0)}


class Controller:
    """Turns manual commands and move/turn steps into (v, w) for the ESP32.

    Call `step(t)` at a fixed rate (20 Hz); it returns (v m/s, w rad/s).
    """

    def __init__(self, cfg: Config, odom: Odometry) -> None:
        self.cfg = cfg
        self.odom = odom
        self.mode = 'idle'  # idle | manual | move | turn | settle
        self.queue: deque[Step] = deque()
        self.current: Step | None = None
        self.history: deque[dict] = deque(maxlen=50)
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._manual = (0.0, 0.0)
        self._manual_t = 0.0
        self._t_last: float | None = None
        self._task: dict = {}
        self._settle_until = 0.0
        self._pending_result: dict | None = None
        self.assist = SteeringAssist(cfg, odom)

    # -- commands -------------------------------------------------------------------

    def manual(self, v: float, w: float, t: float) -> None:
        """Drive at (v, w) until the next command or `manual_timeout`. Cancels any steps."""
        if self.mode not in ('manual', 'idle'):
            self._abort('cancelled by manual control', t)
        self._manual = (v, w)
        self._manual_t = t
        self.mode = 'manual' if (v or w) else 'idle'

    def run(self, steps: list[Step], t: float) -> None:
        """Replace whatever runs with this list of steps."""
        if self.mode in ('move', 'turn', 'path', 'settle'):
            self._abort('replaced by a new command', t)
        self.queue = deque(steps)
        self.current = None
        self.mode = 'idle'

    def stop(self, t: float) -> None:
        self._abort('stopped', t)
        self.queue.clear()
        self._manual = (0.0, 0.0)
        self.mode = 'idle'
        self.v_cmd = self.w_cmd = 0.0

    # -- control loop ---------------------------------------------------------------

    def step(self, t: float) -> tuple[float, float]:
        dt = 0.05 if self._t_last is None else clamp(t - self._t_last, 0.0, 0.2)
        self._t_last = t

        if self.mode == 'manual':
            if t - self._manual_t > self.cfg.manual_timeout:
                self.mode = 'idle'
                v, w = 0.0, 0.0
            else:
                v, w = self._manual
                w = self.assist.apply(v, w, t)
        elif self.mode == 'settle':
            v, w = 0.0, 0.0
            if t >= self._settle_until:
                if self._needs_correction():
                    self._task['corrections'] += 1
                    self._task['stall_since'] = None
                    self._task['v_ramp'] = 0.0
                    self._pending_result = None
                    self.mode = self.current.kind
                else:
                    self._finish_result()
                    self.mode = 'idle'
        else:
            v, w = 0.0, 0.0

        if self.mode != 'manual' and self.assist.mode not in ('idle', 'off'):
            self.assist.reset()

        if self.mode == 'idle' and self.queue:
            self._start(self.queue.popleft(), t)

        if self.mode == 'move':
            v, w = self._move(t, dt)
        elif self.mode == 'turn':
            v, w = self._turn(t)
        elif self.mode == 'path':
            v, w = self._path(t, dt)

        self.v_cmd, self.w_cmd = v, w
        return v, w

    def _start(self, s: Step, t: float) -> None:
        self.current = s
        self.mode = s.kind
        o = self.odom
        self._task = {
            'step': s,
            't0': t,
            'dist0': o.distance,
            'odo0': o.odometer,
            'cnt0': o.counter_distance,
            'head0': o.heading,
            'x0': o.x,
            'y0': o.y,
            # Absolute goal, in odometry units: heading (rad) for a turn, distance (m) for a move.
            'goal': (o.heading + math.radians(s.value))
            if s.kind == 'turn'
            else (o.distance + s.value),
            'corrections': 0,
            'v_ramp': 0.0,
            'stall_since': None,
        }
        if s.kind == 'path':
            h, x0, y0 = o.heading, o.x, o.y
            if s.frame == 'local':  # x forward, y left of the rover now -> odometry frame
                pts = [
                    (
                        x0 + px * math.cos(h) - py * math.sin(h),
                        y0 + px * math.sin(h) + py * math.cos(h),
                    )
                    for px, py in s.points
                ]
            else:
                pts = [(px, py) for px, py in s.points]
            follower = PathFollower(pts, (x0, y0), self.cfg.path_corner_deg)
            speed = s.speed or self.cfg.path_speed
            self._task['follower'] = follower
            self._task['speed'] = speed
            # Generous: pivots and slowing down near the end take time too.
            self._task['timeout'] = max(
                self.cfg.task_timeout, 3.0 * follower.length / speed + 10.0
            )

    def _needs_correction(self) -> bool:
        """The rover stopped too early or too late: move or turn again towards the goal."""
        k = self._task
        if self.current is None or self.current.kind == 'path':
            return False
        if (self._pending_result or {}).get('status') != 'done':
            return False
        if k['corrections'] >= self.cfg.corrections:
            return False
        if self.current.kind == 'turn':
            return abs(k['goal'] - self.odom.heading) > math.radians(self.cfg.angle_tol_deg)
        return abs(k['goal'] - self.odom.distance) > 2 * self.cfg.dist_tol

    def _move(self, t: float, dt: float) -> tuple[float, float]:
        c, o, k = self.cfg, self.odom, self._task
        error = k['goal'] - o.distance
        direction = 1.0 if error >= 0 else -1.0
        remaining = abs(error)
        coasting = abs(o.v) * c.lag if o.v * direction > 0 else 0.0
        if remaining - coasting <= c.dist_tol / 2:
            return self._done('done', t)
        if self._failed(t):
            return 0.0, 0.0

        k['v_ramp'] = min(c.v_max, k['v_ramp'] + c.accel * dt)
        speed = clamp(c.k_dist * remaining, c.v_min, c.v_max)
        speed = min(speed, max(k['v_ramp'], c.v_min))
        w = clamp(c.k_head * (k['head0'] - o.heading), -1.0, 1.0)
        return direction * speed, w

    def _turn(self, t: float) -> tuple[float, float]:
        c, o, k = self.cfg, self.odom, self._task
        error = k['goal'] - o.heading
        direction = 1.0 if error >= 0 else -1.0
        remaining = abs(error)
        # Only anticipate the inertia when turning towards the goal (not after an overshoot).
        coasting = abs(o.yaw_rate) * c.lag if o.yaw_rate * direction > 0 else 0.0
        if remaining - coasting <= math.radians(c.angle_tol_deg) / 2:
            return self._done('done', t)
        if self._failed(t):
            return 0.0, 0.0
        return 0.0, direction * clamp(c.k_turn * remaining, c.w_min, c.w_max)

    def _path(self, t: float, dt: float) -> tuple[float, float]:
        c, o, k = self.cfg, self.odom, self._task
        k['v_ramp'] = min(k['speed'], k['v_ramp'] + c.accel * dt)
        v, w, done = k['follower'].update(o.x, o.y, o.heading, o.v, max(k['v_ramp'], c.v_min), c)
        if done:
            return self._done('done', t)
        if self._failed(t):
            return 0.0, 0.0
        return v, w

    def _failed(self, t: float) -> bool:
        """Timeout, or wheels not turning for `stall_time` while commanded."""
        k = self._task
        if t - k['t0'] > k.get('timeout', self.cfg.task_timeout):
            self._done('timeout', t)
            return True
        wheels_turn = abs(self.odom.L) > 0.01 or abs(self.odom.R) > 0.01
        if wheels_turn:
            k['stall_since'] = None
        elif k['stall_since'] is None:
            k['stall_since'] = t
        elif t - k['stall_since'] > self.cfg.stall_time:
            self._done('stalled', t)
            self.queue.clear()
            return True
        return False

    def _done(self, status: str, t: float) -> tuple[float, float]:
        self._pending_result = {'status': status, 'duration': t - self._task['t0']}
        self.mode = 'settle'
        self._settle_until = t + self.cfg.settle
        if status != 'done':
            self.queue.clear()
        return 0.0, 0.0

    def _finish_result(self) -> None:
        """Measure the step once the rover has stopped, and log it."""
        k, o = self._task, self.odom
        s: Step = k['step']
        res = self._pending_result or {'status': 'done', 'duration': 0.0}
        dist = o.distance - k['dist0']
        turned = math.degrees(o.heading - k['head0'])
        entry = {
            'kind': s.kind,
            'target': s.value,
            'status': res['status'],
            'duration': round(res['duration'], 2),
            'corrections': k.get('corrections', 0),
            'distance': round(dist, 4),
            'turned_deg': round(turned, 2),
            'drift_m': round(
                math.hypot(o.x - k['x0'], o.y - k['y0']) if s.kind == 'turn' else 0.0, 4
            ),
        }
        if s.kind == 'move':
            entry['error'] = round(dist - s.value, 4)
            if k.get('cnt0') is not None and o.counter_distance is not None:
                entry['counter_distance'] = round(o.counter_distance - k['cnt0'], 3)
        elif s.kind == 'path':
            f: PathFollower = k['follower']
            entry['target'] = round(f.length, 3)  # including the way to the first point
            entry['error'] = round(math.dist((o.x, o.y), f.end), 4)  # end point miss, m
            entry['max_cte'] = round(f.max_cte, 4)  # largest distance from the path, m
            entry['distance'] = round(o.odometer - k['odo0'], 4)
        else:
            entry['error'] = round(turned - s.value, 2)
        self.history.appendleft(entry)
        self.current = None
        self._pending_result = None

    def _abort(self, reason: str, t: float) -> None:
        if self.mode in ('move', 'turn', 'path'):
            self._pending_result = {'status': reason, 'duration': t - self._task['t0']}
            self._finish_result()
        elif self.mode == 'settle':
            self._finish_result()
        self.current = None

    def snapshot(self) -> dict:
        k = self._task
        progress = None
        if self.mode in ('move', 'turn', 'path') and k:
            if self.mode == 'move':
                progress = self.odom.distance - k['dist0']
            elif self.mode == 'turn':
                progress = math.degrees(self.odom.heading - k['head0'])
            else:
                progress = k['follower'].s[k['follower'].i]
        follower = k.get('follower') if self.current is not None else None
        return {
            'mode': self.mode,
            'current': dataclasses.asdict(self.current) if self.current else None,
            'progress': progress,
            'queued': [dataclasses.asdict(s) for s in self.queue],
            'v_cmd': self.v_cmd,
            'w_cmd': self.w_cmd,
            'assist': self.assist.snapshot(),
            'path': follower.snapshot() if follower else None,
            'history': list(self.history),
        }


# --------------------------------------------------------------------------------------
# ESP32 link
# --------------------------------------------------------------------------------------


def find_esp32_port() -> str | None:
    """First USB serial port that looks like the driver board."""
    for pattern in ('/dev/ttyUSB*', '/dev/ttyACM*'):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    try:  # Windows / macOS
        from serial.tools import list_ports

        for p in list_ports.comports():
            if p.vid is not None:
                return p.device
    except ImportError:
        pass
    return None


class Esp32Link:
    """JSON-lines link to the ESP32 driver board, with automatic reconnection.

    `opener()` returns an open serial-like object (pyserial, or the simulated board).
    Each JSON message received is passed to `on_message(msg, receive_time)`.
    """

    INIT_COMMANDS = (
        {'T': 900, 'main': 1, 'module': 0},  # robot type = RaspRover (wheel size, track)
        {'T': 143, 'cmd': 0},  # no command echo
        {'T': 142, 'cmd': 50},  # feedback at most every 50 ms
        {'T': 131, 'cmd': 1},  # continuous chassis feedback (T:1001)
    )

    BOOT_TIMEOUT = 15.0  # s

    def __init__(
        self,
        opener: Callable[[], object],
        on_message: Callable[[dict, float], None],
        wait_boot: bool = True,
    ) -> None:
        self._opener = opener
        self._wait_boot_enabled = wait_boot
        self._on_message = on_message
        self._ser = None
        self._lock = threading.Lock()
        self.connected = False
        self.error: str | None = None
        self.last_rx: float | None = None
        self.rx_count = 0
        self.bad_lines = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='esp32-rx', daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except OSError:
                    pass

    def send(self, msg: dict) -> bool:
        line = (json.dumps(msg, separators=(',', ':')) + '\n').encode()
        with self._lock:
            if self._ser is None:
                return False
            try:
                self._ser.write(line)
                return True
            except Exception as e:  # noqa: BLE001 - any serial error means "disconnected"
                self._drop(f'write failed: {e}')
                return False

    def _drop(self, reason: str) -> None:
        self.connected = False
        self.error = reason
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass
        self._ser = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._ser is None:
                try:
                    ser = self._opener()
                except Exception as e:  # noqa: BLE001
                    self.error = f'cannot open the ESP32 port: {e}'
                    self._stop.wait(2.0)
                    continue
                if self._wait_boot_enabled:
                    self.error = 'ESP32 booting (keep the rover still)'
                    try:
                        self._wait_boot(ser)
                    except Exception as e:  # noqa: BLE001
                        self.error = f'ESP32 boot: {e}'
                        try:
                            ser.close()
                        except Exception:  # noqa: BLE001
                            pass
                        self._stop.wait(2.0)
                        continue
                with self._lock:
                    self._ser = ser
                for cmd in self.INIT_COMMANDS:
                    self.send(cmd)
                    time.sleep(0.05)
                self.connected = True
                self.error = None
            try:
                raw = self._ser.readline()
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._drop(f'read failed: {e}')
                continue
            text = raw.decode('utf-8', errors='replace').strip() if raw else ''
            if not text:
                continue
            t = time.monotonic()
            try:
                msg = json.loads(text)
            except ValueError:
                self.bad_lines += 1
                continue
            if isinstance(msg, dict):
                self.last_rx = t
                self.rx_count += 1
                self._on_message(msg, t)

    def _wait_boot(self, ser) -> None:
        """Opening the port reboots the ESP32: wait for "UGV started" before sending anything.

        If the board did not reboot (no boot text, feedback already flowing), go on at once.
        """
        t0 = time.monotonic()
        booting = False
        while time.monotonic() - t0 < self.BOOT_TIMEOUT and not self._stop.is_set():
            raw = ser.readline()
            text = raw.decode('utf-8', errors='replace').strip() if raw else ''
            if 'UGV started' in text:
                time.sleep(0.5)
                break
            if text and not text.startswith('{'):
                booting = True
            elif text.startswith('{"T":1001') and not booting and time.monotonic() - t0 > 1.0:
                break
        ser.reset_input_buffer()


def open_serial(port: str):
    """Open the real board. DTR/RTS low, although the board still reboots (CH343 + ESP32)."""
    import serial  # pyserial, only needed on the robot

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUDRATE
    ser.timeout = 0.2
    ser.dtr = False  # DTR/RTS low: do not hold the ESP32 in reset
    ser.rts = False
    ser.open()
    return ser


# --------------------------------------------------------------------------------------
# Rover: link + odometry + controller + 20 Hz loop
# --------------------------------------------------------------------------------------


def battery_volts(v: float | None) -> float | None:
    """The UGV firmware sends centivolts (1158 = 11.58 V); older ones send volts."""
    if v is None:
        return None
    return v / 100.0 if v > 50 else float(v)


def tilt_deg(msg: dict) -> dict:
    """Roll and pitch, from the firmware (r, p) or from the raw accelerometer."""
    if 'r' in msg and 'p' in msg:
        return {'roll_deg': msg['r'], 'pitch_deg': msg['p']}
    if all(k in msg for k in ('ax', 'ay', 'az')) and (msg['ay'] or msg['az']):
        ax, ay, az = float(msg['ax']), float(msg['ay']), float(msg['az'])
        return {
            'roll_deg': math.degrees(math.atan2(ay, az)),
            'pitch_deg': math.degrees(math.atan2(-ax, math.hypot(ay, az))),
        }
    return {'roll_deg': None, 'pitch_deg': None}


class Rover:
    LOOP_HZ = 20.0

    def __init__(self, cfg: Config, opener: Callable[[], object], wait_boot: bool = True) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.odom = Odometry(cfg)
        self.ctrl = Controller(cfg, self.odom)
        self.link = Esp32Link(opener, self._on_message, wait_boot)
        self.chassis: dict = {}  # last T:1001
        self.imu: dict = {}  # last T:1002
        self._gyro_in_feedback = False  # UGV firmware: gz comes with T:1001
        self.gimbal = Gimbal(cfg)
        self._was_connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='rover-loop', daemon=True)

    def start(self) -> None:
        self.link.start()
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.link.send({'T': 13, 'X': 0, 'Z': 0})
        self.link.close()

    # -- commands (thread safe) -----------------------------------------------------

    def manual(self, v: float, w: float) -> None:
        with self.lock:
            self.ctrl.manual(v, w, time.monotonic())

    def run(self, steps: list[Step]) -> None:
        with self.lock:
            self.ctrl.run(steps, time.monotonic())

    def stop(self) -> None:
        with self.lock:
            self.ctrl.stop(time.monotonic())
        self.link.send({'T': 13, 'X': 0, 'Z': 0})

    def reset_odometry(self) -> None:
        with self.lock:
            self.odom.reset()

    def recalibrate_gyro(self) -> None:
        with self.lock:
            self.odom.recalibrate_gyro()

    def gimbal_rate(self, pan_rate: float, tilt_rate: float) -> None:
        """Move the camera while repeated (like the arrow keys): -1, 0 or 1 per axis."""
        with self.lock:
            self.gimbal.set_rate(pan_rate, tilt_rate, time.monotonic())

    def gimbal_set(self, pan: float, tilt: float) -> None:
        with self.lock:
            self.gimbal.set_angles(pan, tilt)

    def raw(self, msg: dict) -> None:
        """Forward a raw JSON command to the ESP32 (e.g. OLED text)."""
        self.link.send(msg)

    # -- internals ------------------------------------------------------------------

    def _on_message(self, msg: dict, t: float) -> None:
        kind = msg.get('T')
        with self.lock:
            if kind == 1001:
                self.chassis = msg
                self.odom.on_wheels(float(msg.get('L', 0.0)), float(msg.get('R', 0.0)), t)
                if 'gz' in msg:
                    self._gyro_in_feedback = True
                    self.odom.on_gyro(float(msg['gz']), t)
                if 'odl' in msg and 'odr' in msg:
                    self.odom.on_counters(float(msg['odl']), float(msg['odr']))
            elif kind == 1002:  # other firmwares: gyro only on request (T:126)
                self.imu = msg
                if 'gz' in msg and not self._gyro_in_feedback:
                    self.odom.on_gyro(float(msg['gz']), t)

    def _loop(self) -> None:
        period = 1.0 / self.LOOP_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            connected = self.link.connected
            with self.lock:
                v, w = self.ctrl.step(t)
                self.odom.commanded = abs(v) > 1e-4 or abs(w) > 1e-4
                if connected and not self._was_connected:
                    self.gimbal.resend()  # the ESP32 (re)booted: put the camera back
                gimbal_cmd = self.gimbal.step(t, period) if connected else None
            self._was_connected = connected
            # Sent every tick: also the heartbeat (the ESP32 stops the motors after 3 s).
            self.link.send({'T': 13, 'X': round(v, 4), 'Z': round(w, 4)})
            if gimbal_cmd is not None:
                self.link.send(gimbal_cmd)
            if not self._gyro_in_feedback:  # older firmwares: ask for the IMU (T:1002)
                self.link.send({'T': 126})
            nxt += period
            self._stop.wait(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:  # fell far behind: do not burst to catch up
                nxt = time.monotonic()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            link = self.link
            return {
                'link': {
                    'connected': link.connected,
                    'error': link.error,
                    'rx_age': None if link.last_rx is None else now - link.last_rx,
                    'rx_count': link.rx_count,
                    'bad_lines': link.bad_lines,
                },
                'battery_v': battery_volts(self.chassis.get('v')),
                **tilt_deg(self.chassis),
                'counters_cm': self.odom.counters,
                'gimbal': self.gimbal.snapshot(),
                'odom': self.odom.snapshot(now),
                'ctrl': self.ctrl.snapshot(),
            }
