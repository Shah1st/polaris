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
import traceback
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
    # Give up a turn (or a pivot on a path) when the wheels turn in opposite directions but
    # the gyro sees less than no_rotation_rate for no_rotation_time: rover lifted or stuck.
    no_rotation_time: float = 2.0  # s
    no_rotation_rate: float = 0.1  # rad/s

    # Tilt safety. Roll = sideways (+ = left side up), pitch = nose (+ = nose up), degrees.
    # PROVISIONAL thresholds: to be set from the tipping tests (record them on the page).
    tilt_guard: int = 1  # 1 on, 0 off (off while tipping the rover on purpose)
    roll_warn_deg: float = 15.0  # from here the rover slows down...
    roll_stop_deg: float = 25.0  # ... and here it stops and backs off
    pitch_warn_deg: float = 15.0
    pitch_stop_deg: float = 25.0
    tilt_min_scale: float = 0.3  # speed fraction left at the stop angle (linear from warn)
    # Tipping in progress, before the stop angle: once tilted more than tip_min_deg, the
    # angle grows faster than tip_rate_dps, or accelerates away from level faster than
    # tip_acc_dps2.
    tip_min_deg: float = 8.0
    tip_rate_dps: float = 40.0
    tip_acc_dps2: float = 400.0
    tilt_backoff_dist: float = 0.15  # m, reverse this far along the way the rover came
    tilt_backoff_speed: float = 0.12  # m/s
    tilt_backoff_time: float = 3.0  # s, at most
    tilt_filter_tau: float = 0.5  # s, gyro/accelerometer blend (higher: trust the gyro more)
    imu_roll_sign: int = 1  # -1 if the gyro's roll rate has the wrong sign (see /state tilt)
    imu_pitch_sign: int = 1

    # Calibration
    dist_scale: float = 1.0  # real distance / distance measured by the encoders
    yaw_sign: int = 1  # -1 if the heading decreases when the rover turns left
    # raw gz -> rad/s. UGV firmware: 16.4 raw per deg/s (+-2000 deg/s range), checked by
    # turning the rover by hand: 16.2 to 17.0 on 2026-11-14.
    gyro_scale: float = math.radians(1.0) / 16.4
    track: float = 0.125  # m, wheel track used by the ESP32 firmware for the RaspRover
    bias_calib_time: float = 2.0  # s of stillness to measure the gyro bias
    # rad/s: a gyro reading further than this from the bias is a real rotation (the rover
    # carried or pushed by hand, wheels still), never learned as bias. Noise is ~0.008.
    bias_max_rate: float = 0.03
    # rad/s: a gyro bias is small (ours: ~0.004). A larger average during the start-up
    # measure means the rover was turning (carried): measure again.
    bias_limit: float = 0.05

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
        """Settings from `path`. Never fails: the robot must start even with a bad file."""
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, encoding='utf-8') as f:
                values = json.load(f)
            if not isinstance(values, dict):
                raise ValueError('not a JSON object')
        except (OSError, ValueError) as e:
            os.replace(path, path + '.bad')
            print(f'Settings file {path} unreadable ({e}): moved to .bad, using the defaults.')
            return cfg
        try:
            cfg.update(values)
        except (ValueError, TypeError):  # keep the good values, drop the bad ones
            for name, value in values.items():
                try:
                    cfg.update({name: value})
                except (ValueError, TypeError) as e:
                    print(f'Settings file {path}: {e}; using the default.')
        return cfg

    def save(self, path: str) -> None:
        # Temporary file + rename: a power cut while saving leaves the old file, not half a file.
        tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f, indent=2, allow_nan=False)
        os.replace(tmp, path)

    def update(self, values: dict) -> list[str]:
        """Set the known fields from `values` (with type conversion). Returns the ones set.

        All or nothing: a value that is not finite, out of CONFIG_LIMITS or inconsistent with
        the others raises ValueError and changes nothing.
        """
        new = {}
        for fld in dataclasses.fields(self):
            if fld.name not in values:
                continue
            raw = values[fld.name]
            if isinstance(raw, bool) or raw is None:
                raise ValueError(f'{fld.name} must be a number, not {raw!r}')
            x = float(raw)
            lo, hi = CONFIG_LIMITS[fld.name]
            if not (math.isfinite(x) and lo <= x <= hi):
                raise ValueError(f'{fld.name} must be within [{lo}, {hi}], not {raw!r}')
            if fld.type in (int, 'int'):
                if x != int(x):
                    raise ValueError(f'{fld.name} must be a whole number, not {raw!r}')
                x = int(x)
            new[fld.name] = x
        merged = {**dataclasses.asdict(self), **new}
        for name in ('yaw_sign', 'pan_sign', 'tilt_sign', 'imu_roll_sign', 'imu_pitch_sign'):
            if merged[name] not in (-1, 1):
                raise ValueError(f'{name} must be 1 or -1')
        for lo_name, hi_name in (
            ('v_min', 'v_max'),
            ('w_min', 'w_max'),
            ('pan_min', 'pan_max'),
            ('tilt_min', 'tilt_max'),
            ('roll_warn_deg', 'roll_stop_deg'),
            ('pitch_warn_deg', 'pitch_stop_deg'),
        ):
            if merged[lo_name] > merged[hi_name]:
                raise ValueError(f'{lo_name} must not be above {hi_name}')
        for name, x in new.items():
            setattr(self, name, x)
        return list(new)


# Accepted range of each setting. Wide on purpose: they refuse nonsense (NaN, zero where the
# code divides, a value off by 1000, a disabled dead man), they do not tune the rover.
CONFIG_LIMITS: dict[str, tuple[float, float]] = {
    'v_manual': (0.0, 1.0),
    'v_manual_fast': (0.0, 1.0),
    'w_manual': (0.0, 5.0),
    'w_manual_fast': (0.0, 5.0),
    'manual_timeout': (0.1, 2.0),  # the dead man: cannot be switched off
    'v_max': (0.01, 1.0),
    'v_min': (0.0, 1.0),
    'accel': (0.01, 10.0),
    'k_dist': (0.01, 50.0),
    'k_head': (0.0, 50.0),
    'dist_tol': (0.0005, 0.2),
    'w_min': (0.0, 5.0),
    'w_max': (0.05, 5.0),
    'k_turn': (0.01, 50.0),
    'angle_tol_deg': (0.05, 30.0),
    'corrections': (0, 10),
    'lag': (0.0, 2.0),
    'settle': (0.0, 5.0),
    'task_timeout': (1.0, 120.0),
    'stall_time': (0.2, 10.0),
    'no_rotation_time': (0.2, 30.0),
    'no_rotation_rate': (0.01, 2.0),  # 0 would never see "not rotating"
    'tilt_guard': (0, 1),
    'roll_warn_deg': (0.0, 90.0),
    'roll_stop_deg': (1.0, 90.0),
    'pitch_warn_deg': (0.0, 90.0),
    'pitch_stop_deg': (1.0, 90.0),
    'tilt_min_scale': (0.0, 1.0),
    'tip_min_deg': (0.0, 90.0),
    'tip_rate_dps': (1.0, 2000.0),  # the gyro's range
    'tip_acc_dps2': (1.0, 100000.0),
    'tilt_backoff_dist': (0.0, 1.0),
    'tilt_backoff_speed': (0.0, 1.0),
    'tilt_backoff_time': (0.0, 10.0),
    'tilt_filter_tau': (0.01, 10.0),
    'imu_roll_sign': (-1, 1),
    'imu_pitch_sign': (-1, 1),
    'dist_scale': (0.5, 2.0),
    'yaw_sign': (-1, 1),
    'gyro_scale': (1e-5, 1e-2),
    'track': (0.05, 1.0),  # divides the wheel speed difference
    'bias_calib_time': (0.2, 30.0),
    'bias_max_rate': (0.005, 1.0),  # the start-up measure needs a spread under half of it
    'bias_limit': (0.001, 1.0),
    'assist': (0, 1),
    'assist_hz': (0.5, 50.0),
    'assist_deadband_deg': (0.0, 30.0),
    'assist_k': (0.0, 50.0),
    'assist_ki': (0.0, 50.0),
    'assist_k_rate': (0.0, 50.0),
    'assist_ki_rate': (0.0, 50.0),
    'assist_w_max': (0.0, 5.0),
    'assist_settle': (0.0, 5.0),
    'assist_min_speed': (0.0, 1.0),
    'path_speed': (0.01, 1.0),  # divides the path length (timeout)
    'path_lookahead': (0.01, 5.0),
    'path_lookahead_time': (0.0, 10.0),
    'path_w_max': (0.05, 5.0),
    'path_pivot_deg': (1.0, 180.0),
    'path_pivot_exit_deg': (0.5, 180.0),
    'path_goal_tol': (0.001, 1.0),
    'path_corner_deg': (0.0, 180.0),
    'gimbal_speed': (0.0, 720.0),
    'gimbal_hz': (0.5, 50.0),
    'gimbal_timeout': (0.05, 10.0),
    'gimbal_spd': (0, 10000),
    'gimbal_acc': (0, 1000),
    'pan_min': (-180.0, 180.0),
    'pan_max': (-180.0, 180.0),
    'tilt_min': (-90.0, 90.0),
    'tilt_max': (-90.0, 90.0),
    'pan_sign': (-1, 1),
    'tilt_sign': (-1, 1),
}

# Limits on the numbers of a command (HTTP API). Generous: a 5 m move across a hall or two
# full turns are legitimate; NaN, infinity, a unit error (mm for m) or a typo are not.
MAX_MOVE_M = 10.0  # one measured move
MAX_TURN_DEG = 720.0  # one measured turn
MAX_PATH_COORD_M = 50.0  # |x| and |y| of a waypoint, local or odometry frame
MAX_PATH_POINTS = 1000
MAX_PATH_LENGTH_M = 50.0  # total length of the waypoints
MAX_MANUAL_V = 1.0  # m/s, manual speed (the rover tops out well below)
MAX_MANUAL_W = 6.0  # rad/s, manual rotation


class NotReadyError(RuntimeError):
    """The rover cannot take this command now (e.g. the gyro bias is not measured yet)."""


def finite(value, name: str, limit: float) -> float:
    """`value` as a float, refusing booleans, NaN, infinities and |value| > limit."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{name} must be a number, not {value!r}')
    x = float(value)
    if not (math.isfinite(x) and abs(x) <= limit):
        raise ValueError(f'{name} must be a finite number within +-{limit}, not {value!r}')
    return x


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------------------
# Odometry
# --------------------------------------------------------------------------------------


class TiltMonitor:
    """Roll and pitch of the rover, their rates and angular accelerations, from the IMU.

    The accelerometer alone gives the tilt only when the rover does not accelerate; the gyro
    alone drifts. A complementary filter blends them: the gyro rates (gx roll, gy pitch,
    16.4 raw per deg/s like gz) follow fast changes, the accelerometer pulls the angle back
    over `tilt_filter_tau`. While the measured acceleration is far from 1 g (bumps, braking)
    the accelerometer is trusted much less.

    Conventions: roll + = left side up, pitch + = nose up (from the accelerometer formulas
    below). Whether the board's gyro axes agree is checked live (`sign_check`), and fixed
    with imu_roll_sign / imu_pitch_sign.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.ready = False
        self.roll = self.pitch = 0.0  # deg, filtered
        self.roll_acc_deg = self.pitch_acc_deg = 0.0  # deg, accelerometer only
        self.roll_rate = self.pitch_rate = 0.0  # deg/s
        self.roll_accel = self.pitch_accel = 0.0  # deg/s^2
        self.bias = [0.0, 0.0]  # gyro bias, deg/s (roll, pitch)
        self.g_ref: float | None = None  # accelerometer norm at rest (raw)
        self.peaks = {'roll': 0.0, 'pitch': 0.0, 'roll_rate': 0.0, 'pitch_rate': 0.0}
        self._t: float | None = None
        self._prev_acc_angles: tuple[float, float] | None = None
        self._agree = [[0, 0], [0, 0]]  # per axis: [samples where signs agree, disagree]

    def reset_peaks(self) -> None:
        self.peaks = dict.fromkeys(self.peaks, 0.0)

    def update(self, ax, ay, az, gx, gy, t: float, still: bool) -> None:
        c = self.cfg
        dt = 0.0 if self._t is None else clamp(t - self._t, 0.0, 0.2)
        self._t = t
        roll_a = math.degrees(math.atan2(ay, az))
        pitch_a = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
        dps = math.degrees(c.gyro_scale)  # raw -> deg/s
        rates = [gx * dps * c.imu_roll_sign, gy * dps * c.imu_pitch_sign]
        norm = math.sqrt(ax * ax + ay * ay + az * az)

        if still:  # learn the gyro biases and the 1 g norm while standing still
            for i in (0, 1):
                if abs(rates[i] - self.bias[i]) < 3.0:
                    self.bias[i] += 0.02 * (rates[i] - self.bias[i])
            self.g_ref = norm if self.g_ref is None else self.g_ref + 0.02 * (norm - self.g_ref)
        rr, pr = rates[0] - self.bias[0], rates[1] - self.bias[1]

        if not self.ready:
            self.roll, self.pitch = roll_a, pitch_a
            self.ready = True
        elif dt > 0:
            k = c.tilt_filter_tau / (c.tilt_filter_tau + dt)
            if self.g_ref and abs(norm / self.g_ref - 1.0) > 0.15:
                k = 0.995  # bump or hard braking: the accelerometer is not the gravity now
            self.roll = k * (self.roll + rr * dt) + (1 - k) * roll_a
            self.pitch = k * (self.pitch + pr * dt) + (1 - k) * pitch_a
            # angular accelerations from the (smoothed) rates
            new_rr = 0.5 * self.roll_rate + 0.5 * rr
            new_pr = 0.5 * self.pitch_rate + 0.5 * pr
            self.roll_accel = 0.5 * self.roll_accel + 0.5 * (new_rr - self.roll_rate) / dt
            self.pitch_accel = 0.5 * self.pitch_accel + 0.5 * (new_pr - self.pitch_rate) / dt
            self.roll_rate, self.pitch_rate = new_rr, new_pr
            # do the gyro rates agree with the accelerometer angles' changes?
            if self._prev_acc_angles is not None:
                d = (roll_a - self._prev_acc_angles[0], pitch_a - self._prev_acc_angles[1])
                for i, (rate, change) in enumerate(zip((rr, pr), d, strict=False)):
                    if abs(rate) > 10 and abs(change) > 0.2:
                        self._agree[i][0 if rate * change > 0 else 1] += 1
        self._prev_acc_angles = (roll_a, pitch_a)
        self.roll_acc_deg, self.pitch_acc_deg = roll_a, pitch_a
        p = self.peaks
        p['roll'] = max(p['roll'], abs(self.roll))
        p['pitch'] = max(p['pitch'], abs(self.pitch))
        p['roll_rate'] = max(p['roll_rate'], abs(self.roll_rate))
        p['pitch_rate'] = max(p['pitch_rate'], abs(self.pitch_rate))

    def sign_check(self) -> dict:
        """Per axis: 'ok', 'wrong' (flip imu_*_sign) or None (not enough tilting seen)."""
        out = {}
        for i, name in enumerate(('roll', 'pitch')):
            agree, disagree = self._agree[i]
            if agree + disagree < 10:
                out[name] = None
            else:
                out[name] = 'ok' if agree >= disagree else 'wrong'
        return out

    def danger(self) -> str | None:
        """Why the rover must stop now, or None."""
        c = self.cfg
        if abs(self.roll) >= c.roll_stop_deg:
            return f'side tilt {self.roll:+.0f} deg'
        if abs(self.pitch) >= c.pitch_stop_deg:
            return f'{"nose up" if self.pitch > 0 else "nose down"} {self.pitch:+.0f} deg'
        for name, angle, rate, acc in (
            ('side', self.roll, self.roll_rate, self.roll_accel),
            ('front/back', self.pitch, self.pitch_rate, self.pitch_accel),
        ):
            if abs(angle) > c.tip_min_deg:
                away = 1.0 if angle > 0 else -1.0  # direction away from level
                if rate * away > c.tip_rate_dps:
                    return f'tipping {name}: {angle:+.0f} deg at {rate:+.0f} deg/s'
                if acc * away > c.tip_acc_dps2 and rate * away > 0:
                    return f'tipping {name}: {angle:+.0f} deg, accelerating {acc:+.0f} deg/s2'
        return None

    def caution_scale(self) -> float:
        """Speed factor between the warn and stop angles (1 = no limit)."""
        c = self.cfg
        worst = 0.0
        for angle, warn, stop in (
            (self.roll, c.roll_warn_deg, c.roll_stop_deg),
            (self.pitch, c.pitch_warn_deg, c.pitch_stop_deg),
        ):
            if stop > warn:
                worst = max(worst, clamp((abs(angle) - warn) / (stop - warn), 0.0, 1.0))
        return 1.0 - worst * (1.0 - c.tilt_min_scale)

    def snapshot(self) -> dict:
        r = lambda x: round(x, 2)  # noqa: E731
        return {
            'ready': self.ready,
            'roll': r(self.roll),
            'pitch': r(self.pitch),
            'roll_rate': r(self.roll_rate),
            'pitch_rate': r(self.pitch_rate),
            'roll_accel': r(self.roll_accel),
            'pitch_accel': r(self.pitch_accel),
            'roll_from_acc': r(self.roll_acc_deg),
            'pitch_from_acc': r(self.pitch_acc_deg),
            'peaks': {k: r(v) for k, v in self.peaks.items()},
            'sign_check': self.sign_check(),
        }


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
        self.tilt = TiltMonitor(cfg)
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

    def on_imu(self, ax, ay, az, gx, gy, t: float) -> None:
        self.tilt.update(ax, ay, az, gx, gy, t, still=self._still_since is not None)

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

    def feedback_age(self, now: float) -> float | None:
        """s since the last wheel feedback (T:1001), None before the first one."""
        return None if self._last_wheels is None else now - self._last_wheels

    def on_gyro(self, gz_raw: float, t: float) -> None:
        self.gz_raw = gz_raw
        gz = gz_raw * self.cfg.gyro_scale * self.cfg.yaw_sign
        dt = 0.0 if self._last_gyro is None else clamp(t - self._last_gyro, 0.0, 0.2)
        self._last_gyro = t

        if not self.bias_ready:
            self._calibrate_bias(gz, t)
            return

        # Slowly re-learn the bias whenever the rover has been still for a while, but only
        # from readings that look like noise: wheels still does not mean not rotating.
        still = self._still_since is not None and t - self._still_since > 0.5
        if still and abs(gz - self.bias) < self.cfg.bias_max_rate:
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
            n = len(self._bias_samples)
            mean = sum(self._bias_samples) / n
            spread = math.sqrt(sum((g - mean) ** 2 for g in self._bias_samples) / n)
            self._bias_samples.clear()
            self._bias_start = None
            if spread > self.cfg.bias_max_rate / 2 or abs(mean) > self.cfg.bias_limit:
                return  # turned during the measure: start over
            self.bias = mean
            self.bias_ready = True

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

    def __post_init__(self) -> None:
        # Every step goes through here, whatever built it: no NaN or absurd goal can run.
        if self.kind == 'move':
            self.value = finite(self.value, 'distance', MAX_MOVE_M)
        elif self.kind == 'turn':
            self.value = finite(self.value, 'angle', MAX_TURN_DEG)
        elif self.kind == 'path':
            self.value = finite(self.value, 'path length', MAX_PATH_LENGTH_M)
        else:
            raise ValueError(f'unknown step kind {self.kind!r}')

    @classmethod
    def path(cls, points, frame: str = 'local', speed: float | None = None) -> Step:
        if not isinstance(points, list | tuple):
            raise ValueError('path points must be a list of [x, y]')
        if len(points) > MAX_PATH_POINTS:
            raise ValueError(f'a path has at most {MAX_PATH_POINTS} points')
        pts = []
        for p in points:
            if not isinstance(p, list | tuple) or len(p) < 2:
                raise ValueError(f'a path point is [x, y], not {p!r}')
            pts.append([finite(p[0], 'x', MAX_PATH_COORD_M), finite(p[1], 'y', MAX_PATH_COORD_M)])
        if not pts:
            raise ValueError('a path needs at least one point')
        if frame not in ('local', 'odom'):
            raise ValueError(f'path frame must be "local" or "odom", not {frame!r}')
        if speed is not None:
            speed = finite(speed, 'path speed', 1.0)
            if speed <= 0.0:
                raise ValueError('path speed must be in ]0, 1] m/s')
        start = [[0.0, 0.0]] if frame == 'local' else []
        chain = start + pts
        length = sum(math.dist(a, b) for a, b in zip(chain, chain[1:], strict=False))
        if length > MAX_PATH_LENGTH_M:
            raise ValueError(f'a path is at most {MAX_PATH_LENGTH_M} m long, not {length:.1f} m')
        return cls('path', round(length, 3), pts, frame, speed)

    @classmethod
    def parse(cls, d: dict) -> Step:
        if not isinstance(d, dict):
            raise ValueError(f'a step is a JSON object, not {d!r}')
        if 'move' in d:
            return cls('move', d['move'])
        if 'turn' in d:
            return cls('turn', d['turn'])
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

    Without wheel feedback the odometry freezes on its last speeds, so the rover would drive
    blind until a timeout: after FEEDBACK_TIMEOUT s without feedback, whatever runs is
    stopped and dropped (`fail`), and nothing restarts by itself when the feedback is back.
    """

    FEEDBACK_TIMEOUT = 0.5  # s; the firmware sends T:1001 every 50 ms

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
        self.fault: str | None = None  # why the controller last stopped by itself
        self.guard_state = 'off'  # off | ok | caution | backoff | hold
        self.guard_reason: str | None = None
        self._guard: dict = {'blocked_dir': 0.0}
        self._last_v = (0.0, -1e9)  # last non-zero speed command and when

    # -- commands -------------------------------------------------------------------

    def manual(self, v: float, w: float, t: float) -> None:
        """Drive at (v, w) until the next command or `manual_timeout`. Cancels any steps."""
        v = finite(v, 'v', MAX_MANUAL_V)
        w = finite(w, 'omega', MAX_MANUAL_W)
        if self.mode not in ('manual', 'idle'):
            self._abort('cancelled by manual control', t)
        self.queue.clear()  # taking over by hand: the rest of a sequence must not resume
        self._manual = (v, w)
        self._manual_t = t
        self.mode = 'manual' if (v or w) else 'idle'
        if v or w:
            self.fault = None

    def run(self, steps: list[Step], t: float) -> None:
        """Replace whatever runs with this list of steps.

        Raises NotReadyError before the gyro bias is measured: the encoder heading cannot hold a
        turn (it oscillates until the timeout) and moving would restart the measurement.
        """
        if steps and not self.odom.bias_ready:
            raise NotReadyError('gyro not calibrated yet: keep the rover still until "Gyro OK"')
        if self.mode in ('move', 'turn', 'path', 'settle'):
            self._abort('replaced by a new command', t)
        self.queue = deque(steps)
        self.current = None
        self.mode = 'idle'
        if steps:
            self.fault = None

    def stop(self, t: float, reason: str = 'stopped') -> None:
        try:
            self._abort(reason, t)
        except Exception:  # noqa: BLE001 - a broken step must never prevent a stop
            self._task = {}
        self.current = None
        self._pending_result = None
        self.queue.clear()
        self._manual = (0.0, 0.0)
        self.mode = 'idle'
        self.v_cmd = self.w_cmd = 0.0
        if self.guard_state == 'backoff':  # the tilt guard's back-off is a motion too
            self.guard_state, self._guard['hold_until'] = 'hold', t + 1.0

    def busy(self) -> bool:
        """Something drives the rover: a step, the queue, manual driving or a tilt back-off."""
        return self.mode != 'idle' or bool(self.queue) or self.guard_state == 'backoff'

    def fail(self, reason: str, t: float) -> None:
        """Safety stop decided here (lost feedback or link, error): drop everything, say why."""
        if not self.busy():
            return
        self.stop(t, reason)
        self.fault = reason

    # -- control loop ---------------------------------------------------------------

    def step(self, t: float) -> tuple[float, float]:
        dt = 0.05 if self._t_last is None else clamp(t - self._t_last, 0.0, 0.2)
        self._t_last = t

        if self.busy():
            age = self.odom.feedback_age(t)
            if age is None or age > self.FEEDBACK_TIMEOUT:
                self.fail('no ESP32 feedback', t)

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

        if abs(v) > 0.01:
            self._last_v = (v, t)
        v, w = self._tilt_guard(v, w, t)
        self.v_cmd, self.w_cmd = v, w
        return v, w

    def _start(self, s: Step, t: float) -> None:
        # Everything is built before the mode changes: if it raises, the controller stays idle
        # instead of a 'path' mode without a follower (which broke /stop and /state).
        o = self.odom
        task = {
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
            task['follower'] = follower
            task['speed'] = speed
            # Generous: pivots and slowing down near the end take time too.
            task['timeout'] = max(self.cfg.task_timeout, 3.0 * follower.length / speed + 10.0)
        self._task = task
        self.current = s
        self.mode = s.kind

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
        w = direction * clamp(c.k_turn * remaining, c.w_min, c.w_max)
        if self._not_rotating(t, 0.0, w):
            return 0.0, 0.0
        return 0.0, w

    def _tilt_guard(self, v: float, w: float, t: float) -> tuple[float, float]:
        """Slow down when tilted, stop and back off when about to tip over."""
        c, o, g = self.cfg, self.odom, self._guard
        tilt = o.tilt
        if not c.tilt_guard or not tilt.ready:
            self.guard_state = 'off'
            return v, w
        if self.guard_state == 'backoff':
            done = o.odometer - g['odo0'] >= c.tilt_backoff_dist
            if done or t - g['t0'] > c.tilt_backoff_time:
                self.guard_state, g['hold_until'] = 'hold', t + 1.0
                return 0.0, 0.0
            return g['dir'] * c.tilt_backoff_speed, 0.0
        if self.guard_state == 'hold' and t < g['hold_until']:
            return 0.0, 0.0

        reason = tilt.danger()
        if reason and g['blocked_dir'] == 0.0:  # trip: abort everything and back off
            last_v, last_t = self._last_v
            self.guard_reason = reason
            self._abort(f'tilt: {reason}', t)
            self.queue.clear()
            self._manual = (0.0, 0.0)
            self.mode = 'idle'
            self.fault = f'tilt: {reason}'
            # Never back off blind: without fresh wheel feedback the odometer cannot end it.
            age = o.feedback_age(t)
            fresh = age is not None and age <= self.FEEDBACK_TIMEOUT
            moving = fresh and t - last_t < 1.0
            g['blocked_dir'] = math.copysign(1.0, last_v) if moving else 0.0
            if moving:
                self.guard_state = 'backoff'
                g.update(dir=-g['blocked_dir'], odo0=o.odometer, t0=t)
                return g['dir'] * c.tilt_backoff_speed, 0.0
            self.guard_state, g['hold_until'] = 'hold', t + 1.0
            return 0.0, 0.0

        scale = tilt.caution_scale()
        if scale >= 1.0:
            g['blocked_dir'] = 0.0  # back on safe ground
            self.guard_state = 'ok'
            return v, w
        self.guard_state = 'caution'
        if g['blocked_dir'] and v * g['blocked_dir'] > 0 and reason:
            v = 0.0  # still too tilted: only the way back is allowed
        return v * scale, w * scale

    def _not_rotating(self, t: float, v: float, w: float) -> bool:
        """Pivoting on the spot, wheels turning, but the gyro sees no rotation: give up."""
        c, o, k = self.cfg, self.odom, self._task
        pivot = abs(v) < 0.02 and abs(w) >= 0.9 * c.w_min
        wheels_turn = abs(o.L) > 0.01 or abs(o.R) > 0.01
        if not (pivot and wheels_turn and o.gyro_alive(t)) or abs(o.yaw_rate) > c.no_rotation_rate:
            k['norot_since'] = None
            return False
        if k.get('norot_since') is None:
            k['norot_since'] = t
            return False
        if t - k['norot_since'] > c.no_rotation_time:
            self._done('no rotation (lifted or blocked?)', t)
            return True
        return False

    def _path(self, t: float, dt: float) -> tuple[float, float]:
        c, o, k = self.cfg, self.odom, self._task
        k['v_ramp'] = min(k['speed'], k['v_ramp'] + c.accel * dt)
        v, w, done = k['follower'].update(o.x, o.y, o.heading, o.v, max(k['v_ramp'], c.v_min), c)
        if done:
            return self._done('done', t)
        if self._failed(t) or self._not_rotating(t, v, w):
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
            # The step ended (status 'done'), but a stop, a manual command, another client's
            # step or a fault came before the rover had settled: log that reason, not 'done'.
            if self._pending_result and self._pending_result.get('status') == 'done':
                self._pending_result['status'] = reason
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
            'guard': {
                'state': self.guard_state,
                'reason': self.guard_reason,
                'blocked_dir': self._guard['blocked_dir'],
            },
            'path': follower.snapshot() if follower else None,
            'history': list(self.history),
            'fault': self.fault,
        }


# --------------------------------------------------------------------------------------
# ESP32 link
# --------------------------------------------------------------------------------------


def find_esp32_port(usb_only: bool = False) -> str | None:
    """First USB serial port that looks like the driver board.

    usb_only: only the Linux /dev/ttyUSB* and /dev/ttyACM* ports (on the rover), never a
    guess among a laptop's COM ports.
    """
    for pattern in ('/dev/ttyUSB*', '/dev/ttyACM*'):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    if usb_only:
        return None
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
        self.handler_errors = 0  # messages that made on_message raise (kept running)
        self.last_handler_error: str | None = None
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
                # A bad message must not end this thread: without it the odometry freezes.
                try:
                    self._on_message(msg, t)
                except Exception as e:  # noqa: BLE001
                    self.handler_errors += 1
                    self.last_handler_error = f'{type(e).__name__}: {e}'[:200]

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


def feedback_number(msg: dict, key: str, default: float | None = None) -> float:
    """A finite number from an ESP32 message (null or NaN would poison the odometry)."""
    x = float(msg[key] if default is None else msg.get(key, default))
    if not math.isfinite(x):
        raise ValueError(f'{key} is not finite: {msg.get(key)!r}')
    return x


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


class Recorder:
    """20 Hz log of what matters for the tipping tests, downloadable as CSV."""

    COLUMNS = (
        't', 'mode', 'guard', 'v_cmd', 'w_cmd', 'L', 'R', 'roll', 'pitch', 'roll_rate',
        'pitch_rate', 'roll_accel', 'pitch_accel', 'roll_from_acc', 'pitch_from_acc',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'heading', 'x', 'y', 'battery_v',
    )  # fmt: skip
    MAX_ROWS = 20 * 60 * 20  # 20 minutes

    def __init__(self) -> None:
        self.on = False
        self.rows: list[tuple] = []
        self._t0 = 0.0

    def start(self, t: float) -> None:
        self.rows, self._t0, self.on = [], t, True

    def stop(self) -> None:
        self.on = False

    def add(self, t: float, rover: Rover) -> None:
        o, tl, ch, c = rover.odom, rover.odom.tilt, rover.chassis, rover.ctrl
        self.rows.append((
            round(t - self._t0, 3), c.mode, c.guard_state, round(c.v_cmd, 3), round(c.w_cmd, 3),
            o.L, o.R, round(tl.roll, 2), round(tl.pitch, 2), round(tl.roll_rate, 2),
            round(tl.pitch_rate, 2), round(tl.roll_accel, 1), round(tl.pitch_accel, 1),
            round(tl.roll_acc_deg, 2), round(tl.pitch_acc_deg, 2),
            ch.get('ax'), ch.get('ay'), ch.get('az'), ch.get('gx'), ch.get('gy'), ch.get('gz'),
            round(math.degrees(o.heading), 2), round(o.x, 3), round(o.y, 3),
            battery_volts(ch.get('v')),
        ))  # fmt: skip
        if len(self.rows) >= self.MAX_ROWS:
            self.on = False

    def csv(self) -> str:
        lines = [','.join(self.COLUMNS)]
        lines += [','.join('' if v is None else str(v) for v in row) for row in self.rows]
        return '\n'.join(lines) + '\n'


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
        self.recorder = Recorder()
        self._was_connected = False
        self.loop_errors = 0
        self.last_loop_error: str | None = None
        self.stop_count = 0  # stop() calls (POST /stop, /cmd {"T":0}) since this Rover started
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
        try:
            with self.lock:
                # Counted even when the rover is idle (that stop leaves no trace in history):
                # a mission client compares it with the stops it sent itself.
                self.stop_count += 1
                self.ctrl.stop(time.monotonic())
        finally:
            self.link.send({'T': 13, 'X': 0, 'Z': 0})

    def reset_odometry(self) -> None:
        # Stop first: a step in the 'odom' frame aims at absolute points of the old origin and
        # would drive the whole way back to them from the new one.
        try:
            with self.lock:
                self.ctrl.stop(time.monotonic(), 'odometry reset')
                self.odom.reset()
        finally:
            self.link.send({'T': 13, 'X': 0, 'Z': 0})

    def recalibrate_gyro(self) -> None:
        # Stop first, as for reset_odometry: the heading restarts and a running step would chase
        # a goal in the old one (and moving would spoil the bias measurement).
        try:
            with self.lock:
                self.ctrl.stop(time.monotonic(), 'gyro recalibration')
                self.odom.recalibrate_gyro()
        finally:
            self.link.send({'T': 13, 'X': 0, 'Z': 0})

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
                left = feedback_number(msg, 'L', 0.0)
                right = feedback_number(msg, 'R', 0.0)
                self.chassis = msg
                self.odom.on_wheels(left, right, t)
                if 'gz' in msg:
                    self._gyro_in_feedback = True
                    self.odom.on_gyro(feedback_number(msg, 'gz'), t)
                if 'odl' in msg and 'odr' in msg:
                    self.odom.on_counters(feedback_number(msg, 'odl'), feedback_number(msg, 'odr'))
                if all(k in msg for k in ('ax', 'ay', 'az', 'gx', 'gy')):
                    imu = [feedback_number(msg, k) for k in ('ax', 'ay', 'az', 'gx', 'gy')]
                    self.odom.on_imu(*imu, t)
            elif kind == 1002:  # other firmwares: gyro only on request (T:126)
                self.imu = msg
                if 'gz' in msg and not self._gyro_in_feedback:
                    self.odom.on_gyro(feedback_number(msg, 'gz'), t)

    def _loop(self) -> None:
        period = 1.0 / self.LOOP_HZ
        nxt = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            connected = self.link.connected
            try:
                with self.lock:
                    if not connected:  # commands are lost: never resume a step afterwards
                        self.ctrl.fail('ESP32 link lost', t)
                    v, w = self.ctrl.step(t)
                    self.odom.commanded = abs(v) > 1e-4 or abs(w) > 1e-4
                    if connected and not self._was_connected:
                        self.gimbal.resend()  # the ESP32 (re)booted: put the camera back
                    gimbal_cmd = self.gimbal.step(t, period) if connected else None
                    if self.recorder.on:
                        self.recorder.add(t, self)
            except Exception as e:  # noqa: BLE001 - this loop must outlive any bug: /stop
                v, w, gimbal_cmd = 0.0, 0.0, None
                self._loop_error(e, t)
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

    def _loop_error(self, e: Exception, t: float) -> None:
        """Stop whatever ran, keep the error for /state, and keep the loop alive."""
        self.loop_errors += 1
        self.last_loop_error = f'{type(e).__name__}: {e}'[:200]
        if self.loop_errors == 1 or self.loop_errors % 100 == 0:  # a stuck error: not 20/s
            traceback.print_exc()
        with self.lock:
            self.ctrl.fail(f'controller error: {self.last_loop_error}', t)
            self.ctrl.fault = f'controller error: {self.last_loop_error}'

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
                    'handler_errors': link.handler_errors,
                    'last_handler_error': link.last_handler_error,
                },
                'loop_errors': self.loop_errors,
                'last_loop_error': self.last_loop_error,
                'stop_count': self.stop_count,
                'battery_v': battery_volts(self.chassis.get('v')),
                **(
                    {'roll_deg': self.odom.tilt.roll, 'pitch_deg': self.odom.tilt.pitch}
                    if self.odom.tilt.ready
                    else tilt_deg(self.chassis)
                ),
                'tilt': self.odom.tilt.snapshot(),
                'recording': {'on': self.recorder.on, 'rows': len(self.recorder.rows)},
                'counters_cm': self.odom.counters,
                'gimbal': self.gimbal.snapshot(),
                'odom': self.odom.snapshot(now),
                'ctrl': self.ctrl.snapshot(),
            }
