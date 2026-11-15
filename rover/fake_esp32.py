"""Simulated ESP32 driver board, to try the server and the controller without the robot.

It speaks like the Waveshare "UGV" firmware measured on the robot (2026-11-14):
* T:1001 on its own at 20 Hz: L, R wheel speeds in m/s by steps of ~0.06 m/s, raw IMU
  (gz: 16.4 per deg/s, az: ~8192 per g), odl/odr wheel counters in cm, v in centivolts;
* T:126 answers T:1002 with zero angles (the fused attitude is not available);
* opening the port prints boot text and drops commands until "UGV started".

Rough skid-steer physics, chosen to show the real robot's defects:
* the ground distance is 2 % shorter than the encoders say (`ground_ratio`), which
  `dist_scale` corrects;
* the rover only turns 75 % as fast as the wheel speeds imply (`turn_efficiency`), which
  is why the heading comes from the gyro;
* the gyro has a bias and noise; very slow wheel speeds do not move the motors;
* `yaw_drift` (rad/s) turns the rover by itself while it moves, like a slope or uneven
  grip, to try the gyro steering assist.

T:133 moves the simulated camera pan-tilt (pan/tilt attributes, no angle feedback).
"""

from __future__ import annotations

import json
import math
import random
import threading
import time

GYRO_LSB_PER_DPS = 16.4
ACC_LSB_PER_G = 8192.0


class FakeEsp32:
    TRACK = 0.125  # m, same value as the firmware for the RaspRover
    HEARTBEAT = 3.0  # s, the firmware stops the motors without commands
    FEEDBACK_PERIOD = 0.05  # s
    SPEED_STEP = 0.05984  # m/s, resolution of the L/R speeds sent by the firmware

    def __init__(
        self,
        seed: int = 0,
        gyro_bias_raw: float = -3.4,
        gyro_noise_raw: float = 7.4,
        ground_ratio: float = 0.98,
        turn_efficiency: float = 0.75,
        wheel_tau: float = 0.12,
        deadband: float = 0.02,
        yaw_drift: float = 0.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.gyro_bias_raw = gyro_bias_raw
        self.gyro_noise_raw = gyro_noise_raw
        self.ground_ratio = ground_ratio
        self.turn_efficiency = turn_efficiency
        self.wheel_tau = wheel_tau
        self.deadband = deadband
        self.yaw_drift = yaw_drift
        self.pan = self.tilt = 0.0  # camera pan-tilt, deg (T:133)
        self.blocked = False  # True: wheels stuck (tests the stall detection)
        self.hand_rate = 0.0  # rad/s: someone turns the rover by hand, wheels still
        self.booted = True  # FakeSerial sets it False while "booting"

        self.x = self.y = self.yaw = 0.0  # true pose
        self.wl = self.wr = 0.0  # true wheel speeds, m/s
        self.odl = self.odr = 0.0  # wheel distances, m (what the counters count)
        self.yaw_rate = 0.0
        self.target = (0.0, 0.0)
        self.t = 0.0
        self.last_cmd = 0.0
        self.feedback_on = True
        self._next_feedback = 0.0
        self.battery = 11.9
        self.trail: list[tuple[float, float]] = []

    # -- protocol -------------------------------------------------------------------

    def handle(self, cmd: dict) -> list[dict]:
        """Apply one command. Returns the immediate replies."""
        if not self.booted:
            return []
        kind = cmd.get('T')
        if kind == 13:
            x, z = float(cmd.get('X', 0)), float(cmd.get('Z', 0))
            self.target = (x - z * self.TRACK / 2, x + z * self.TRACK / 2)
            self.last_cmd = self.t
        elif kind == 1:
            self.target = (float(cmd.get('L', 0)), float(cmd.get('R', 0)))
            self.last_cmd = self.t
        elif kind == 0:
            self.target = (0.0, 0.0)
        elif kind == 131:
            self.feedback_on = bool(cmd.get('cmd'))
        elif kind == 133:
            self.pan = max(-180.0, min(180.0, float(cmd.get('X', 0))))
            self.tilt = max(-30.0, min(90.0, float(cmd.get('Y', 0))))
        elif kind == 126:
            return [{'T': 1002, 'r': 0, 'p': 0, 'y': 0, 'q0': 0, 'q1': 0, 'q2': 0, 'q3': 0}]
        elif kind == 130:
            return [self._chassis()]
        return []

    def advance(self, dt: float) -> list[dict]:
        """Move the physics forward by `dt`. Returns the periodic feedback messages."""
        self.t += dt
        tl, tr = self.target
        if self.t - self.last_cmd > self.HEARTBEAT:
            tl = tr = 0.0
        if self.blocked:
            tl = tr = 0.0
        a = 1.0 - math.exp(-dt / self.wheel_tau)
        self.wl += ((tl if abs(tl) >= self.deadband else 0.0) - self.wl) * a
        self.wr += ((tr if abs(tr) >= self.deadband else 0.0) - self.wr) * a
        self.odl += self.wl * dt
        self.odr += self.wr * dt

        v = (self.wl + self.wr) / 2 * self.ground_ratio
        w_kin = (self.wr - self.wl) / self.TRACK
        # Skid steering: a slow pivot does not overcome the tyres' friction.
        if abs(v) < 0.02 and abs(w_kin) < 0.5:
            w_kin = 0.0
        self.yaw_rate = w_kin * self.turn_efficiency
        if abs(v) > 0.02:
            self.yaw_rate += self.yaw_drift
        self.yaw_rate += self.hand_rate
        self.yaw += self.yaw_rate * dt
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        last = self.trail[-1] if self.trail else None
        if last is None or math.hypot(self.x - last[0], self.y - last[1]) > 0.01:
            self.trail.append((self.x, self.y))
            del self.trail[:-2000]
        self.battery = max(10.5, self.battery - 1e-6 * dt)

        out = []
        if self.booted and self.feedback_on and self.t >= self._next_feedback:
            self._next_feedback = self.t + self.FEEDBACK_PERIOD
            out.append(self._chassis())
        return out

    def _speed(self, w: float) -> float:
        """Firmware speed reading: coarse steps, right on average (stochastic rounding)."""
        if abs(w) < 0.002:
            return 0.0
        q = self.SPEED_STEP
        return math.floor(w / q + self.rng.random()) * q

    def _chassis(self) -> dict:
        g = self.rng.gauss
        gz = math.degrees(self.yaw_rate) * GYRO_LSB_PER_DPS + self.gyro_bias_raw
        return {
            'T': 1001,
            'L': self._speed(self.wl),
            'R': self._speed(self.wr),
            'ax': round(g(-80, 60)),
            'ay': round(g(-70, 60)),
            'az': round(ACC_LSB_PER_G * 1.03 + g(0, 30)),
            'gx': round(g(-12, 8)),
            'gy': round(g(1, 4)),
            'gz': round(gz + g(0, self.gyro_noise_raw)),
            'mx': -660,
            'my': -90,
            'mz': 648,
            'odl': round(self.odl * 100),
            'odr': round(self.odr * 100),
            'v': round(self.battery * 100),
        }


class FakeSerial:
    """Serial-port look-alike around FakeEsp32, running the physics in real time."""

    BOOT_TEXT = (
        'ets Jul 29 2019 12:21:46',
        'rst:0x1 (POWERON_RESET),boot:0x12 (SPI_FAST_FLASH_BOOT)',
        'WiFi init.',
        'IMU Calibrating',
        'UGV started.',
    )

    def __init__(
        self, board: FakeEsp32 | None = None, rate_hz: float = 200.0, boot_time: float = 1.5
    ) -> None:
        self.board = board or FakeEsp32()
        self._rx: list[bytes] = []
        self._cond = threading.Condition()
        self._closed = False
        self._period = 1.0 / rate_hz
        self._buf = b''
        self.board.booted = False
        threading.Thread(target=self._boot, args=(boot_time,), daemon=True).start()
        threading.Thread(target=self._physics, name='fake-esp32', daemon=True).start()

    def _boot(self, boot_time: float) -> None:
        for line in self.BOOT_TEXT:
            time.sleep(boot_time / len(self.BOOT_TEXT))
            self._push_raw(line.encode() + b'\n')
        with self._cond:
            self.board.booted = True

    def _push_raw(self, data: bytes) -> None:
        with self._cond:
            self._rx.append(data)
            del self._rx[:-200]
            self._cond.notify_all()

    def _push(self, msgs: list[dict]) -> None:
        for m in msgs:
            self._push_raw(json.dumps(m, separators=(',', ':')).encode() + b'\n')

    def _physics(self) -> None:
        last = time.monotonic()
        while not self._closed:
            time.sleep(self._period)
            now = time.monotonic()
            with self._cond:
                msgs = self.board.advance(now - last)
            last = now
            self._push(msgs)

    def write(self, data: bytes) -> int:
        self._buf += data
        while b'\n' in self._buf:
            line, self._buf = self._buf.split(b'\n', 1)
            try:
                cmd = json.loads(line)
            except ValueError:
                continue
            with self._cond:
                replies = self.board.handle(cmd)
            self._push(replies)
        return len(data)

    def readline(self) -> bytes:
        with self._cond:
            if not self._rx:
                self._cond.wait(timeout=0.2)
            return self._rx.pop(0) if self._rx else b''

    def reset_input_buffer(self) -> None:
        with self._cond:
            self._rx.clear()

    def close(self) -> None:
        self._closed = True
