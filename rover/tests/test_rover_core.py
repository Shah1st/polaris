"""Controller and odometry against the simulated ESP32, in simulated time (fast)."""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from rover_core import (  # noqa: E402
    Config,
    Controller,
    Esp32Link,
    Odometry,
    Rover,
    Step,
    battery_volts,
    tilt_deg,
)

PHYSICS_DT = 0.005
CONTROL_DT = 0.05


class Sim:
    """Odometry + controller + fake board, stepped like the real 20 Hz loop."""

    def __init__(self, cfg: Config | None = None, **board) -> None:
        self.cfg = cfg or Config()
        self.board = FakeEsp32(**board)
        self.odom = Odometry(self.cfg)
        self.ctrl = Controller(self.cfg, self.odom)
        self.t = 0.0
        self._next_control = 0.0

    def feed(self, msgs: list[dict]) -> None:
        for m in msgs:
            if m['T'] == 1001:  # UGV firmware: everything comes in T:1001
                self.odom.on_wheels(m['L'], m['R'], self.t)
                self.odom.on_gyro(m['gz'], self.t)
                self.odom.on_counters(m['odl'], m['odr'])

    def run(self, seconds: float) -> None:
        end = self.t + seconds
        while self.t < end:
            if self.t >= self._next_control:
                self._next_control += CONTROL_DT
                v, w = self.ctrl.step(self.t)
                self.odom.commanded = abs(v) > 1e-4 or abs(w) > 1e-4
                self.feed(self.board.handle({'T': 13, 'X': v, 'Z': w}))
            self.t += PHYSICS_DT
            self.feed(self.board.advance(PHYSICS_DT))

    def run_until_idle(self, limit: float = 30.0) -> None:
        end = self.t + limit
        while self.t < end:
            self.run(0.1)
            if self.ctrl.mode == 'idle' and not self.ctrl.queue:
                return
        raise AssertionError(f'still {self.ctrl.mode} after {limit} s')

    def start(self, *steps: Step) -> None:
        self.ctrl.run(list(steps), self.t)


@pytest.fixture
def sim() -> Sim:
    s = Sim()
    s.run(2.5)  # still: gyro bias calibration
    assert s.odom.bias_ready
    return s


def test_gyro_bias_is_measured_while_still(sim: Sim) -> None:
    expected = sim.board.gyro_bias_raw * sim.cfg.gyro_scale
    assert sim.odom.bias == pytest.approx(expected, abs=0.003)


@pytest.mark.parametrize('distance', [0.5, 1.0, -0.3, 0.1])
def test_move_reaches_distance(sim: Sim, distance: float) -> None:
    sim.start(Step('move', distance))
    sim.run_until_idle()
    result = sim.ctrl.history[0]
    assert result['status'] == 'done'
    assert result['distance'] == pytest.approx(distance, abs=0.015)
    # The firmware counters (1 cm steps) agree.
    assert result['counter_distance'] == pytest.approx(distance, abs=0.025)
    # Goes straight: the true heading barely changes.
    assert abs(math.degrees(sim.board.yaw)) < 2.0


@pytest.mark.parametrize('angle', [90.0, -90.0, 180.0, 15.0, 360.0])
def test_turn_reaches_angle(sim: Sim, angle: float) -> None:
    sim.start(Step('turn', angle))
    sim.run_until_idle()
    result = sim.ctrl.history[0]
    assert result['status'] == 'done'
    # The true rotation, not only the estimated one, since the wheels slip in turns.
    assert math.degrees(sim.board.yaw) == pytest.approx(angle, abs=3.0)
    assert result['turned_deg'] == pytest.approx(angle, abs=3.0)


def test_dist_scale_corrects_ground_slip() -> None:
    cfg = Config(dist_scale=0.98)  # = board.ground_ratio, as found with a tape measure
    s = Sim(cfg)
    s.run(2.5)
    s.start(Step('move', 1.0))
    s.run_until_idle()
    assert s.board.x == pytest.approx(1.0, abs=0.015)


def test_square_comes_back_near_start(sim: Sim) -> None:
    steps = [Step('move', 0.5), Step('turn', 90.0)] * 4
    sim.start(*steps)
    sim.run_until_idle(limit=60.0)
    assert len(sim.ctrl.history) == 8
    assert all(h['status'] == 'done' for h in sim.ctrl.history)
    # Encoders over-read by 2 % (no dist_scale here), so allow a few cm.
    assert math.hypot(sim.board.x, sim.board.y) < 0.06


def test_manual_command_times_out(sim: Sim) -> None:
    sim.ctrl.manual(0.2, 0.0, sim.t)
    sim.run(0.3)
    assert sim.ctrl.mode == 'manual'
    assert sim.board.wl > 0.1
    sim.run(0.5)  # no new command: stops after manual_timeout
    assert sim.ctrl.mode == 'idle'
    sim.run(0.5)
    assert abs(sim.board.wl) < 0.01


def test_stop_cancels_steps(sim: Sim) -> None:
    sim.start(Step('move', 1.0), Step('turn', 90.0))
    sim.run(0.5)
    sim.ctrl.stop(sim.t)
    sim.run(1.0)
    assert sim.ctrl.mode == 'idle'
    assert not sim.ctrl.queue
    assert sim.ctrl.history[0]['status'] == 'stopped'
    assert abs(sim.board.wl) < 0.01


def test_manual_cancels_steps(sim: Sim) -> None:
    sim.start(Step('move', 1.0))
    sim.run(0.5)
    sim.ctrl.manual(0.0, 1.0, sim.t)
    assert sim.ctrl.history[0]['status'] == 'cancelled by manual control'
    assert sim.ctrl.mode == 'manual'


def test_stalled_wheels_abort_the_move(sim: Sim) -> None:
    sim.board.blocked = True
    sim.start(Step('move', 0.5), Step('turn', 90.0))
    sim.run_until_idle()
    assert sim.ctrl.history[0]['status'] == 'stalled'
    assert len(sim.ctrl.history) == 1  # the turn was dropped


def test_heading_falls_back_to_encoders_without_gyro() -> None:
    cfg = Config()
    odom = Odometry(cfg)
    for i in range(20):  # wheel feedback only, turning left at 1 rad/s
        odom.on_wheels(-0.0625, 0.0625, i * 0.05)
    assert odom.heading == pytest.approx(0.95, abs=0.01)
    assert odom.snapshot(1.0)['heading_source'] == 'encoders'


def test_yaw_sign_flips_the_gyro(sim: Sim) -> None:
    sim.cfg.yaw_sign = -1
    sim.odom.recalibrate_gyro()
    sim.run(2.5)
    sim.ctrl.manual(0.0, 1.5, sim.t)
    sim.run(0.4)
    assert sim.odom.heading_gyro < 0 < sim.board.yaw


def test_config_roundtrip(tmp_path) -> None:
    path = str(tmp_path / 'cfg.json')
    cfg = Config()
    assert cfg.update({'dist_scale': '0.97', 'yaw_sign': -1.0, 'unknown': 3}) == [
        'dist_scale',
        'yaw_sign',
    ]
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.dist_scale == 0.97
    assert loaded.yaw_sign == -1 and isinstance(loaded.yaw_sign, int)


def test_step_parse() -> None:
    assert Step.parse({'move': 0.5}) == Step('move', 0.5)
    assert Step.parse({'turn': '-90'}) == Step('turn', -90.0)
    with pytest.raises(ValueError):
        Step.parse({'jump': 1})


class ScriptedSerial:
    """Serial look-alike that replays lines, then fails like an unplugged cable."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)
        self.written: list[dict] = []

    def write(self, data: bytes) -> int:
        self.written.extend(json.loads(x) for x in data.splitlines())
        return len(data)

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        raise OSError('device disconnected')

    def close(self) -> None:
        pass


def test_link_parses_lines_and_sends_init() -> None:
    import threading

    ser = ScriptedSerial([b'{"T":1001,"L":0.1,"R":0.1,"v":12.1}\n', b'garbage\n', b'\n'])
    got: list[dict] = []
    done = threading.Event()
    opened = []

    def opener():
        if opened:
            done.set()
            raise OSError('gone')
        opened.append(1)
        return ser

    link = Esp32Link(opener, lambda m, t: got.append(m), wait_boot=False)
    link.start()
    assert done.wait(5.0)
    link.close()
    assert got == [{'T': 1001, 'L': 0.1, 'R': 0.1, 'v': 12.1}]
    assert link.bad_lines == 1
    assert ser.written[0] == {'T': 900, 'main': 1, 'module': 0}
    assert {'T': 131, 'cmd': 1} in ser.written
    assert not link.connected


def test_battery_and_tilt_conversions() -> None:
    assert battery_volts(1158) == pytest.approx(11.58)  # UGV firmware: centivolts
    assert battery_volts(11.6) == pytest.approx(11.6)
    assert battery_volts(None) is None
    flat = tilt_deg({'ax': 0, 'ay': 0, 'az': 8192})
    assert flat['roll_deg'] == pytest.approx(0) and flat['pitch_deg'] == pytest.approx(0)
    rolled = tilt_deg({'ax': 0, 'ay': 8192, 'az': 8192})
    assert rolled['roll_deg'] == pytest.approx(45)
    assert tilt_deg({'r': 1.5, 'p': -2})['pitch_deg'] == -2


def test_rover_waits_for_boot_then_drives() -> None:
    """Real-time run on the simulated serial port: boot text, init, gyro, a short move."""
    import time

    board = FakeEsp32()
    rover = Rover(Config(bias_calib_time=0.5), lambda: FakeSerial(board, boot_time=0.6))
    rover.start()
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not rover.odom.bias_ready:
            time.sleep(0.05)
        snap = rover.snapshot()
        assert snap['link']['connected'] and snap['link']['error'] is None
        assert snap['battery_v'] == pytest.approx(11.9, abs=0.05)
        assert snap['odom']['heading_source'] == 'gyro'
        rover.run([Step('move', 0.1)])
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not rover.ctrl.history:
            time.sleep(0.05)
        assert rover.ctrl.history[0]['status'] == 'done'
        assert board.x == pytest.approx(0.1 * board.ground_ratio, abs=0.02)
    finally:
        rover.close()
