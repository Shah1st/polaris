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
    Gimbal,
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
                self.odom.on_imu(m['ax'], m['ay'], m['az'], m['gx'], m['gy'], self.t)

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

    def drive(self, v: float, w: float, seconds: float) -> None:
        """Hold a manual command, repeated every 0.1 s like the web page does."""
        end = self.t + seconds
        while self.t < end:
            self.ctrl.manual(v, w, self.t)
            self.run(0.1)


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


# -- gyro steering assist ---------------------------------------------------------------


def calibrated(assist: bool, **board) -> Sim:
    s = Sim(Config(assist=int(assist)), **board)
    s.run(2.5)
    assert s.odom.bias_ready
    return s


def test_assist_keeps_straight_despite_drift() -> None:
    """A slope that turns the rover by itself at 0.15 rad/s (8.6 deg/s)."""
    free = calibrated(False, yaw_drift=0.15)
    free.drive(0.2, 0.0, 4.0)
    helped = calibrated(True, yaw_drift=0.15)
    helped.drive(0.2, 0.0, 4.0)
    assert abs(math.degrees(free.board.yaw)) > 25
    assert abs(math.degrees(helped.board.yaw)) < 3
    assert helped.ctrl.assist.mode == 'hold'
    assert abs(helped.board.y) < 0.05  # stayed on the line it started on


def test_assist_does_nothing_without_drift() -> None:
    s = calibrated(True)
    s.drive(0.2, 0.0, 3.0)
    assert abs(math.degrees(s.board.yaw)) < 1.5
    assert abs(s.ctrl.assist.w_corr) < 0.1


def test_assist_makes_turns_follow_the_input() -> None:
    """The rover skids: it turns at 75 % of the asked rate without help."""

    def mean_rate(sim: Sim) -> float:
        sim.drive(0.0, 1.0, 1.0)  # get going
        yaw0, t0 = sim.board.yaw, sim.t
        sim.drive(0.0, 1.0, 2.0)
        return (sim.board.yaw - yaw0) / (sim.t - t0)

    assert mean_rate(calibrated(False)) == pytest.approx(0.75, abs=0.05)
    assert mean_rate(calibrated(True)) == pytest.approx(1.0, abs=0.08)


def test_assist_holds_the_heading_reached_after_a_turn() -> None:
    s = calibrated(True, yaw_drift=0.1)
    s.drive(0.15, 1.0, 1.2)
    s.drive(0.15, 0.0, 0.4)  # the turn input ended: settle, then hold
    held = s.ctrl.assist.ref
    assert held is not None
    yaw_at_hold = s.board.yaw
    s.drive(0.15, 0.0, 3.0)
    assert math.degrees(abs(s.board.yaw - yaw_at_hold)) < 3


def test_assist_off_when_disabled_or_stopped() -> None:
    s = calibrated(True)
    s.drive(0.0, 0.0, 0.5)
    assert s.ctrl.assist.mode == 'idle'
    s.cfg.assist = 0
    s.drive(0.2, 0.0, 0.5)
    assert s.ctrl.assist.mode == 'off' and s.ctrl.w_cmd == 0.0


# -- camera pan-tilt ----------------------------------------------------------------------


def test_gimbal_moves_while_held_and_respects_limits() -> None:
    cfg = Config(gimbal_speed=90.0, gimbal_hz=10.0)
    g = Gimbal(cfg)
    sent = []
    t = 0.0
    while t < 1.0:  # pan right, tilt up, key repeated every 0.1 s
        if round(t * 100) % 10 == 0:
            g.set_rate(1, 1, t)
        cmd = g.step(t, 0.05)
        if cmd:
            sent.append((t, cmd))
        t = round(t + 0.05, 3)
    assert g.pan == pytest.approx(90, abs=5) and g.tilt == pytest.approx(90, abs=5)
    assert all(c['T'] == 133 for _, c in sent)
    assert len(sent) <= 11  # at most gimbal_hz commands per second
    # The key is released (rate 0 sent at once): the camera stops, nothing more is sent.
    g.set_rate(0, 0, t)
    pan = g.pan
    for _ in range(5):
        t += 0.05
        cmd = g.step(t, 0.05)
    assert cmd is None and g.pan == pan
    # Key repeats lost (Wi-Fi): the camera stops after gimbal_timeout, not before.
    g.set_rate(1, 0, t)
    t0 = t
    while t - t0 < cfg.gimbal_timeout - 0.1:
        t += 0.05
        g.step(t, 0.05)
    assert g.pan > pan
    for _ in range(10):
        t += 0.05
        g.step(t, 0.05)
    stopped = g.pan
    t += 0.5
    g.step(t, 0.5)
    assert g.pan == stopped
    g.set_angles(500, -100)
    assert (g.pan, g.tilt) == (cfg.pan_max, cfg.tilt_min)


def test_gimbal_sign_and_resend() -> None:
    g = Gimbal(Config(pan_sign=-1))
    g.set_angles(30, 10)
    cmd = g.step(0.0, 0.05)
    assert cmd['X'] == -30 and cmd['Y'] == 10
    assert g.step(1.0, 0.05) is None  # unchanged: not sent again
    g.resend()
    assert g.step(2.0, 0.05)['X'] == -30


def test_rover_sends_gimbal_commands_to_the_board() -> None:
    import time

    board = FakeEsp32()
    rover = Rover(Config(), lambda: FakeSerial(board, boot_time=0.3))
    rover.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not rover.link.connected:
            time.sleep(0.05)
        rover.gimbal_set(-45, 20)
        time.sleep(0.4)
        assert (board.pan, board.tilt) == (-45, 20)
    finally:
        rover.close()


# -- path following -------------------------------------------------------------------------


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    n2 = dx * dx + dy * dy
    t = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def follow(points, drift: float = 0.0, limit: float = 60.0, **step) -> tuple[Sim, float, float]:
    """Follow a local path; returns (sim, true max deviation, true end miss) in metres."""
    s = Sim(Config(dist_scale=0.98), yaw_drift=drift)  # dist_scale = the board's ground_ratio
    s.run(2.5)
    poly = [(0.0, 0.0)] + [tuple(p) for p in points]
    s.start(Step.path(points, **step))
    worst = 0.0
    end = s.t + limit
    while s.t < end:
        s.run(0.05)
        here = (s.board.x, s.board.y)
        worst = max(
            worst, min(_seg_dist(here, a, b) for a, b in zip(poly, poly[1:], strict=False))
        )
        if s.ctrl.mode == 'idle' and not s.ctrl.queue:
            break
    return s, worst, math.dist((s.board.x, s.board.y), poly[-1])


CIRCLE = [
    [0.5 * math.sin(k * math.pi / 18), 0.5 - 0.5 * math.cos(k * math.pi / 18)]
    for k in range(1, 37)
]
SQUARE = [[0.6, 0.0], [0.6, 0.6], [0.0, 0.6], [0.0, 0.0]]


@pytest.mark.parametrize(
    'name, points, max_dev',
    [
        ('straight', [[1.0, 0.0]], 0.02),
        ('L shape', [[1.0, 0.0], [1.0, 0.5]], 0.035),
        ('square', SQUARE, 0.035),
        ('circle', CIRCLE, 0.035),
        ('behind', [[-0.5, 0.0]], 0.035),
    ],
)
def test_path_is_followed_closely(name, points, max_dev) -> None:
    s, dev, miss = follow(points)
    result = s.ctrl.history[0]
    assert result['kind'] == 'path' and result['status'] == 'done'
    assert dev < max_dev, f'{name}: {dev * 100:.1f} cm off the path'
    assert miss < 0.035, f'{name}: ended {miss * 100:.1f} cm from the end point'


def test_path_holds_on_a_slope() -> None:
    s, dev, miss = follow(SQUARE, drift=0.1)
    assert s.ctrl.history[0]['status'] == 'done'
    assert dev < 0.04 and miss < 0.04


def test_path_stops_and_pivots_on_corners_only() -> None:
    from rover_core import PathFollower

    square = PathFollower([tuple(p) for p in SQUARE], (0.0, 0.0), corner_deg=45)
    assert len(square.corners) == 3  # the end point is not a corner
    circle = PathFollower([tuple(p) for p in CIRCLE], (0.0, 0.0), corner_deg=45)
    assert circle.corners == []  # 10 deg bends: followed as a curve
    assert PathFollower([(1.0, 0.0), (1.0, 1.0)], (0.0, 0.0), corner_deg=0).corners == []


def test_path_in_odometry_frame_and_in_a_sequence() -> None:
    s = Sim(Config(dist_scale=0.98))
    s.run(2.5)
    s.start(
        Step('turn', 90.0),
        Step.path([[0.0, 0.5], [0.5, 0.5]], frame='odom'),  # absolute: ignores the turn
        Step('move', 0.1),
    )
    s.run_until_idle(limit=60)
    kinds = [h['kind'] for h in reversed(s.ctrl.history)]
    assert kinds == ['turn', 'path', 'move']
    assert all(h['status'] == 'done' for h in s.ctrl.history)
    assert math.dist((s.board.x, s.board.y), (0.6, 0.5)) < 0.05


def test_path_speed_and_progress_snapshot() -> None:
    s = Sim(Config(dist_scale=0.98))
    s.run(2.5)
    s.start(Step.path([[2.0, 0.0]], speed=0.3))
    s.run(4.0)
    snap = s.ctrl.snapshot()
    assert snap['mode'] == 'path' and snap['path']['length'] == pytest.approx(2.0)
    assert 0.5 < snap['path']['done_m'] < 2.0
    assert max(s.ctrl.v_cmd, s.odom.v) > 0.25  # cruising faster than the 0.2 default


def test_path_can_be_stopped_or_replaced() -> None:
    s = Sim()
    s.run(2.5)
    s.start(Step.path([[2.0, 0.0]]))
    s.run(1.0)
    s.start(Step.path([[0.0, 0.5]]))  # a new trajectory replaces the running one
    assert s.ctrl.history[0]['status'] == 'replaced by a new command'
    s.run(1.0)
    s.ctrl.stop(s.t)
    assert s.ctrl.history[0]['status'] == 'stopped'
    s.run(0.5)
    assert abs(s.board.wl) < 0.01 and abs(s.board.wr) < 0.01


def test_path_aborts_when_the_rover_is_blocked() -> None:
    s = Sim()
    s.run(2.5)
    s.board.blocked = True
    s.start(Step.path([[1.0, 0.0]]))
    s.run_until_idle()
    assert s.ctrl.history[0]['status'] == 'stalled'


def test_path_step_parsing() -> None:
    step = Step.parse({'path': [[0.3, 0], [0.3, 0.4]], 'speed': 0.15})
    assert step.kind == 'path' and step.value == pytest.approx(0.7) and step.speed == 0.15
    assert Step.parse({'path': [[1, 1]], 'frame': 'odom'}).frame == 'odom'
    for bad in ({'path': []}, {'path': [[1, 0]], 'frame': 'map'}, {'path': [[1, 0]], 'speed': 3}):
        with pytest.raises(ValueError):
            Step.parse(bad)


# -- gyro bias vs hand rotations ------------------------------------------------------------


def test_turning_the_rover_by_hand_is_not_learned_as_bias(sim: Sim) -> None:
    bias = sim.odom.bias
    sim.board.hand_rate = 0.5  # rad/s, wheels still, for 2 s
    sim.run(2.0)
    sim.board.hand_rate = 0.0
    sim.run(1.0)
    assert sim.odom.bias == pytest.approx(bias, abs=0.002)
    assert sim.odom.heading == pytest.approx(sim.board.yaw, abs=0.05)
    assert sim.odom.heading == pytest.approx(1.0, abs=0.05)


def test_bias_calibration_waits_until_the_rover_is_still() -> None:
    s = Sim()
    s.board.hand_rate = 0.4  # carried and turned while the server starts
    s.run(3.0)
    assert not s.odom.bias_ready
    s.board.hand_rate = 0.0
    s.run(4.5)
    assert s.odom.bias_ready
    assert s.odom.bias == pytest.approx(s.board.gyro_bias_raw * s.cfg.gyro_scale, abs=0.003)


# -- rover lifted or stuck: a turn that does not turn -----------------------------------------


def test_turn_gives_up_when_the_rover_does_not_rotate(sim: Sim) -> None:
    sim.board.lifted = True  # on its stand: the wheels spin, the body does not turn
    sim.start(Step('turn', 90.0), Step('move', 0.5))
    t0 = sim.t
    sim.run_until_idle()
    h = sim.ctrl.history[0]
    assert h['status'] == 'no rotation (lifted or blocked?)'
    assert sim.t - t0 < 4.0  # not the 20 s timeout
    assert len(sim.ctrl.history) == 1  # the next step was dropped


def test_path_gives_up_when_the_pivot_does_not_rotate(sim: Sim) -> None:
    sim.board.lifted = True
    sim.start(Step.path([[0.0, 0.5]]))  # 90 deg to the left: starts with a pivot
    sim.run_until_idle()
    assert sim.ctrl.history[0]['status'] == 'no rotation (lifted or blocked?)'


def test_slow_real_turn_is_not_mistaken_for_no_rotation(sim: Sim) -> None:
    sim.board.turn_efficiency = 0.3  # very grippy floor: turns, but slowly
    sim.start(Step('turn', 90.0))
    sim.run_until_idle()
    assert sim.ctrl.history[0]['status'] == 'done'


# -- tilt: estimate, guard, recorder ------------------------------------------------------


def test_tilt_estimate_follows_a_slope(sim: Sim) -> None:
    sim.board.roll_target, sim.board.pitch_target = 12.0, -14.0
    sim.run(2.5)
    tilt = sim.odom.tilt
    assert tilt.roll == pytest.approx(12.0, abs=1.0)
    assert tilt.pitch == pytest.approx(-14.0, abs=1.0)
    assert tilt.sign_check() == {'roll': 'ok', 'pitch': 'ok'}
    assert tilt.peaks['roll'] == pytest.approx(12.0, abs=1.5)


def test_wrong_gyro_sign_is_reported() -> None:
    s = Sim(Config(imu_roll_sign=-1))
    s.run(2.5)
    s.board.roll_target = 15.0
    s.run(2.0)
    assert s.odom.tilt.sign_check()['roll'] == 'wrong'


def test_guard_slows_down_between_warn_and_stop(sim: Sim) -> None:
    sim.board.roll_target = 20.0  # halfway between warn 15 and stop 25
    sim.run(2.0)
    sim.drive(0.2, 0.0, 0.5)
    assert sim.ctrl.guard_state == 'caution'
    assert sim.ctrl.v_cmd == pytest.approx(0.2 * (1 - 0.5 * 0.7), abs=0.02)


def test_guard_stops_and_backs_off_when_the_side_tilt_gets_too_steep(sim: Sim) -> None:
    sim.drive(0.2, 0.0, 1.0)
    sim.board.roll_target = 32.0  # driving onto a side slope too steep
    x_before = None
    for _ in range(40):
        sim.drive(0.2, 0.0, 0.1)
        if sim.ctrl.guard_state == 'backoff' and x_before is None:
            x_before = sim.board.x
            assert sim.odom.tilt.roll >= 24.0
    assert x_before is not None, 'never backed off'
    assert sim.ctrl.guard_reason.startswith('side tilt')
    sim.run(3.0)
    assert sim.board.x == pytest.approx(x_before - 0.15, abs=0.05)  # reversed ~15 cm
    # Still on the steep slope: forward is refused, backward is allowed (slowly).
    sim.drive(0.2, 0.0, 0.5)
    assert sim.ctrl.v_cmd == 0.0
    sim.drive(-0.2, 0.0, 0.5)
    assert sim.ctrl.v_cmd < -0.03


def test_guard_catches_a_fast_tip_before_the_stop_angle(sim: Sim) -> None:
    sim.board.slope_rate = 70.0  # deg/s: a wheel drops into a hole
    sim.drive(0.2, 0.0, 0.5)
    sim.board.pitch_target = -20.0  # below the 25 deg stop angle
    for _ in range(10):
        sim.drive(0.2, 0.0, 0.1)
    assert sim.ctrl.guard_reason and sim.ctrl.guard_reason.startswith('tipping front/back')
    assert sim.ctrl.guard_state in ('backoff', 'hold')


def test_guard_aborts_a_path(sim: Sim) -> None:
    sim.start(Step.path([[2.0, 0.0]]))
    sim.run(1.5)
    sim.board.roll_target = -30.0
    sim.run(3.0)
    assert sim.ctrl.history[0]['status'].startswith('tilt: side tilt')
    assert sim.ctrl.mode == 'idle'


def test_guard_off_for_tipping_tests() -> None:
    s = Sim(Config(tilt_guard=0))
    s.run(2.5)
    s.board.roll_target = 30.0
    s.run(2.0)
    s.drive(0.2, 0.0, 0.5)
    assert s.ctrl.guard_state == 'off' and s.ctrl.v_cmd == pytest.approx(0.2)


def test_recorder_csv() -> None:
    from rover_core import Recorder

    board = FakeEsp32()
    rover = Rover(Config(), lambda: FakeSerial(board, boot_time=0.2))
    import time

    rover.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not rover.odom.tilt.ready:
            time.sleep(0.05)
        rover.recorder.start(time.monotonic())
        time.sleep(0.6)
        rover.recorder.stop()
        lines = rover.recorder.csv().splitlines()
        assert lines[0].split(',') == list(Recorder.COLUMNS)
        assert 8 <= len(lines) - 1 <= 15  # 20 Hz for 0.6 s
        assert len(lines[1].split(',')) == len(Recorder.COLUMNS)
    finally:
        rover.close()
