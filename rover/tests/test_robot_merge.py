"""Seams between the robot's code of 15 Nov (tilt guard, IMU, recorder) and the safety fixes."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_rover_core import Sim  # noqa: E402
from test_safety import ready_rover, wait_for  # noqa: E402

from fake_esp32 import FakeSerial  # noqa: E402
from rover_core import CONFIG_LIMITS, Config, Rover, Step  # noqa: E402

# Statuses that merging_map/rover_reel.py takes for an operator's stop (ARRETS_OPERATEUR)
OPERATOR_STOPS = ('stopped', 'cancelled by manual control', 'replaced by a new command')
ROBOT_CONFIG = os.path.join(HERE, 'robot_config_20261115.json')  # the robot's file, 11:33 UTC
IMU = ('ax', 'ay', 'az', 'gx', 'gy')


class TiltSim(Sim):
    """Sim that also feeds the IMU (tilt), and whose board can go silent."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.mute = False

    def feed(self, msgs: list[dict]) -> None:
        if self.mute:
            return
        super().feed(msgs)
        for m in msgs:
            if m['T'] == 1001:
                self.odom.on_imu(*(m[k] for k in IMU), self.t)

    def until(self, cond, limit: float) -> bool:
        end = self.t + limit
        while self.t < end:
            self.run(0.05)
            if cond():
                return True
        return False


@pytest.fixture
def tsim() -> TiltSim:
    s = TiltSim()
    s.run(2.5)
    assert s.odom.bias_ready and s.odom.tilt.ready
    assert s.ctrl.guard_state == 'ok'
    return s


def trip_while_moving(s: TiltSim) -> None:
    """Drive forward up a slope that steepens until the guard trips and backs off."""
    s.start(Step('move', 2.0))
    s.run(0.6)
    s.board.pitch_target = 35.0  # nose up, past pitch_stop_deg (25)
    assert s.until(lambda: s.ctrl.guard_state == 'backoff', 3.0)
    s.run(0.2)
    assert s.ctrl.v_cmd < 0  # reversing along the way it came


# -- tilt guard back-off: stopped by /stop, stale feedback and a lost link ---------------


def test_trip_ends_the_step_with_a_tilt_status_not_an_operator_stop(tsim: TiltSim) -> None:
    trip_while_moving(tsim)
    status = tsim.ctrl.history[0]['status']
    assert status.startswith('tilt: ') and status not in OPERATOR_STOPS
    assert tsim.ctrl.fault == status
    assert tsim.ctrl.mode == 'idle' and not tsim.ctrl.queue
    assert tsim.until(lambda: tsim.ctrl.guard_state != 'backoff', 4.0)  # ends by itself
    tsim.run(1.5)
    assert tsim.ctrl.v_cmd == 0.0 and len(tsim.ctrl.history) == 1


def test_stop_ends_the_back_off(tsim: TiltSim) -> None:
    trip_while_moving(tsim)
    tsim.ctrl.stop(tsim.t)
    for _ in range(40):  # 2 s: the back-off never comes back
        tsim.run(0.05)
        assert tsim.ctrl.v_cmd == 0.0 and tsim.ctrl.guard_state != 'backoff'
    assert abs(tsim.board.wl) < 0.01 and abs(tsim.board.wr) < 0.01
    # the stop added no 'stopped' entry: the step ended by the tilt is the last one
    assert len(tsim.ctrl.history) == 1 and tsim.ctrl.history[0]['status'].startswith('tilt')


def test_stale_feedback_ends_the_back_off(tsim: TiltSim) -> None:
    trip_while_moving(tsim)
    tsim.mute = True  # the ESP32 still obeys but says nothing
    tsim.run(0.8)
    assert tsim.ctrl.fault == 'no ESP32 feedback'
    assert tsim.ctrl.v_cmd == 0.0 and tsim.ctrl.guard_state != 'backoff'
    tsim.mute = False  # feedback back: nothing restarts
    tsim.run(2.0)
    assert tsim.ctrl.v_cmd == 0.0 and tsim.ctrl.guard_state != 'backoff'


def test_lost_link_ends_the_back_off(tsim: TiltSim) -> None:
    trip_while_moving(tsim)
    tsim.ctrl.fail('ESP32 link lost', tsim.t)  # what Rover._loop does when disconnected
    tsim.run(0.05)
    assert tsim.ctrl.v_cmd == 0.0 and tsim.ctrl.fault == 'ESP32 link lost'
    assert tsim.ctrl.guard_state != 'backoff'


def test_no_back_off_on_stale_tilt(tsim: TiltSim) -> None:
    tsim.drive(0.15, 0.0, 1.0)
    tsim.mute = True
    tsim.run(0.6)  # fed nothing: the drive is stopped, the tilt frozen
    tsim.odom.tilt.pitch = 40.0  # a stale reading past the stop angle
    tsim.run(0.1)
    assert tsim.ctrl.guard_state == 'hold' and tsim.ctrl.v_cmd == 0.0


# -- IMU fields validated like the other feedback numbers --------------------------------


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), 'x'])
def test_bad_imu_field_is_refused_and_tilt_untouched(bad) -> None:
    rover = Rover(Config(), lambda: None)
    good = {'T': 1001, 'L': 0.0, 'R': 0.0, 'gz': 0, 'ax': 0, 'ay': 0, 'az': 8400, 'gx': 0, 'gy': 0}
    rover._on_message(good, 1.0)
    assert rover.odom.tilt.ready and rover.odom.tilt.roll == 0.0
    with pytest.raises((TypeError, ValueError)):
        rover._on_message({**good, 'ay': bad}, 1.05)
    assert rover.odom.tilt.roll == 0.0 and rover.odom.tilt.pitch == 0.0
    assert rover.odom.feedback_age(1.05) == 0.0  # the wheels of that message still count


def test_receive_thread_survives_a_bad_imu_message() -> None:
    from fake_esp32 import FakeEsp32

    board = FakeEsp32()
    holder = {}

    def opener():
        holder['ser'] = FakeSerial(board, boot_time=0.3)
        return holder['ser']

    rover = ready_rover(opener)
    try:
        holder['ser']._push_raw(
            b'{"T":1001,"L":0,"R":0,"ax":NaN,"ay":0,"az":8400,"gx":0,"gy":0}\n'
        )
        holder['ser']._push_raw(
            b'{"T":1001,"L":0,"R":0,"ax":0,"ay":null,"az":8400,"gx":0,"gy":0}\n'
        )
        assert wait_for(lambda: rover.link.handler_errors >= 2, 2.0)
        assert math.isfinite(rover.odom.tilt.roll) and math.isfinite(rover.odom.tilt.pitch)
        count = rover.link.rx_count
        assert wait_for(lambda: rover.link.rx_count > count + 5, 2.0)
    finally:
        rover.close()


# -- recorder inside the protected loop ---------------------------------------------------


def test_recorder_runs_in_the_loop_and_its_error_is_caught() -> None:
    from fake_esp32 import FakeEsp32

    board = FakeEsp32()
    rover = ready_rover(lambda: FakeSerial(board, boot_time=0.3))
    try:
        with rover.lock:
            rover.recorder.start(time.monotonic())
        assert wait_for(lambda: len(rover.recorder.rows) >= 10, 2.0)
        lines = rover.recorder.csv().splitlines()
        assert lines[0].split(',') == list(rover.recorder.COLUMNS)
        assert len(lines[1].split(',')) == len(rover.recorder.COLUMNS)

        def broken(t, r):
            raise ZeroDivisionError('recorder bug')

        rover.manual(0.15, 0.0)
        assert rover.ctrl.mode == 'manual'
        rover.recorder.add = broken
        assert wait_for(lambda: rover.loop_errors >= 1, 2.0)
        assert rover._thread.is_alive() and 'recorder bug' in rover.last_loop_error
        # the error zeroes the wheels at once; manual mode itself ends after manual_timeout
        assert wait_for(lambda: rover.ctrl.mode == 'idle', 2.0)
        assert rover.ctrl.fault.startswith('controller error')
        rover.stop()
        with rover.lock:
            rover.recorder.stop()
        rover.manual(0.15, 0.0)
        assert wait_for(lambda: board.wl > 0.1, 2.0)  # drivable again
    finally:
        rover.close()


# -- settings: every new field has limits, the robot's own file loads unchanged -----------


def test_every_field_of_the_robot_code_has_limits() -> None:
    cfg = Config()
    assert set(CONFIG_LIMITS) == set(cfg.__dict__)
    for name, value in cfg.__dict__.items():
        lo, hi = CONFIG_LIMITS[name]
        assert lo <= value <= hi, name


@pytest.mark.parametrize(
    'values',
    [
        {'imu_roll_sign': 0},
        {'imu_pitch_sign': 2},
        {'tilt_guard': 0.5},
        {'roll_warn_deg': 30.0, 'roll_stop_deg': 20.0},
        {'pitch_stop_deg': float('nan')},
        {'tilt_filter_tau': 0.0},
        {'bias_max_rate': 0.0},
        {'no_rotation_rate': 0.0},
    ],
)
def test_bad_tilt_and_gyro_settings_are_refused(values) -> None:
    cfg = Config()
    before = dict(cfg.__dict__)
    with pytest.raises((ValueError, TypeError)):
        cfg.update(values)
    assert cfg.__dict__ == before


def test_robot_config_loads_unchanged(tmp_path, capsys) -> None:
    path = str(tmp_path / 'rover_config.json')
    shutil.copy(ROBOT_CONFIG, path)  # Config.load may rename a bad file: never the original
    with open(ROBOT_CONFIG, encoding='utf-8') as f:
        live = json.load(f)
    cfg = Config.load(path)
    assert 'using the default' not in capsys.readouterr().out
    assert set(live) == set(dataclasses.asdict(cfg))
    for name, value in live.items():
        assert getattr(cfg, name) == value and type(getattr(cfg, name)) is type(value), name
    assert os.path.exists(path) and not os.path.exists(path + '.bad')
    # the page's "Save settings" sends every field back as parseFloat: accepted as is
    assert cfg.update({k: float(v) for k, v in live.items()})
