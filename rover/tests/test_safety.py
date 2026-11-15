"""Safety: lost ESP32 feedback or link, manual take-over, bad inputs and settings, loop errors."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from test_rover_core import Sim  # noqa: E402

from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from rover_core import (  # noqa: E402
    MAX_MOVE_M,
    Config,
    Controller,
    NotReadyError,
    Odometry,
    Rover,
    Step,
)


class MutableSim(Sim):
    """Sim whose board can stop sending feedback while it keeps obeying commands."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.mute = False

    def feed(self, msgs: list[dict]) -> None:
        if not self.mute:
            super().feed(msgs)


@pytest.fixture
def sim() -> Sim:
    s = Sim()
    s.run(2.5)
    assert s.odom.bias_ready
    return s


@pytest.fixture
def msim() -> MutableSim:
    s = MutableSim()
    s.run(2.5)
    assert s.odom.bias_ready
    return s


# -- feedback watchdog (rover-core F1) ---------------------------------------------------


def test_lost_feedback_aborts_the_move_and_the_queue(msim: MutableSim) -> None:
    msim.start(Step('move', 1.0), Step('turn', 90.0))
    msim.run(1.0)
    assert msim.ctrl.mode == 'move'
    x_cut = msim.board.x
    msim.mute = True  # the ESP32 still drives but says nothing
    msim.run(1.0)
    assert msim.ctrl.mode == 'idle' and not msim.ctrl.queue
    assert msim.ctrl.history[0]['status'] == 'no ESP32 feedback'
    assert msim.ctrl.fault == 'no ESP32 feedback'
    assert msim.ctrl.v_cmd == 0.0 and msim.ctrl.w_cmd == 0.0
    msim.run(1.0)
    # Stopped within the 0.5 s limit (+ wheel inertia), not 5 m later at the timeout.
    assert msim.board.x - x_cut < 0.25
    assert abs(msim.board.wl) < 0.01
    msim.mute = False  # feedback is back: nothing restarts by itself
    msim.run(2.0)
    assert msim.ctrl.mode == 'idle' and len(msim.ctrl.history) == 1


def test_lost_feedback_stops_manual_driving(msim: MutableSim) -> None:
    msim.mute = True
    msim.drive(0.2, 0.0, 1.0)  # commands keep coming, feedback does not
    assert msim.ctrl.mode == 'idle' and msim.ctrl.v_cmd == 0.0
    assert msim.ctrl.fault == 'no ESP32 feedback'
    msim.mute = False
    msim.drive(0.2, 0.0, 0.5)  # feedback back: a new command drives again
    assert msim.ctrl.mode == 'manual' and msim.ctrl.fault is None


def test_idle_rover_without_feedback_is_not_a_fault(msim: MutableSim) -> None:
    msim.mute = True
    msim.run(2.0)
    assert msim.ctrl.mode == 'idle' and msim.ctrl.fault is None


# -- manual take-over (rover-core F3) ----------------------------------------------------


def test_manual_command_cancels_the_queued_steps_too(sim: Sim) -> None:
    sim.start(Step('move', 0.5), Step('turn', 90.0), Step('move', 0.5))
    sim.run(0.5)
    sim.drive(0.0, 0.0, 0.1)  # one {v: 0, omega: 0}, as the page sends on key release
    assert not sim.ctrl.queue
    sim.run(3.0)
    assert sim.ctrl.mode == 'idle'
    assert [h['status'] for h in sim.ctrl.history] == ['cancelled by manual control']
    assert abs(math.degrees(sim.board.yaw)) < 5


def test_manual_while_steps_only_queued_cancels_them(sim: Sim) -> None:
    sim.ctrl.run([Step('move', 0.5)], sim.t)  # queued, not started before the next tick
    sim.ctrl.manual(0.1, 0.0, sim.t)
    assert not sim.ctrl.queue


# -- gyro not ready (rover-core M1) ------------------------------------------------------


def test_steps_refused_before_the_gyro_is_ready() -> None:
    cfg = Config()
    ctrl = Controller(cfg, Odometry(cfg))
    with pytest.raises(NotReadyError):
        ctrl.run([Step('turn', 90.0)], 0.0)
    assert not ctrl.queue
    ctrl.manual(0.1, 0.0, 0.0)  # driving by hand stays possible
    ctrl.run([], 0.0)  # an empty list only cancels: allowed


# -- input validation (rover-core F5, rover-server F2) -----------------------------------


@pytest.mark.parametrize(
    'step',
    [
        {'move': float('nan')},
        {'move': 'nan'},
        {'move': 'inf'},
        {'move': True},
        {'move': MAX_MOVE_M + 1},
        {'turn': float('-inf')},
        {'turn': 7200},
        {'path': [[float('nan'), 0.0]]},
        {'path': [['inf', 0.0]]},
        {'path': [[1e309, 0.0]]},
        {'path': [[1.0]]},
        {'path': [1.0, 2.0]},
        {'path': 'abc'},
        {'path': [[1000.0, 0.0]]},
        {'path': [[0.1 * (k % 2), 0.0] for k in range(5000)]},
        {'path': [[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0], [-20.0, 0.0]]},
        {'path': [[1.0, 0.0]], 'speed': float('nan')},
        {'path': [[1.0, 0.0]], 'speed': 0},
    ],
)
def test_bad_steps_are_refused(step) -> None:
    with pytest.raises((ValueError, TypeError)):
        Step.parse(step)


def test_legitimate_steps_still_accepted() -> None:
    assert Step.parse({'move': 5.0}).value == 5.0
    assert Step.parse({'turn': '-360'}).value == -360.0
    assert Step.parse({'path': [[3.0, 0.0], [3.0, 2.0]], 'frame': 'odom'}).frame == 'odom'


@pytest.mark.parametrize('v, w', [(float('nan'), 0), (0, float('inf')), (50, 0), (True, 0)])
def test_bad_manual_commands_are_refused(sim: Sim, v, w) -> None:
    with pytest.raises((ValueError, TypeError)):
        sim.ctrl.manual(v, w, sim.t)
    assert sim.ctrl.mode == 'idle'


# -- settings (rover-core F4, rover-server F3/F12) ---------------------------------------


@pytest.mark.parametrize(
    'values',
    [
        {'track': 0},
        {'path_speed': 0},
        {'manual_timeout': float('nan')},
        {'manual_timeout': 0},
        {'manual_timeout': 30},
        {'task_timeout': 1e9},
        {'v_max': 5.0},
        {'yaw_sign': 0.5},
        {'yaw_sign': 2},
        {'v_min': 0.5, 'v_max': 0.2},
        {'dist_scale': 'inf'},
        {'w_min': None},
    ],
)
def test_bad_settings_are_refused_whole(values) -> None:
    cfg = Config()
    before = dict(cfg.__dict__)
    with pytest.raises((ValueError, TypeError)):
        cfg.update({'v_manual': 0.25, **values})
    assert cfg.__dict__ == before  # nothing applied, not even v_manual


def test_every_setting_has_limits_and_defaults_pass() -> None:
    from rover_core import CONFIG_LIMITS

    cfg = Config()
    assert set(CONFIG_LIMITS) == set(cfg.__dict__)
    assert cfg.update(dict(cfg.__dict__))  # the page sends every field back


def test_page_style_settings_accepted(tmp_path) -> None:
    cfg = Config()
    body = {k: float(v) for k, v in cfg.__dict__.items()}  # parseFloat on every field
    body['yaw_sign'] = -1.0
    cfg.update(body)
    assert cfg.yaw_sign == -1 and isinstance(cfg.yaw_sign, int)


def test_save_is_atomic_and_load_survives_a_broken_file(tmp_path) -> None:
    path = str(tmp_path / 'cfg.json')
    cfg = Config()
    cfg.update({'v_manual': 0.2})
    cfg.save(path)
    assert os.listdir(tmp_path) == ['cfg.json']
    assert Config.load(path).v_manual == 0.2
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{\n  "v_manual": 0.15,\n  "v_ma')  # cut while writing
    loaded = Config.load(path)
    assert loaded == Config()
    assert os.path.exists(path + '.bad')


def test_load_skips_only_the_bad_values(tmp_path) -> None:
    path = str(tmp_path / 'cfg.json')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{"v_manual": 0.2, "manual_timeout": NaN, "track": 0}')
    loaded = Config.load(path)
    assert loaded.v_manual == 0.2
    assert loaded.manual_timeout == Config().manual_timeout
    assert loaded.track == Config().track


# -- the real threads: link loss, loop and receive errors (rover-core F2/F4, server F1) --


class Unpluggable:
    """Opener for FakeSerial whose cable can be pulled out and plugged back."""

    def __init__(self, board: FakeEsp32) -> None:
        self.board = board
        self.plugged = True
        self.opened = 0
        self.ser = None

    def __call__(self):
        if not self.plugged:
            raise OSError('could not open port')
        self.opened += 1
        self.ser = FakeSerial(self.board, boot_time=0.3)
        real_read, real_write = self.ser.readline, self.ser.write

        def readline():
            if not self.plugged:
                raise OSError('device disconnected')
            return real_read()

        def write(data):
            if not self.plugged:
                raise OSError('device disconnected')
            return real_write(data)

        self.ser.readline, self.ser.write = readline, write
        return self.ser


def wait_for(cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def ready_rover(opener, **cfg) -> Rover:
    rover = Rover(Config(bias_calib_time=0.5, **cfg), opener)
    rover.start()
    assert wait_for(lambda: rover.odom.bias_ready, 8.0)
    return rover


def test_link_loss_aborts_and_nothing_resumes() -> None:
    board = FakeEsp32()
    cable = Unpluggable(board)
    rover = ready_rover(cable)
    try:
        rover.run([Step('move', 2.0), Step('turn', 90.0)])
        assert wait_for(lambda: rover.ctrl.mode == 'move' and board.x > 0.05, 3.0)
        cable.plugged = False
        assert wait_for(lambda: rover.ctrl.mode == 'idle', 1.5)
        assert not rover.ctrl.queue
        assert rover.ctrl.history[0]['status'] in ('ESP32 link lost', 'no ESP32 feedback')
        cable.plugged = True
        assert wait_for(lambda: rover.link.connected and cable.opened == 2, 6.0)
        time.sleep(1.0)
        assert rover.ctrl.mode == 'idle' and len(rover.ctrl.history) == 1
        assert abs(board.wl) < 0.01 and abs(board.wr) < 0.01
    finally:
        rover.close()


def test_feedback_switched_off_stops_a_move() -> None:
    board = FakeEsp32()
    rover = ready_rover(lambda: FakeSerial(board, boot_time=0.3))
    try:
        rover.run([Step('move', 2.0)])
        assert wait_for(lambda: board.x > 0.05, 3.0)
        rover.raw({'T': 131, 'cmd': 0})  # the board stops its feedback, the link stays up
        x_cut = board.x
        assert wait_for(lambda: rover.ctrl.mode == 'idle', 1.5)
        assert rover.ctrl.history[0]['status'] == 'no ESP32 feedback'
        time.sleep(0.5)
        assert board.x - x_cut < 0.3 and abs(board.wl) < 0.01
    finally:
        rover.close()


def test_loop_survives_an_exception_and_stop_still_works() -> None:
    board = FakeEsp32()
    rover = ready_rover(lambda: FakeSerial(board, boot_time=0.3))
    try:
        rover.manual(0.2, 0.0)
        real_step = rover.ctrl.step
        calls = []

        def broken(t):
            calls.append(t)
            if len(calls) == 1:
                raise ZeroDivisionError('float division by zero')
            return real_step(t)

        rover.ctrl.step = broken
        assert wait_for(lambda: len(calls) > 3, 2.0)
        assert rover._thread.is_alive()
        snap = rover.snapshot()
        assert snap['loop_errors'] == 1 and 'ZeroDivisionError' in snap['last_loop_error']
        assert rover.ctrl.mode == 'idle' and rover.ctrl.fault.startswith('controller error')
        rover.stop()
        rover.manual(0.15, 0.0)
        assert wait_for(lambda: board.wl > 0.1, 2.0)  # still drivable
    finally:
        rover.close()


def test_receive_thread_survives_a_bad_message() -> None:
    board = FakeEsp32()
    holder = {}

    def opener():
        holder['ser'] = FakeSerial(board, boot_time=0.3)
        return holder['ser']

    rover = ready_rover(opener)
    try:
        holder['ser']._push_raw(b'{"T":1001,"L":null,"R":0.0}\n')
        holder['ser']._push_raw(b'{"T":1001,"L":NaN,"R":0.0}\n')
        assert wait_for(lambda: rover.link.handler_errors >= 2, 2.0)
        assert rover.link._thread.is_alive()
        assert math.isfinite(rover.odom.x) and math.isfinite(rover.odom.L)
        count = rover.link.rx_count
        assert wait_for(lambda: rover.link.rx_count > count + 5, 2.0)
        assert rover.snapshot()['link']['handler_errors'] >= 2
    finally:
        rover.close()


def test_path_start_failure_leaves_a_clean_controller(sim: Sim) -> None:
    step = Step.parse({'path': [[1.0, 0.0]]})
    step.points = [[float('inf'), 0.0]]  # bypasses the checks, as a bug would
    sim.ctrl.run([step], sim.t)
    with pytest.raises((OverflowError, ValueError)):
        sim.ctrl.step(sim.t)
    assert sim.ctrl.mode != 'path'
    sim.ctrl.stop(sim.t)  # does not raise
    json.dumps(sim.ctrl.snapshot())
