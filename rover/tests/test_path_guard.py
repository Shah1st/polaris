"""Local guard and non-stop path replacement, against the simulated ESP32 in simulated time.

The depth camera is replaced by `see`: a perfect one, 1.25 m of range, that sees the obstacles
given to it from the rover's true pose.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The two helper imports below come from this folder: pytest's importlib mode does not add it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_local_map import picture  # noqa: E402
from test_rover_core import Sim  # noqa: E402

from local_map import LocalMap  # noqa: E402
from rover_core import Config, Step  # noqa: E402


def guarded(**cfg) -> Sim:
    s = Sim(Config(local_guard=1, dist_scale=0.98, **cfg))
    s.ctrl.local = LocalMap()
    s.run(2.5)  # gyro bias
    return s


def see(s: Sim, obstacles) -> None:
    """One picture of the fake depth camera: `obstacles` are (x, y) in the world."""
    b = s.board
    c, si = math.cos(b.yaw), math.sin(b.yaw)
    local = []
    for ox, oy in obstacles:
        dx, dy = ox - b.x, oy - b.y
        x, y = c * dx + si * dy, -si * dx + c * dy
        if 0.0 <= x < 1.25 and abs(y) < 1.25:
            local.append((x, y))
    o = s.odom
    s.ctrl.local.integrate(*picture(local), (o.x, o.y, o.heading), s.t)


def drive(s: Sim, obstacles, limit: float = 40.0, hz: float = 10.0) -> list[float]:
    """Run until nothing drives the rover any more (the back-off after a guard stop included),
    with pictures at `hz`; the commanded speeds."""
    speeds = []
    end = s.t + limit
    while s.t < end:
        if obstacles is not None:
            see(s, obstacles() if callable(obstacles) else obstacles)
        s.run(1.0 / hz)
        speeds.append(s.ctrl.v_cmd)
        if not s.ctrl.busy() and s.ctrl.guard_state != 'hold':
            return speeds
    raise AssertionError(f'still {s.ctrl.mode} after {limit} s')


NOSE = 0.115  # m from the rover's centre to its front


def test_clear_path_is_driven_at_full_speed() -> None:
    s = guarded()
    s.start(Step.path([[1.5, 0.0]]))
    speeds = drive(s, [])
    assert s.ctrl.history[0]['status'] == 'done'
    assert max(speeds) == pytest.approx(0.2, abs=0.01)
    assert s.board.x == pytest.approx(1.5, abs=0.04)


def test_obstacle_on_the_path_slows_blocks_without_contact_and_backs_off() -> None:
    s = guarded()
    box = (1.02, 0.02)  # its cell is 1.00 .. 1.05
    s.start(Step.path([[2.0, 0.0]]))
    states, furthest = [], 0.0
    end = s.t + 30
    while s.t < end and (s.ctrl.busy() or s.ctrl.guard_state == 'hold'):
        see(s, [box])
        s.run(0.1)
        states.append((s.ctrl.local_state['state'], s.ctrl.v_cmd))
        furthest = max(furthest, s.board.x)
    entry = s.ctrl.history[0]
    assert entry['status'] == 'blocked'
    assert entry['obstacle'] == pytest.approx([1.025, 0.025], abs=0.06)  # odometry frame
    names = [n for n, _ in states]
    assert 'slow' in names and names.index('slow') < names.index('blocked')
    slow = [v for n, v in states if n == 'slow' and v > 0]
    assert min(slow) < 0.12  # it braked before stopping
    assert furthest + NOSE < 1.0 - 0.04  # the nose stayed 4 cm or more from the box
    # then it moved back about guard_backoff, so that a new path can start
    assert furthest - s.board.x == pytest.approx(s.cfg.guard_backoff, abs=0.04)
    assert abs(s.board.wl) < 0.01 and abs(s.board.wr) < 0.01


def test_after_a_stop_a_path_that_turns_away_can_start() -> None:
    """Stopped in front of the box, then sent round it: the guard lets the rover pivot and
    leave (before the back-off, the box stayed inside the corridor of every new path)."""
    s = guarded()
    box = [(1.02, 0.02), (1.02, -0.03)]
    s.start(Step.path([[2.0, 0.0]]))
    drive(s, box)
    assert s.ctrl.history[0]['status'] == 'blocked'
    x = s.odom.x
    s.start(Step.path([[x + 0.1, 0.35], [1.6, 0.35]], frame='odom'))
    drive(s, box)
    assert s.ctrl.history[0]['status'] == 'done'
    assert s.board.x == pytest.approx(1.6, abs=0.06) and s.board.y > 0.25


def test_without_back_off_the_rover_stays_where_it_stopped() -> None:
    s = guarded(guard_backoff=0.0)
    s.start(Step.path([[2.0, 0.0]]))
    xs = []
    end = s.t + 30
    while s.t < end and s.ctrl.busy():
        see(s, [(1.02, 0.02)])
        s.run(0.1)
        xs.append(s.board.x)
    assert s.ctrl.history[0]['status'] == 'blocked'
    assert max(xs) - s.board.x < 0.02 and s.ctrl.guard_state != 'backoff'


def test_a_phantom_seen_in_one_picture_does_not_stop_the_rover() -> None:
    s = guarded()
    s.start(Step.path([[1.5, 0.0]]))
    n = [0]

    def scene():
        n[0] += 1
        # every fifth picture has a speck right on the path, 30 cm ahead of the rover
        return [(s.board.x + 0.3, 0.02)] if n[0] % 5 == 0 else []

    speeds = drive(s, scene)
    assert s.ctrl.history[0]['status'] == 'done'
    assert s.board.x == pytest.approx(1.5, abs=0.04)
    assert min(speeds[5:-12]) > 0.15  # it did not even slow down


def test_a_path_the_planner_may_send_15_cm_from_an_obstacle_is_driven() -> None:
    """The planner keeps the rover's centre 15 cm from obstacle cells. The guard's corridor
    is the body (10 cm to a cell's centre): such a path must not trip it."""
    s = guarded()
    s.start(Step.path([[1.5, 0.0]]))
    wall = [(0.5 + 0.05 * k, side) for k in range(10) for side in (0.13, -0.13)]
    speeds = drive(s, wall)
    assert s.ctrl.history[0]['status'] == 'done'
    assert max(speeds) == pytest.approx(0.2, abs=0.01)
    # ... and something that really is in the body's way still stops it
    s = guarded()
    s.start(Step.path([[1.5, 0.0]]))
    drive(s, [(0.8, 0.07)])
    assert s.ctrl.history[0]['status'] == 'blocked'


def test_without_pictures_the_rover_creeps_then_drives_on_unprotected() -> None:
    s = guarded()
    s.start(Step.path([[2.0, 0.0]]))
    s.run(2.5)  # the capture server is silent
    assert s.ctrl.local_state['state'] == 'blind'
    assert 0 < s.ctrl.v_cmd <= s.cfg.guard_blind_speed + 1e-6
    s.run(1.5)  # more than guard_blind_time: the depth camera is given up
    assert s.ctrl.local_state['state'] == 'no_camera'
    assert s.ctrl.snapshot()['local']['state'] == 'no_camera'
    s.run(1.0)
    assert s.ctrl.v_cmd == pytest.approx(0.2, abs=0.01)
    s.run_until_idle()
    assert s.ctrl.history[0]['status'] == 'done'


def test_with_guard_blind_stop_the_path_ends_instead() -> None:
    s = guarded(guard_blind_stop=1)
    s.start(Step.path([[2.0, 0.0]]))
    s.run_until_idle()
    assert s.ctrl.history[0]['status'] == 'no depth camera'
    assert s.board.x < 0.4  # crept for guard_blind_time, then stopped


def test_pictures_that_stop_and_come_back() -> None:
    s = guarded()
    s.start(Step.path([[3.0, 0.0]]))
    for _ in range(20):
        see(s, [])
        s.run(0.1)
    assert s.ctrl.v_cmd == pytest.approx(0.2, abs=0.01)
    s.run(0.6)  # a few late pictures: under guard_max_age, nothing happens
    assert s.ctrl.local_state['state'] == 'ok' and s.ctrl.v_cmd == pytest.approx(0.2, abs=0.01)
    s.run(1.0)  # a real gap
    assert s.ctrl.local_state['state'] == 'blind'
    assert s.ctrl.v_cmd <= s.cfg.guard_blind_speed + 1e-6
    for _ in range(10):  # the camera is back
        see(s, [])
        s.run(0.1)
    assert s.ctrl.local_state['state'] == 'ok' and s.ctrl.v_cmd == pytest.approx(0.2, abs=0.01)


def test_what_was_seen_is_forgotten_even_when_pictures_stop() -> None:
    """Blocked by a box, then the capture server dies and the box is taken away: once the
    rover's memory has run out (local_hold_near: the box was under its nose), it drives on
    (unprotected, and saying so). Before, such a cell was kept for ever."""
    s = guarded(guard_backoff=0.0)
    s.start(Step.path([[2.0, 0.0]]))
    drive(s, [(1.02, 0.02)])
    assert s.ctrl.history[0]['status'] == 'blocked'
    s.run(s.cfg.local_hold + 1.0)  # no picture at all
    s.start(Step.path([[2.0, 0.0]], frame='odom'))
    s.run_until_idle(limit=60.0)
    assert s.ctrl.history[0]['status'] == 'blocked'  # still remembered: it was very near
    s.run(s.cfg.local_hold_near)
    s.start(Step.path([[2.0, 0.0]], frame='odom'))
    s.run_until_idle(limit=60.0)
    assert s.ctrl.history[0]['status'] == 'done' and s.board.x == pytest.approx(2.0, abs=0.06)


def test_obstacle_that_leaves_lets_the_next_path_through() -> None:
    s = guarded()
    s.start(Step.path([[2.0, 0.0]]))
    drive(s, [(1.02, 0.02)])
    assert s.ctrl.history[0]['status'] == 'blocked'
    s.start(Step.path([[2.0, 0.0]], frame='odom'))
    drive(s, [])  # it is gone: the camera sees the floor there
    assert s.ctrl.history[0]['status'] == 'done'
    assert s.board.x == pytest.approx(2.0, abs=0.06)


def test_guard_off_by_default_changes_nothing() -> None:
    s = Sim(Config(dist_scale=0.98))
    assert s.cfg.local_guard == 0
    s.ctrl.local = LocalMap()
    s.run(2.5)
    s.start(Step.path([[1.5, 0.0]]))
    drive(s, [(1.02, 0.02)])
    assert s.ctrl.history[0]['status'] == 'done'  # drove through: nobody was looking
    assert s.ctrl.snapshot()['local'] == {'state': 'off'}


# -- replacing a path in motion ---------------------------------------------------------


def test_path_with_an_id_takes_over_without_slowing_down() -> None:
    s = Sim(Config(dist_scale=0.98))
    s.run(2.5)
    s.start(Step.path([[1.0, 0.0]], frame='odom', id='m1'))
    s.run(2.5)
    assert s.ctrl.snapshot()['current']['id'] == 'm1'
    v_before = s.ctrl.v_cmd
    assert v_before == pytest.approx(0.2, abs=0.01)
    # the same line, further, starting a little behind the rover (planned a moment ago)
    s.start(Step.path([[s.odom.x - 0.1, 0.0], [2.0, 0.0]], frame='odom', id='m2'))
    slowest = v_before
    for _ in range(20):
        s.run(0.05)
        slowest = min(slowest, s.ctrl.v_cmd)
    assert slowest > 0.18  # no stop, no pivot, no ramp from zero
    old = s.ctrl.history[0]
    assert old['status'] == 'replaced by a new command'
    assert old['id'] == 'm1' and old['replaced_by'] == 'm2'
    s.run_until_idle()
    assert s.ctrl.history[0]['status'] == 'done' and s.ctrl.history[0]['id'] == 'm2'
    assert s.board.x == pytest.approx(2.0, abs=0.05) and abs(s.board.y) < 0.03


def test_replacement_that_bends_is_followed_as_an_arc() -> None:
    s = Sim(Config(dist_scale=0.98))
    s.run(2.5)
    s.start(Step.path([[2.0, 0.0]], frame='odom', id='a'))
    s.run(2.5)
    x = s.odom.x
    # the plan changed: bear left by 30 degrees from a little ahead of the rover
    turn = (x + 0.15, 0.0)
    goal = (turn[0] + 0.8 * math.cos(math.radians(30)), 0.8 * math.sin(math.radians(30)))
    s.start(Step.path([[x, 0.0], list(turn), list(goal)], frame='odom', id='b'))
    slowest = 1.0
    while not (s.ctrl.mode == 'idle' and not s.ctrl.queue):
        s.run(0.05)
        if s.ctrl.mode == 'path' and math.dist((s.board.x, s.board.y), goal) > 0.25:
            slowest = min(slowest, s.ctrl.v_cmd)
    assert slowest > 0.1  # never stopped to pivot
    assert math.dist((s.board.x, s.board.y), goal) < 0.05


def test_path_without_id_or_far_from_the_rover_starts_as_before() -> None:
    s = Sim(Config(dist_scale=0.98))
    s.run(2.5)
    s.start(Step.path([[2.0, 0.0]], frame='odom'))
    s.run(2.5)
    s.start(Step.path([[2.0, 0.0]], frame='odom'))  # no id: the old behaviour
    s.run(0.05)
    assert s.ctrl.v_cmd <= s.cfg.v_min + 0.04
    assert 'replaced_by' not in s.ctrl.history[0] and 'id' not in s.ctrl.history[0]
    s.run(2.0)
    # an id, but the new path starts half a metre to the side: an ordinary start
    s.start(Step.path([[s.odom.x, 0.5], [s.odom.x + 1.0, 0.5]], frame='odom', id='far'))
    s.run(0.05)
    assert s.ctrl.v_cmd == 0.0 and s.ctrl.snapshot()['path']['pivoting']


def test_path_id_is_checked() -> None:
    assert Step.parse({'path': [[1, 0]], 'id': 'seg-7'}).id == 'seg-7'
    assert Step.parse({'path': [[1, 0]]}).id is None
    for bad in ('', 'x' * 41, 7, ['a']):
        with pytest.raises(ValueError):
            Step.path([[1, 0]], id=bad)
