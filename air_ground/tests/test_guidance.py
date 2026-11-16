"""Steering the rover from the air, on a fake rover whose odometry is wrong on purpose (it
starts anywhere, under-counts the distance and drifts in heading): the air camera must take
it to the goal anyway."""

from __future__ import annotations

import math
import os
import random
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from guidance import (  # noqa: E402
    ClockSync,
    Guide,
    GuideConfig,
    OdomHistory,
    OdomLink,
    Pose2,
    wrap,
)

AIR_DELAY = 0.15  # s between a picture and its result


class FakeRover:
    """A rover on a floor: `truth` is its real pose (floor frame). Its odometry starts at zero
    wherever it is, counts `scale` of the distance and drifts by `drift` rad/s while moving.
    It follows a path given in ITS odometry frame, like the real path follower."""

    def __init__(self, truth: Pose2, scale: float = 0.9, drift: float = math.radians(1.5)):
        self.truth = truth
        self.odom = Pose2(0.0, 0.0, 0.0)
        self.scale, self.drift = scale, drift
        self.points: list[tuple[float, float]] = []
        self.speed = 0.0
        self.mode = 'idle'
        self.stop_count = 0
        self.boot_id = 'boot-1'
        self.paths: list[list[tuple[float, float]]] = []
        self.path_ids: list[str | None] = []
        self.trail: list[tuple[float, Pose2]] = []
        self.t = 0.0

    def path(self, points, speed, path_id=None):
        self.points, self.speed, self.mode = [tuple(p) for p in points], speed, 'path'
        self.paths.append(self.points)
        self.path_ids.append(path_id)

    def stop(self):
        self.points, self.mode = [], 'idle'
        self.stop_count += 1

    def state(self) -> dict:
        o = self.odom
        return {
            'odom': {'x': o.x, 'y': o.y, 'heading_deg': math.degrees(o.heading)},
            'ctrl': {'mode': self.mode},
            'stop_count': self.stop_count,
            'boot_id': self.boot_id,
        }

    def tick(self, dt: float) -> None:
        self.t += dt
        v = w = 0.0
        if self.mode == 'path':
            o = self.odom
            while self.points and math.dist((o.x, o.y), self.points[0]) < 0.02:
                self.points.pop(0)
            if not self.points:
                self.mode = 'idle'
            else:
                tx, ty = self.points[0]
                err = wrap(math.atan2(ty - o.y, tx - o.x) - o.heading)
                if abs(err) > math.radians(20):
                    w = math.copysign(1.2, err)
                else:
                    v, w = self.speed, 2.5 * err
        t, o = self.truth, self.odom
        self.truth = Pose2(
            t.x + v * math.cos(t.heading) * dt, t.y + v * math.sin(t.heading) * dt,
            wrap(t.heading + w * dt),
        )  # fmt: skip
        vo, wo = v * self.scale, w + (self.drift if v else 0.0)
        self.odom = Pose2(
            o.x + vo * math.cos(o.heading) * dt, o.y + vo * math.sin(o.heading) * dt,
            wrap(o.heading + wo * dt),
        )  # fmt: skip
        self.trail.append((self.t, self.truth))

    def truth_at(self, t: float) -> Pose2:
        return min(self.trail, key=lambda item: abs(item[0] - t))[1] if self.trail else self.truth


def sighting(p: Pose2, rng: random.Random, noise_m=0.004, noise_deg=1.0, side_px=60.0):
    return SimpleNamespace(
        x=p.x + rng.gauss(0, noise_m),
        y=p.y + rng.gauss(0, noise_m),
        heading=p.heading + math.radians(rng.gauss(0, noise_deg)),
        side_px=side_px,
        scale_error=0.01,
    )


def run(guide: Guide, rover: FakeRover, seconds: float, air=lambda t: True, seed: int = 5):
    """The whole loop in simulated time: rover at 50 Hz, its state read at 10 Hz, pictures at
    15 Hz arriving AIR_DELAY late, the guide stepped at 10 Hz. Returns the status lines."""
    rng = random.Random(seed)
    statuses = []
    steps = round(seconds / 0.02)
    for k in range(1, steps + 1):
        rover.tick(0.02)
        t = rover.t
        if k % 5 == 0:
            guide.on_state(rover.state(), t)
        if k % 3 == 0 and air(t):
            guide.on_sighting(sighting(rover.truth_at(t - AIR_DELAY), rng), t - AIR_DELAY)
        if k % 5 == 0:
            status = guide.step(t)
            if not statuses or statuses[-1] != status:
                statuses.append(status)
    return statuses


# -- the pieces ---------------------------------------------------------------------------


def test_history_gives_the_pose_at_the_time_of_a_picture() -> None:
    h = OdomHistory(seconds=2.0)
    assert h.at(1.0) is None
    for k in range(11):  # 0.1 m/s along x, turning through +pi
        h.add(k * 0.1, Pose2(0.01 * k, 0.0, wrap(3.0 + 0.05 * k)))
    mid = h.at(0.45)
    assert mid.x == pytest.approx(0.045) and mid.heading == pytest.approx(wrap(3.225))
    assert h.at(1.2).x == pytest.approx(0.10)  # a little after the last reading: the last one
    assert h.at(2.0) is None and h.at(-1.0) is None  # too far from what is known
    h.add(5.0, Pose2(1, 1, 0))
    assert h.at(0.5) is None  # older than the window: forgotten


def test_clock_sync_uses_the_fastest_trip() -> None:
    sync = ClockSync(min_latency=0.05)
    # the phone's clock is 1000 s behind; trips take 0.08 to 0.30 s
    for k, trip in enumerate([0.30, 0.12, 0.08, 0.20]):
        local = sync.local_time(10.0 + k, 1010.0 + k + trip)
    assert local == pytest.approx(1013.0 + 0.08 - 0.05)  # picture time on this clock
    assert sync.latency(13.0, 1013.20) == pytest.approx(0.20 - 0.08 + 0.05)


def test_link_finds_the_transform_and_holds_it_against_noise() -> None:
    rng = random.Random(2)
    theta, tx, ty = math.radians(70), 0.4, -0.3  # floor <- odom, unknown to the link
    link = OdomLink()
    for k in range(120):  # the rover drives a curve in its odometry frame
        odom = Pose2(0.01 * k, 0.002 * k, 0.01 * k)
        c, s = math.cos(theta), math.sin(theta)
        seen = Pose2(
            c * odom.x - s * odom.y + tx, s * odom.x + c * odom.y + ty, odom.heading + theta
        )
        assert link.update(sighting(seen, rng, 0.006, 1.5), odom)
    # 1.5 degrees of noise per sighting, about 0.6 once filtered
    assert math.degrees(wrap(link.theta - theta)) == pytest.approx(0, abs=2.0)
    est = link.to_floor(Pose2(1.0, 0.2, 0.0))
    c, s = math.cos(theta), math.sin(theta)
    assert (est.x, est.y) == pytest.approx((c - 0.2 * s + tx, s + 0.2 * c + ty), abs=0.015)
    back = link.to_odom(est.x, est.y)
    assert back == pytest.approx((1.0, 0.2), abs=1e-9)


def test_link_ignores_a_wild_sighting_but_follows_a_real_reset() -> None:
    link = OdomLink()
    odom = Pose2(0.5, 0.0, 0.0)
    link.update(Pose2(1.0, 1.0, 0.5), odom)
    before = (link.theta, link.tx, link.ty)
    assert not link.update(Pose2(2.0, 1.0, 0.5), odom)  # 1 m away: another marker? ignored
    assert (link.theta, link.tx, link.ty) == before and link.last_error_m == pytest.approx(1.0)
    link.update(Pose2(1.0, 1.0, 0.5), odom)  # a good one in between: the count starts again
    for _ in range(4):
        assert not link.update(Pose2(2.0, 1.0, 2.0), odom)
    assert link.update(Pose2(2.0, 1.0, 2.0), odom)  # five in a row: the odometry was reset
    assert link.resets == 1
    here = link.to_floor(odom)
    assert (here.x, here.y, here.heading) == pytest.approx((2.0, 1.0, 2.0))


# -- steering -----------------------------------------------------------------------------


def test_reaches_the_goal_although_the_odometry_is_wrong() -> None:
    start = Pose2(0.4, -0.3, math.radians(70))
    goal = (-0.5, 0.6)
    rover = FakeRover(start)
    guide = Guide(rover)
    guide.set_goal(*goal)
    statuses = run(guide, rover, 30.0)
    assert statuses[-1].startswith('arrived'), statuses
    assert math.dist((rover.truth.x, rover.truth.y), goal) < 0.05
    assert rover.mode == 'idle' and guide.goal is None
    # corrections on the way, but not one path per picture: the rover's own loop does the work
    assert 2 <= len(rover.paths) <= 15, len(rover.paths)

    # The same run with the air camera cut after the first second: the rover believes its
    # odometry and misses the goal by much more.
    blind = FakeRover(start)
    guide = Guide(blind, GuideConfig(blind_s=1e9))
    guide.set_goal(*goal)
    run(guide, blind, 30.0, air=lambda t: t < 1.0)
    assert math.dist((blind.truth.x, blind.truth.y), goal) > 0.10


def test_waits_for_the_first_sighting_and_stops_when_blind() -> None:
    rover = FakeRover(Pose2(0.0, 0.0, 0.0))
    guide = Guide(rover, GuideConfig(blind_max_m=0.3))
    guide.set_goal(1.5, 0.0)
    statuses = run(guide, rover, 2.0, air=lambda t: False)
    assert statuses == ['waiting: the rover has not been seen from the air yet']
    assert rover.paths == []  # nothing sent before the rover is seen once
    # seen for 3 s (it drives), then lost for 5 s: it goes on 0.3 m on its odometry, then
    # stops and waits; seen again, it goes on to the goal
    statuses = run(guide, rover, 20.0, air=lambda t: t < 5.0 or t > 10.0)
    assert any('on odometry' in s for s in statuses)
    stopped = [s for s in statuses if s.startswith('stopped: rover not seen')]
    assert stopped and rover.stop_count == 1  # stopped once, not at every step
    assert statuses[-1].startswith('arrived')
    assert math.dist((rover.truth.x, rover.truth.y), (1.5, 0.0)) < 0.05


def test_a_short_loss_of_the_picture_does_not_stop_the_rover() -> None:
    """The phone loses its picture for 2 s in the middle of the drive: the rover goes on along
    its route on odometry, without stopping, and is corrected once seen again."""
    rover = FakeRover(Pose2(0.0, 0.0, 0.0))
    guide = Guide(rover)
    guide.set_goal(1.5, 0.3)
    stops = []

    def air(t):
        return not 5.0 < t < 7.0

    statuses = run(guide, rover, 25.0, air=air)
    assert rover.stop_count == 0
    assert any('on odometry' in s for s in statuses)
    moving_while_blind = [p for t, p in rover.trail if 5.5 < t < 6.9]
    assert math.dist((moving_while_blind[0].x, moving_while_blind[0].y),
                     (moving_while_blind[-1].x, moving_while_blind[-1].y)) > 0.1  # fmt: skip
    assert statuses[-1].startswith('arrived') and not stops
    assert math.dist((rover.truth.x, rover.truth.y), (1.5, 0.3)) < 0.05
    assert guide.blind_m == 0.0 and guide.blind_uncertainty() == 0.0


def test_corrections_while_driving_do_not_resend_the_path_every_time() -> None:
    """The air camera corrects the rover's place 15 times a second; the rover drives its
    route on odometry and gets it again only for a large correction, at most every 2 s."""
    rover = FakeRover(Pose2(0.0, 0.0, 0.0))
    guide = Guide(rover)
    guide.set_goal(2.0, 0.0)
    statuses = run(guide, rover, 30.0)
    assert statuses[-1].startswith('arrived')
    assert len(rover.paths) <= 6, len(rover.paths)  # 13 s of driving


def test_yields_to_an_operator_stop_and_to_manual_driving() -> None:
    rover = FakeRover(Pose2(0.0, 0.0, 0.0))
    guide = Guide(rover)
    guide.set_goal(1.5, 0.0)
    run(guide, rover, 3.0)
    assert rover.mode == 'path'
    rover.stop()  # someone presses STOP on the dashboard
    sent = len(rover.paths)
    statuses = run(guide, rover, 3.0)
    assert statuses[-1] == 'stopped: someone else took the rover over'
    assert len(rover.paths) == sent and rover.mode == 'idle' and guide.goal is None

    guide.set_goal(1.5, 0.0)
    rover.mode = 'manual'  # someone drives with the keys before the guide sent anything
    statuses = run(guide, rover, 1.0)
    assert statuses[-1] == 'stopped: someone else took the rover over'
    assert len(rover.paths) == sent


def test_follows_waypoints_and_a_server_restart_resets_the_link() -> None:
    rover = FakeRover(Pose2(0.0, 0.0, 0.0), scale=1.0, drift=0.0)
    guide = Guide(rover)
    guide.set_goal(1.0, 1.0, waypoints=[(1.0, 0.0)])  # around a corner, as a planner would say
    statuses = run(guide, rover, 25.0)
    assert statuses[-1].startswith('arrived')
    assert min(math.dist((p.x, p.y), (1.0, 0.0)) for _, p in rover.trail) < 0.08
    assert math.dist((rover.truth.x, rover.truth.y), (1.0, 1.0)) < 0.05

    # The rover's server restarts: its odometry is zero again, somewhere else.
    rover.odom, rover.boot_id = Pose2(0.0, 0.0, 0.0), 'boot-2'
    guide.set_goal(0.2, 1.0)
    statuses = run(guide, rover, 20.0)
    assert statuses[-1].startswith('arrived'), statuses
    assert math.dist((rover.truth.x, rover.truth.y), (0.2, 1.0)) < 0.05


def test_bad_sightings_are_dropped_with_a_reason() -> None:
    guide = Guide(FakeRover(Pose2(0, 0, 0)))
    guide.on_state({'odom': {'x': 0, 'y': 0, 'heading_deg': 0}, 'ctrl': {'mode': 'idle'}}, 1.0)
    rng = random.Random(1)
    assert not guide.on_sighting(None, 1.0) and guide.dropped is None
    assert not guide.on_sighting(sighting(Pose2(0, 0, 0), rng, side_px=9), 1.0)
    assert 'too small' in guide.dropped
    wrong = sighting(Pose2(0, 0, 0), rng)
    wrong.scale_error = 0.35
    assert not guide.on_sighting(wrong, 1.0) and 'off its size' in guide.dropped
    assert not guide.on_sighting(sighting(Pose2(0, 0, 0), rng), 9.0)
    assert 'no odometry' in guide.dropped
    # the marker is 6 cm ahead of the rover's centre: the rover is 6 cm behind the marker
    guide.cfg.marker_offset = (0.06, 0.0)
    seen = SimpleNamespace(x=1.0, y=0.5, heading=math.pi / 2, side_px=50, scale_error=0.0)
    assert guide.on_sighting(seen, 1.0)
    here = guide.rover_on_floor()
    assert (here.x, here.y) == pytest.approx((1.0, 0.44)) and guide.snapshot()['link']['ready']


def test_no_second_command_before_the_rover_state_is_read_again() -> None:
    rover = FakeRover(Pose2(0.0, 0.0, 0.0), scale=1.0, drift=0.0)
    guide = Guide(rover)
    guide.set_goal(1.0, 0.0)
    guide.on_state(rover.state(), 0.0)
    guide.on_sighting(sighting(rover.truth, random.Random(1)), 0.0)
    for k in range(5):  # the loop runs faster than GET /state: the state still says "idle"
        guide.step(0.01 * k)
    assert len(rover.paths) == 1
    # the real server answers "idle" with the path still queued for its next loop turn
    queued = rover.state() | {'ctrl': {'mode': 'idle', 'queued': [{'kind': 'path'}]}}
    guide.on_state(queued, 0.1)
    assert guide.step(0.1).startswith('driving') and len(rover.paths) == 1
