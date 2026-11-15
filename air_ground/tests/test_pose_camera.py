"""The air camera's geometry, on a synthetic camera that moves and tilts like a hand-held
phone (or a drone): the floor and the rover must stay where they are."""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import synthetic as syn  # noqa: E402
from pose_camera import Pose, PoseError  # noqa: E402
from rover_marker import RoverFinder  # noqa: E402

# Camera positions and aims (floor frame, metres) and roll (degrees): straight down, then
# moved, lower, higher, tilted up to about 30 degrees, and turned around its axis.
VIEWS = [
    ((0.0, 0.0, 1.8), (0.0, 0.0, 0.0), 0),
    ((0.3, -0.2, 1.6), (0.3, -0.2, 0.0), 40),
    ((-0.4, 0.3, 2.1), (-0.1, 0.1, 0.0), -70),
    ((0.6, 0.5, 1.5), (0.1, 0.0, 0.0), 15),
    ((-0.7, -0.6, 1.9), (0.2, -0.1, 0.0), 160),
    ((0.1, 0.9, 1.7), (0.2, 0.0, 0.0), -120),
]


def pose_of(view, **kw) -> Pose:
    eye, target, roll = view
    return Pose.from_arkit(syn.arkit_pose(eye, target, roll, **kw))


def test_projection_follows_apples_camera_convention() -> None:
    rng = np.random.default_rng(1)
    for view in VIEWS:
        d = syn.arkit_pose(*view)
        pose = Pose.from_arkit(d)
        for p in rng.uniform([-0.8, -0.8, 0.0], [0.8, 0.8, 0.3], (20, 3)):
            assert pose.project(p)[0] == pytest.approx(syn.arkit_project(d, p), abs=1e-6)


def test_looking_straight_down_and_pixels_back_to_the_floor() -> None:
    pose = pose_of(VIEWS[0])
    assert pose.c == pytest.approx([0, 0, 1.8]) and pose.tilt_deg == pytest.approx(0, abs=1e-6)
    assert pose.project([0, 0, 0])[0] == pytest.approx([640, 360])
    assert pose.pixel_to_floor([[640, 360]])[0] == pytest.approx([0, 0], abs=1e-9)
    # 1000 px of focal length at 1.8 m: 1.8 mm per pixel on the floor, 1.68 mm on a 12 cm roof
    assert np.linalg.norm(pose.pixel_to_floor([[740, 360]])[0]) == pytest.approx(0.18)
    assert np.linalg.norm(pose.pixel_to_floor([[740, 360]], 0.12)[0]) == pytest.approx(0.168)
    tilted = pose_of(VIEWS[4])
    assert tilted.tilt_deg == pytest.approx(28.5, abs=0.1)
    for p in ([0.5, -0.3, 0.0], [-0.2, 0.4, 0.12]):
        assert tilted.pixel_to_floor(tilted.project(p), p[2])[0] == pytest.approx(p[:2], abs=1e-9)


def test_floor_height_comes_with_the_pose_and_is_flagged_when_unknown() -> None:
    known = pose_of(VIEWS[1])
    assert known.floor_known and known.c[2] == pytest.approx(1.6)
    d = syn.arkit_pose(*VIEWS[1])
    d['floor_y'] = None  # ARKit has not found the floor yet
    unknown = Pose.from_arkit(d)
    assert not unknown.floor_known and unknown.c[2] == pytest.approx(1.6 + syn.FLOOR_Y)
    assert Pose.from_arkit(d, floor_y=syn.FLOOR_Y).c[2] == pytest.approx(1.6)
    with pytest.raises(PoseError, match='incomplete'):
        Pose.from_arkit({'k': [1, 1, 0, 0]})
    with pytest.raises(PoseError, match='not above'):
        unknown.floor_view(np.zeros((720, 1280, 3), np.uint8), (-1, -1, 1, 1), 0.01, z=0.3)


def test_above_the_horizon_is_not_on_the_floor() -> None:
    level = Pose.from_arkit(syn.arkit_pose((0, 0, 1.5), (5, 0, 1.5)))  # looking at the wall
    xy = level.pixel_to_floor([[640, 100], [640, 700]])
    assert np.isnan(xy[0]).all() and xy[1][0] > 0  # top of the picture: sky; bottom: floor ahead
    assert np.isnan(level.project([-1, 0, 0])).all()  # behind the camera


def test_the_top_view_is_the_same_from_every_pose() -> None:
    floor = syn.floor_texture()
    bounds, res = (-0.6, -0.5, 0.6, 0.5), 0.004
    views = []
    for view in VIEWS:
        pose = pose_of(view)
        top = pose.floor_view(syn.render(pose, floor), bounds, res)
        views.append((top.astype(np.int16), pose.seen_mask(bounds, res)))
    ref, ref_seen = views[0]
    assert ref.shape == (250, 300, 3) and ref_seen.mean() > 0.95
    for top, seen in views[1:]:
        both = cv2.erode((seen & ref_seen).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        assert both.mean() > 0.5
        # same floor, same pixels: only resampling blur at the edges of the boxes differs
        assert np.abs(top - ref)[both].mean() < 6
        assert (np.abs(top - ref).max(axis=2)[both] > 60).mean() < 0.03
    # the opposite check: a top view shifted by 2 cm does NOT match
    shifted = np.roll(views[1][0], 5, axis=1)
    both = views[1][1] & ref_seen
    assert np.abs(shifted - ref)[both].mean() > 2 * np.abs(views[1][0] - ref)[both].mean()


def test_the_rover_stays_put_while_the_camera_moves() -> None:
    floor = syn.floor_texture()
    side, height = 0.10, 0.12
    x, y, forward = 0.25, -0.15, math.radians(40)
    # the top of the printed marker points to the rover's front: its top edge is 90 deg right
    printed = syn.marker_image()  # the team's AprilTag 25h9 id 10
    marker = (printed, x, y, forward - math.pi / 2, side * 1.5, height)  # 1.5: white margin
    finder = RoverFinder(size=side, height=height)
    flat = RoverFinder(size=side, height=0.0)  # forgetting the roof's height
    errors, flat_errors = [], []
    for view in VIEWS:
        pose = pose_of(view)
        image = syn.render(pose, floor, marker)
        seen = finder(image, pose)
        assert seen is not None, view
        errors.append(math.hypot(seen.x - x, seen.y - y))
        turn = math.degrees(
            math.atan2(math.sin(seen.heading - forward), math.cos(seen.heading - forward))
        )
        assert abs(turn) < 2.0, (view, turn)
        assert abs(seen.scale_error) < 0.03 and seen.side_px > 40
        wrong = flat(image, pose)
        flat_errors.append(math.hypot(wrong.x - x, wrong.y - y))
    assert max(errors) < 0.005, errors  # within 5 mm from every pose
    assert max(flat_errors) > 0.04  # the roof's height matters: centimetres off without it
    assert finder(syn.render(pose_of(VIEWS[0]), floor), pose_of(VIEWS[0])) is None  # no rover
    assert (
        RoverFinder(marker_id=7)(syn.render(pose_of(VIEWS[0]), floor, marker), pose_of(VIEWS[0]))
        is None
    )


def test_the_marker_drawn_by_hand_on_the_real_rover() -> None:
    """A photo of the rover's roof: an AprilTag 25h9 id 10 drawn with a pen on squared paper."""
    photo = cv2.imread(os.path.join(HERE, 'data', 'drawn_marker_on_rover.png'))
    quad = RoverFinder().corners(photo)
    assert quad is not None
    side = np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
    assert 280 < side < 360  # the black square, about 320 px wide in that photo
    assert RoverFinder(dictionary='DICT_4X4_50', marker_id=0).corners(photo) is None
