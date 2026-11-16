"""The iPhone's LiDAR (ARKit sceneDepth, sent with the pose of its frame on /depth): heights
measured straight away, without waiting for two views; the guide feeds it to the map."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import synthetic as syn  # noqa: E402
from camstream import DepthMap  # noqa: E402
from floor_map import FloorMap  # noqa: E402
from guidance import Pose2  # noqa: E402
from pose_camera import Pose  # noqa: E402

BOX = (0.20, 0.10, 0.45, 0.35, 0.15)  # x0, y0, x1, y1, height
LOW = (-0.50, -0.40, -0.30, -0.20, 0.02)  # a flat mat: not an obstacle


def pose_above(eye=(0.0, -0.3, 1.3), target=(0.0, 0.1, 0.0), t=0.0) -> Pose:
    d = syn.arkit_pose(eye, target, roll_deg=10, t=t)
    d['sid'] = 5
    return Pose.from_arkit(d)


def at(fmap: FloorMap, layer: np.ndarray, x: float, y: float) -> bool:
    c, r = fmap._cells(np.asarray([[x, y]]))[0].astype(int)
    return bool(layer[r, c])


def test_one_depth_map_puts_the_box_on_the_map_and_not_the_mat() -> None:
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    pose = pose_above()
    metres, k = syn.render_depth(pose, [BOX, LOW])
    out = fmap.add_depth(metres, k, pose)
    assert out['used'] and out['points'] > 10000
    height, known, ob = fmap.obstacles()
    assert at(fmap, ob, 0.32, 0.22)  # the box, from one depth map
    c, r = fmap._cells(np.asarray([[0.32, 0.22]]))[0].astype(int)
    assert abs(height[r, c] - 0.15) < 0.02  # measured, in metres
    assert not at(fmap, ob, -0.40, -0.30)  # 2 cm high: floor
    assert not at(fmap, ob, 0.0, 0.0) and at(fmap, known, 0.0, 0.0)
    assert fmap.stats()['depth_maps'] == 1


def test_the_rover_is_left_out_and_a_moved_box_fades() -> None:
    fmap = FloorMap((-1.0, -1.0, 1.0, 1.0), res=0.01)
    pose = pose_above()
    rover = (-0.2, 0.3, 0.12, 0.26)
    metres, k = syn.render_depth(pose, [BOX], rover=rover)
    fmap.add_depth(metres, k, pose, rover_xy=rover[:2])
    _, _, ob = fmap.obstacles()
    assert not at(fmap, ob, -0.2, 0.3)  # the rover itself: never an obstacle
    metres, k = syn.render_depth(pose, [])  # the box taken away
    for _ in range(12):
        fmap.add_depth(metres, k, pose)
    _, _, ob = fmap.obstacles()
    assert not at(fmap, ob, 0.32, 0.22)


class StubDepth:
    """Stands for camstream.DepthPoller: the LiDAR of a phone held above the scene."""

    def __init__(self, pose, boxes):
        self.pose, self.boxes = pose, boxes
        self.count, self.error = 0, None

    def latest(self, max_age=None):
        self.count += 1
        metres, k = syn.render_depth(self.pose, self.boxes)
        return DepthMap(metres, k, self.pose, None, 20, time.monotonic())

    def close(self):
        pass


def test_the_guide_maps_from_the_phones_lidar() -> None:
    from guide_rover import GuideService
    from test_guidance import FakeRover

    rover = FakeRover(Pose2(-0.5, -0.6, math.radians(90)), scale=1.0, drift=0.0)
    s = GuideService('http://127.0.0.1:9', rover, finder=None, port=0, beam=False,
                     bounds=(-1.0, -1.0, 1.0, 1.0), res=0.01, colour=False)  # fmt: skip
    s.stream.close()
    s.depth = StubDepth(pose_above(), [BOX])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and s.mapper.fmap.depth_maps < 2:
            s._use_depth()
            time.sleep(0.05)
        assert s.mapper.fmap.depth_maps >= 2
        assert s.state()['lidar']['maps'] >= 2
        grid = s.mapper.grid(0.05)
        cells = np.asarray(grid['cells'])
        c = int((0.32 - grid['x0']) / 0.05)
        r = int((0.22 - grid['y0']) / 0.05)
        assert cells[r, c] == 1  # in the team's grid: row 0 = south
        assert s.mapper.heights_picture is not None
    finally:
        s.mapper.close()
        s.server.server_close()
