"""Geometry of rovermap/perception.py on synthetic depth (numpy + OpenCV; the floor-fitting test
needs the mapper's torch import and is skipped without it).

Run: /home/romaindcrl/rovermap/.venv/bin/python -m pytest rover_mapper/test_perception.py -q
(or with the repository's integration venv: the torch-free tests run).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rovermap import perception as per  # noqa: E402

W, H = 320, 240
K = np.array([[160.0, 0, W / 2], [0, 160.0, H / 2], [0, 0, 1.0]])
CAM_H = 0.22


def render(boxes=(), cam_h=CAM_H, pitch_deg=0.0, scale=1.0):
    """Z-depth of a level floor (and boxes [x0, x1, y0, y1, height] in the rover frame) seen by
    a camera `cam_h` above the floor, pitched down by `pitch_deg`; depth divided by `scale`."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    xc, yc = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    p = math.radians(pitch_deg)
    # camera axes in the world (x forward, y left, z up): optical axis pitched down by p
    ez = np.array([math.cos(p), 0, -math.sin(p)])
    ey = np.array([-math.sin(p), 0, -math.cos(p)])  # image down
    ex = np.array([0, -1.0, 0])  # image right
    d = xc[..., None] * ex + yc[..., None] * ey + ez  # ray with optical component 1
    o = np.array([0, 0, cam_h])
    t = np.full((H, W), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = np.where(d[..., 2] < -1e-9, -cam_h / d[..., 2], np.inf)
    t = np.minimum(t, tf)
    for bx0, bx1, by0, by1, bh in boxes:
        lo = np.array([bx0, by0, 0.0])
        hi = np.array([bx1, by1, bh])
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - o) / d
            t2 = (hi - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), -1)
        tmax = np.nanmin(np.maximum(t1, t2), -1)
        hitb = (tmax >= tmin) & (tmin > 0)
        t = np.where(hitb, np.minimum(t, tmin), t)
    t[~np.isfinite(t)] = 0.0  # sky: no depth
    return (t / scale).astype(np.float32)


def level_plane(cam_h=CAM_H, scale=1.0):
    return np.array([0.0, -1.0, 0.0]), cam_h / scale


def cell(p, x, y):
    j = math.floor((x - p.x0) / p.res)
    i = math.floor((y - p.y0) / p.res)
    return int(p.cells[i, j])


def test_floor_only_is_free_near_and_unknown_far_and_behind():
    p = per.perceive(render(), K, plane=level_plane(), max_range=1.5)
    assert p.floor
    assert p.scale == pytest.approx(1.0)
    assert cell(p, 0.6, 0.0) == per.FREE
    assert cell(p, 1.0, 0.1) == per.FREE
    assert (p.cells != per.OBSTACLE).all()
    # outside the field of view (90 deg wide: |y| > x is never seen)
    assert cell(p, 0.3, 0.7) == per.UNKNOWN
    assert p.cells.shape == (60, 30)
    assert p.x0 == 0.0 and p.y0 == pytest.approx(-1.5)


def test_box_is_obstacle_and_hides_what_is_behind():
    box = (0.60, 0.70, -0.10, 0.10, 0.15)
    p = per.perceive(render([box]), K, plane=level_plane(), max_range=1.5)
    assert cell(p, 0.62, 0.0) == per.OBSTACLE
    assert cell(p, 0.62, 0.08) == per.OBSTACLE
    assert cell(p, 0.4, 0.0) == per.FREE
    # straight behind the box: hidden, never free
    for x in (0.8, 1.0, 1.3):
        assert cell(p, x, 0.0) == per.UNKNOWN
    # beside the box the floor stays free
    assert cell(p, 1.0, 0.4) == per.FREE


def test_high_thing_is_passed_under_and_low_bump_ignored():
    shelf = (0.5, 0.7, -0.3, 0.3, 0.8)  # a solid block from the floor to 0.8 m: obstacle
    low = (0.4, 0.5, 0.3, 0.4, 0.02)  # 2 cm: below the band
    p = per.perceive(render([shelf, low]), K, plane=level_plane(), max_range=1.5)
    assert cell(p, 0.52, 0.0) == per.OBSTACLE
    assert cell(p, 0.45, 0.35) == per.FREE


def test_scale_from_camera_height():
    # DA3 depth has an arbitrary unit: the same scene at 1/3 the scale gives the same cells
    box = (0.60, 0.70, -0.10, 0.10, 0.15)
    ref = per.perceive(render([box]), K, plane=level_plane(), max_range=1.5)
    p = per.perceive(render([box], scale=3.0), K, plane=level_plane(scale=3.0), max_range=1.5)
    assert p.scale == pytest.approx(3.0)
    assert (p.cells == ref.cells).all()


def test_pitched_camera_keeps_x_on_the_floor():
    box = (0.80, 0.90, 0.20, 0.30, 0.15)
    pitch = 10.0
    # world up in the axes of a camera pitched down by `pitch`
    n = np.array([0.0, -math.cos(math.radians(pitch)), -math.sin(math.radians(pitch))])
    p = per.perceive(render([box], pitch_deg=pitch), K, plane=(n, CAM_H), max_range=1.5)
    assert p.stats['floor_tilt_deg'] == pytest.approx(pitch, abs=0.01)
    assert cell(p, 0.82, 0.25) == per.OBSTACLE
    assert cell(p, 0.6, 0.0) == per.FREE


def test_no_floor_means_all_unknown():
    p = per.perceive(np.ones((H, W), np.float32), K, plane=None, valid=np.zeros((H, W), bool))
    assert not p.floor
    assert (p.cells == per.UNKNOWN).all()


def test_fitted_floor_and_scale():
    pytest.importorskip('torch')
    pytest.importorskip('depth_anything_3')
    box = (0.60, 0.70, -0.10, 0.10, 0.15)
    p = per.perceive(render([box], scale=2.5), K, cam_height=CAM_H, max_range=1.5)
    assert p.floor
    assert p.scale == pytest.approx(2.5, rel=0.02)
    assert cell(p, 0.62, 0.0) == per.OBSTACLE
    assert cell(p, 0.4, 0.0) == per.FREE
    assert cell(p, 1.0, 0.0) == per.UNKNOWN


def test_draw_runs():
    import cv2

    box = (0.60, 0.70, -0.10, 0.10, 0.15)
    p = per.perceive(render([box]), K, plane=level_plane(), max_range=1.5)
    img = np.zeros((H, W, 3), np.uint8)
    out = per.draw(img, p, K)
    assert out.shape == img.shape
    # the box's cells are drawn red in the middle of the image, below the horizon
    assert out[H // 2 + 10 :, W // 2 - 20 : W // 2 + 20, 2].max() > 50
    assert cv2.imencode('.jpg', out)[0]
