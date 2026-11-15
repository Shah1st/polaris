"""Tests of the feeding side of the live fusion (sim/fusion_feed.py), on synthetic scenes.

pytest sim/test_fusion_feed.py
"""

import math
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'merging_map'))

from fusion_carte import CarteFusion, decouper  # noqa: E402

import fusion_feed as ff  # noqa: E402
from fusion_live import remaining_path  # noqa: E402

H = ff.CAM_HEIGHT
K = np.array([[200.0, 0, 79.5], [0, 200.0, 59.5], [0, 0, 1]])
W, HH = 160, 120


def camera_pose(x=0.0, y=0.0, ground=0.0):
    """Camera at height H above `ground`, level, looking along +x (OpenCV axes)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = (x, y, ground + H)
    return T


def render(T, wall=None, drop_at=None, drop=1.0):
    """Z-depth of a flat ground (z = 0), optionally a wall (x = wall[0], |y| < wall[1], up to
    wall[2] m) and a drop of `drop` m for x > drop_at. Sky: 300 m."""
    vs, us = np.mgrid[0:HH, 0:W].astype(float)
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    d = rays @ T[:3, :3].T
    c = T[:3, 3]
    depth = np.full((HH, W), 300.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(d[..., 2] < 0, -c[2] / d[..., 2], np.inf)
        if drop_at is not None:
            hit_x = c[0] + d[..., 0] * s
            low = np.where(d[..., 2] < 0, -(c[2] + drop) / d[..., 2], np.inf)
            s = np.where(hit_x > drop_at, low, s)
            # the lower ground is not reached before the edge: the ray passes over it
        depth = np.where(np.isfinite(s), s, depth)
        if wall is not None:
            sw = (wall[0] - c[0]) / d[..., 0]
            p = c + d * sw[..., None]
            on = (
                (sw > 0)
                & (np.abs(p[..., 1]) < wall[1])
                & (p[..., 2] >= 0)
                & (p[..., 2] <= wall[2])
            )
            depth = np.where(on & (sw < depth), sw, depth)
    return depth.astype(np.float32)


def cell(local, x, y):
    i = math.floor(y / ff.RES) - round(local.y0 / ff.RES)
    j = math.floor(x / ff.RES) - round(local.x0 / ff.RES)
    if not (0 <= i < local.data.shape[0] and 0 <= j < local.data.shape[1]):
        return ff.UNSEEN  # outside the scan's window
    return int(local.data[i, j])


def test_drone_png_rows_are_flipped_to_south_first():
    # The simulator draws grid[i_x, j_y] as picture grid.T[::-1]: picture row 0 = north.
    grid = np.zeros((6, 4), np.int8)  # 6 cells along x, 4 along y
    grid[2, 0] = 2  # obstacle at x index 2, y index 0 (south)
    grid[5, 3] = -1  # not seen, north-east corner
    pic = np.where(grid < 0, 128, np.where(grid > 0, 255, 0)).astype(np.uint8).T[::-1]
    m = ff.drone_grid_from_png(cv2.imencode('.png', pic)[1].tobytes())
    assert m.shape == (4, 6)
    assert m[0, 2] == ff.OBSTACLE  # row 0 = south, column = x
    assert m[3, 5] == ff.UNSEEN
    assert (m == ff.FREE).sum() == 22


def test_flat_ground_with_a_wall():
    T = camera_pose()
    scan = ff.scan_from_depth(render(T, wall=(1.02, 0.2, 0.25)), K, T, (0.0, 0.0, 0.0), 1.0)
    assert scan.points[:, 2].min() > -0.01
    local = ff.classify_scan(scan)
    assert cell(local, 0.8, 0.0) == ff.FREE
    assert cell(local, 1.02, 0.0) == ff.OBSTACLE  # wall face
    assert cell(local, 1.5, 0.0) == ff.UNSEEN  # hidden behind the wall
    assert cell(local, 1.5, 0.45) == ff.FREE  # beside it
    assert cell(local, 2.06, 0.0) == ff.UNSEEN  # beyond the 2 m range
    assert cell(local, -1.0, 0.0) == ff.UNSEEN  # behind the camera
    assert cell(local, 0.0, 0.0) == ff.FREE  # under the rover
    assert local.stats['drop'] == 0


def test_low_vegetation_is_driven_through_but_a_grey_block_is_not():
    T = camera_pose()
    depth = render(T, wall=(1.02, 0.2, 0.25))
    brown = np.full((HH, W, 3), (100, 85, 60), np.uint8)  # RGB, soil
    block = depth < 1.03  # pixels on the block's face (the ground that near is below the image)
    assert block.any()
    green = brown.copy()
    green[block] = (60, 110, 30)  # fern
    grey = brown.copy()
    grey[block] = (70, 68, 66)  # rock
    pose = (0.0, 0.0, 0.0)
    fern = ff.classify_scan(ff.scan_from_depth(depth, K, T, pose, 1.0, rgb=green))
    rock = ff.classify_scan(ff.scan_from_depth(depth, K, T, pose, 1.0, rgb=grey))
    assert cell(fern, 1.02, 0.0) == ff.FREE
    assert fern.stats['vegetation_points'] > 0
    assert (fern.point_kind == ff.POINT_VEGETATION).sum() == fern.stats['vegetation_points']
    assert cell(rock, 1.02, 0.0) == ff.OBSTACLE
    assert rock.stats['vegetation_points'] == 0
    # switched off, or without a picture, green changes nothing
    off = ff.classify_scan(ff.scan_from_depth(depth, K, T, pose, 1.0, rgb=green), vegetation=0)
    assert cell(off, 1.02, 0.0) == ff.OBSTACLE
    plain = ff.classify_scan(ff.scan_from_depth(depth, K, T, pose, 1.0))
    assert cell(plain, 1.02, 0.0) == ff.OBSTACLE


def test_branches_above_the_rover_do_not_block():
    T = camera_pose()
    # a wall from 0.5 to 2 m high: the rover passes under it, its foot is not seen
    depth = render(T)
    vs, us = np.mgrid[0:HH, 0:W].astype(float)
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    z = H - ((vs - K[1, 2]) / K[1, 1]) * 1.0  # height of the ray at x = 1 m
    over = (z > 0.5) & (z < 2.0) & (np.abs(rays[..., 0]) < 0.3)
    depth = np.where(over, 1.0, depth).astype(np.float32)
    local = ff.classify_scan(ff.scan_from_depth(depth, K, T, (0.0, 0.0, 0.0), 1.0))
    assert cell(local, 1.0, 0.0) != ff.OBSTACLE


def test_drop_edge_is_an_obstacle_but_the_near_ground_is_free():
    T = camera_pose()
    scan = ff.scan_from_depth(render(T, drop_at=1.0), K, T, (0.0, 0.0, 0.0), 1.0)
    local = ff.classify_scan(scan)
    assert cell(local, 0.8, 0.0) == ff.FREE
    assert cell(local, 1.3, 0.0) == ff.OBSTACLE  # where the ground should be, but is not
    assert local.stats['drop'] > 0


def test_rover_tiles_land_at_the_right_place_in_the_fusion():
    # Negative coordinates, as in the forest (origin -32, -28): the wall must appear where it is.
    x0, y0 = -12.3, -7.9
    T = camera_pose(x0, y0)
    # the same picture as a camera at the origin with a wall 1.02 m ahead: it is at x0 + 1.02
    local_depth = render(camera_pose(), wall=(1.02, 0.2, 0.25))
    scan = ff.scan_from_depth(local_depth, K, T, (x0, y0, 0.0), 5.0)
    local = ff.classify_scan(scan)
    carte = CarteFusion()
    for msg in decouper(local.data, local.x0, local.y0, carte.zmax, carte, 'rover', 5.0):
        carte.recevoir(msg)
    g = carte.grille(x0 - 3, y0 - 3, x0 + 3, y0 + 3)

    def val(x, y):
        return int(g.val[math.floor(y / g.res) - g.i0, math.floor(x / g.res) - g.j0])

    assert val(x0 + 1.02, y0) == ff.OBSTACLE
    assert val(x0 + 0.8, y0) == ff.FREE
    assert val(x0 - 1.0, y0) == ff.UNSEEN
    assert (g.src[g.val >= 0] == 2).all()  # all from the rover


def test_depth_stats():
    truth = np.array([[0.5, 0.5, 2.0, 0.2]])
    sensor = np.array([[0.51, 0.0, 0.0, 0.2]])
    s = ff.depth_stats(sensor, truth, (0.07, 1.0))
    assert s['mesures_pct'] == 50.0
    assert s['mesures_dans_portee_pct'] == pytest.approx(66.7)
    assert s['erreur_mediane_m'] == pytest.approx(0.005)
    assert s['pixels_communs'] == 2


def test_view_is_north_up():
    cls = np.zeros((4, 4), np.uint8)
    cls[0, 0] = ff.K_DRONE_OBST  # south-west cell
    view = ff.View(0.0, 0.0, 0.2, 0.2, 100.0)  # 20 x 20 px, 5 px per cell
    img = view.raster(cls, 0.0, 0.0)
    assert (img[-1, 0] == ff.DRONE_OBST_C).all()  # bottom-left pixel
    assert (img[0, 0] == ff.UNKNOWN_C).all()
    assert view.px(0.0, 0.2) == (0, 0)


def test_remaining_path_skips_the_rover_surroundings():
    path = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0)]
    ahead, off = remaining_path(path, (0.45, 0.2))
    assert off == pytest.approx(math.hypot(0.05, 0.2))
    assert ahead == [(1.0, 0.0), (2.0, 0.0)]
    assert remaining_path([], (0, 0)) == ([], math.inf)


def test_png_round_trip():
    img = np.zeros((10, 12, 3), np.uint8)
    img[2:5, 3:7] = ff.ROVER_OBST_C
    back = cv2.imdecode(np.frombuffer(ff.encode_png(img), np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == img.shape
    assert (back == img).all()
