"""The floor map: a phone moved above boxes of known heights, on a textured floor."""

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
from floor_map import FloorMap  # noqa: E402
from planner import RoutePlanner  # noqa: E402
from pose_camera import Pose  # noqa: E402

# two boxes (centre, side, height) and the rover's marker
BOXES = [((0.35, 0.25), 0.25, 0.12), ((-0.30, -0.20), 0.30, 0.25)]
ROVER = (-0.25, 0.35)


def views(n: int = 24, height: float = 1.5):
    """A hand-held phone: a loop above the scene, tilted a little, turning."""
    for k in range(n):
        a = 2 * math.pi * k / n
        eye = (0.30 * math.cos(a), 0.25 * math.sin(a), height + 0.05 * math.sin(3 * a))
        target = (0.10 * math.cos(a + 1), 0.08 * math.sin(a), 0.0)
        d = syn.arkit_pose(eye, target, roll_deg=25 * math.sin(a), t=float(k))
        d['sid'] = 3
        yield Pose.from_arkit(d)


def carpet() -> np.ndarray:
    """The test floor with a carpet's grain: specks of a few millimetres, like the real one."""
    floor = syn.floor_texture().astype(np.int16)
    grain = np.random.default_rng(8).normal(0, 40, floor.shape[:2]).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 1.2)[:, :, None]
    return np.clip(floor + grain, 0, 255).astype(np.uint8)


def scene():
    floor = carpet()
    patches = [
        (syn.box_top(i), x, y, 0.3 * i, side, h) for i, ((x, y), side, h) in enumerate(BOXES)
    ]
    marker = (syn.marker_image(), ROVER[0], ROVER[1], 0.0, 0.14 * 1.5, 0.26)
    return floor, patches, marker


def cell_of(fmap: FloorMap, x: float, y: float) -> tuple[int, int]:
    return round((fmap.bounds[3] - y) / fmap.res - 0.5), round(
        (x - fmap.bounds[0]) / fmap.res - 0.5
    )


def test_heights_of_the_boxes_and_the_floor_from_a_moving_phone() -> None:
    floor, patches, marker = scene()
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7))
    for pose in views():
        fmap.add(syn.render(pose, floor, marker, patches), pose, rover_xy=ROVER)
    height, known = fmap.height_map()
    stats = fmap.stats()
    assert stats['pairs'] >= 20 and stats['seen_m2'] > 1.5

    for (x, y), side, h in BOXES:  # the inner part of each box top
        r, c = cell_of(fmap, x, y)
        k = round(0.3 * side / fmap.res)
        top, ok = height[r - k : r + k, c - k : c + k], known[r - k : r + k, c - k : c + k]
        assert ok.mean() > 0.5, (h, ok.mean())
        assert np.median(top[ok]) == pytest.approx(h, abs=0.035), h

    # the floor away from the boxes and the rover: known, and at floor height
    r, c = cell_of(fmap, 0.30, -0.35)
    k = round(0.1 / fmap.res)
    floor_h, floor_ok = height[r - k : r + k, c - k : c + k], known[r - k : r + k, c - k : c + k]
    assert floor_ok.mean() > 0.6 and (floor_h[floor_ok] < fmap.obstacle_m).mean() > 0.95

    # the rover is left out: no obstacle where it stands
    r, c = cell_of(fmap, *ROVER)
    k = round(0.1 / fmap.res)
    assert not (
        known[r - k : r + k, c - k : c + k] & (height[r - k : r + k, c - k : c + k] > 0.05)
    ).any()

    # the grid in the team's format: 5 cm cells, obstacles where the boxes are
    g = fmap.grid(0.05)
    cells = np.array(g['cells'])
    assert cells.shape == (28, 32) and g['res'] == 0.05
    row = lambda y: int((y - g['y0']) / 0.05)  # noqa: E731  (row 0 = south)
    col = lambda x: int((x - g['x0']) / 0.05)  # noqa: E731
    for (x, y), _, _ in BOXES:
        assert cells[row(y), col(x)] == 1
    assert cells[row(-0.35), col(0.30)] == 0
    assert (cells == -1).any() and (cells == 0).sum() > (cells == 1).sum()

    picture = fmap.render()
    assert picture.shape == (280, 320, 3)
    r, c = cell_of(fmap, *BOXES[1][0])
    assert picture[r, c, 2] > picture[r, c, 0] + 40  # obstacles tinted red


def test_a_still_phone_maps_no_height_and_a_new_session_starts_again() -> None:
    floor, patches, marker = scene()
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7))
    still = next(views())
    for _ in range(5):
        fmap.add(syn.render(still, floor, marker, patches), still)
    assert fmap.stats()['pairs'] == 0 and fmap.stats()['known_m2'] == 0  # no baseline, no height
    assert fmap.stats()['seen_m2'] > 0.5  # but the floor's picture is there

    d = syn.arkit_pose((0, 0, 1.5), (0, 0, 0))
    d['sid'] = 99
    fmap.add(syn.render(Pose.from_arkit(d), floor), Pose.from_arkit(d))
    assert fmap.session == 99 and fmap.views == 1  # the old map belonged to another frame

    d['floor_y'] = None
    assert not fmap.add(syn.render(Pose.from_arkit(d), floor), Pose.from_arkit(d))['used']


def test_the_map_centres_itself_where_the_phone_first_looks() -> None:
    floor, patches, marker = scene()
    fmap = FloorMap((-0.5, -0.5, 0.5, 0.5), follow=True)
    d = syn.arkit_pose((0.6, -0.4, 1.4), (0.7, -0.3, 0.0))
    d['sid'] = 5
    pose = Pose.from_arkit(d)
    fmap.add(syn.render(pose, floor, marker, patches), pose)
    assert fmap.bounds == pytest.approx((0.2, -0.8, 1.2, 0.2), abs=0.02)  # same 1 m size
    assert fmap.stats()['seen_m2'] > 0.5
    fmap.add(syn.render(pose, floor), Pose.from_arkit(d | {'T': d['T']}))
    assert fmap.bounds == pytest.approx((0.2, -0.8, 1.2, 0.2), abs=0.02)  # it stays put


def test_the_teams_colour_detection_adds_what_has_no_height_measured() -> None:
    floor, patches, marker = scene()
    poses = list(views())[:4]  # few views: almost no height measured yet
    heights_only = FloorMap((-0.8, -0.7, 0.8, 0.7))
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7), colour=True)
    for pose in poses:
        for m in (heights_only, fmap):
            m.add(syn.render(pose, floor, marker, patches), pose, rover_xy=ROVER)
    before = np.array(heights_only.grid(0.05)['cells'])
    cells = np.array(fmap.grid(0.05)['cells'])
    row = lambda y: int((y - fmap.bounds[1]) / 0.05)  # noqa: E731  (row 0 = south)
    col = lambda x: int((x - fmap.bounds[0]) / 0.05)  # noqa: E731
    for (x, y), _, _ in BOXES:
        assert cells[row(y), col(x)] == 1  # the boxes differ from the floor's colour
    assert cells[row(ROVER[1]), col(ROVER[0])] != 1  # the rover is left out
    assert (cells == 0).sum() > (before == 0).sum()  # the floor seen counts as free now


def test_the_grid_view_is_the_planners_grid_on_the_same_pixels() -> None:
    floor, patches, marker = scene()
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7))
    for pose in views():
        fmap.add(syn.render(pose, floor, marker, patches), pose, rover_xy=ROVER)
    picture, grid = fmap.render_grid(), fmap.render()
    assert picture.shape == grid.shape
    r, c = cell_of(fmap, *BOXES[1][0])
    assert tuple(picture[r + 2, c + 2]) == (40, 40, 190)  # an obstacle cell: dark red


def test_the_moving_rover_leaves_no_trail_of_obstacles() -> None:
    """The rover drives across the floor while the phone moves above it: where it has been
    must not turn into obstacles (it used to be painted into the floor's picture at every
    place, and the colour detection then saw a trail of rover)."""
    floor = carpet()
    body = np.full((200, 200, 3), (30, 30, 35), np.uint8)  # the rover's dark roof
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7), colour=True)
    track = [(-0.45 + 0.03 * k, -0.25) for k in range(30)]  # 0.9 m across
    for k, (x, y) in enumerate(track):
        a = 2 * math.pi * k / 30
        d = syn.arkit_pose((0.2 * math.cos(a), 0.15 * math.sin(a), 1.5), (0.0, 0.0, 0.0),
                           roll_deg=10 * math.sin(a), t=float(k))  # fmt: skip
        d['sid'] = 4
        pose = Pose.from_arkit(d)
        rover = [(body, x, y, 0.0, 0.25, 0.25), (syn.marker_image(), x, y, 0.0, 0.14 * 1.5, 0.26)]
        fmap.add(syn.render(pose, floor, None, rover), pose, rover_xy=(x, y))
    cells = np.array(fmap.grid(0.05)['cells'])
    row = lambda y: int((y - fmap.bounds[1]) / 0.05)  # noqa: E731  (row 0 = south)
    col = lambda x: int((x - fmap.bounds[0]) / 0.05)  # noqa: E731
    trail = {(row(y), col(x)) for x, y in track[:-8]}  # where it was, not where it is
    assert not any(cells[r, c] == 1 for r, c in trail)
    assert sum(cells[r, c] == 0 for r, c in trail) > len(trail) // 2  # seen as floor


def plain_carpet() -> np.ndarray:
    """A carpet of one colour, with its grain only (no coloured squares on it)."""
    rng = np.random.default_rng(8)
    n = round(3.0 / syn.FLOOR_RES)
    grain = cv2.GaussianBlur(rng.normal(0, 40, (n, n)).astype(np.float32), (0, 0), 1.2)
    base = np.full((n, n, 3), (70, 120, 160), np.float32)
    return np.clip(base + grain[:, :, None], 0, 255).astype(np.uint8)


def test_tall_boxes_do_not_grow_until_the_passage_closes() -> None:
    """Two boxes 30 cm high, 45 cm apart: room for the rover (2 x 15 cm). The phone wanders
    farther and farther from the middle, seeing them more and more from the side, where they
    land farther out once laid flat. The map must keep their size and the passage open
    (adding up all those views closed it after about 140 views)."""
    floor = plain_carpet()
    boxes = [
        (syn.box_top(1), -0.35, 0.0, 0.0, 0.25, 0.30),
        (syn.box_top(2), 0.35, 0.0, 0.0, 0.25, 0.30),
    ]
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7), res=0.01, colour=True)
    planner = RoutePlanner(0.15)
    sizes = []
    for k in range(160):
        a = 2 * math.pi * k / 40
        r = 0.15 + 0.45 * k / 160
        d = syn.arkit_pose((r * math.cos(a), 0.8 * r * math.sin(a), 1.4), (0.0, 0.0, 0.0),
                           roll_deg=15 * math.sin(a), t=float(k))  # fmt: skip
        d['sid'] = 5
        pose = Pose.from_arkit(d)
        fmap.add(syn.render(pose, floor, None, boxes), pose)
        if k % 20 == 19:
            g = fmap.grid(fmap.cell)  # the grid the route is planned on
            sizes.append(float((np.array(g['cells']) == 1).sum()) * fmap.cell**2)
            route = planner.route(g, (0.0, -0.45), (0.0, 0.45))
            assert route is not None and len(route) <= 2, (k, sizes)  # straight through
    # about their true size (2 x 0.0625 m2), no halo, and no creeping growth once built
    assert 0.10 <= sizes[2] <= 0.16 and max(sizes) <= 0.20, sizes


def test_the_rover_is_never_an_obstacle_even_unseen() -> None:
    """The rover as an obstacle on its own map: when its marker is not read in a picture,
    or its parts stick out. Its place (from the guide) frees the patch it stands in, and
    where it drove becomes floor; a speck elsewhere is dropped, a box is kept."""
    fmap = FloorMap((-0.8, -0.7, 0.8, 0.7), res=0.01)
    k = fmap.colour_k

    def cell(x, y):
        c, r = fmap._cells(np.asarray([[x, y]], float))[0] / k
        return int(r), int(c)

    for x, y in ((0.0, 0.0), (0.03, 0.0), (0.0, 0.03), (0.03, 0.03)):  # the rover, 6 x 6 cm
        fmap.height_hits[cell(x, y)] = 3
    for dx in range(6):  # a box, 18 x 18 cm, 40 cm away
        for dy in range(6):
            fmap.height_hits[cell(0.4 + 0.03 * dx, 0.03 * dy)] = 3
    fmap.height_hits[cell(-0.5, -0.4)] = 3  # a speck
    _, _, before = fmap.obstacles()
    assert before[fmap._cells(np.asarray([[0.01, 0.01]]))[0][::-1].astype(int).tolist()[0],
                  fmap._cells(np.asarray([[0.01, 0.01]]))[0].astype(int)[0]]  # fmt: skip
    fmap.set_rover(0.02, 0.02)
    _, _, after = fmap.obstacles()
    rc, rr = fmap._cells(np.asarray([[0.02, 0.02]]))[0].astype(int)
    bc, br = fmap._cells(np.asarray([[0.47, 0.07]]))[0].astype(int)
    sc, sr = fmap._cells(np.asarray([[-0.49, -0.39]]))[0].astype(int)
    assert not after[rr, rc] and after[br, bc] and not after[sr, sc]
    assert fmap.height_hits[cell(0.0, 0.0)] == 0  # where it drove: floor until seen again
