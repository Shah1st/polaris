"""The map of the floor seen from the air, which grows and firms up as the camera moves.

Two layers, both in the floor frame of pose_camera.py:

- the picture of the floor (`mosaic`): each new view is laid flat on the floor; a place keeps
  the sharpest view of it (the one taken most from above), slowly giving way to newer ones;
- the height of what stands on the floor (`heights`): what stands above the floor shifts from
  one view to the next once laid flat, the floor does not. Two views a few tens of centimetres
  apart are laid on planes at several heights (plane sweep); where they agree tells the height
  of the surface there. Each pair of views adds votes; the map keeps the height with the most
  votes. No model and no colour rule: a box is an obstacle because it is high, whatever its
  colour or the floor's. A surface with no texture (a plain box top) gives no vote: it stays
  unknown until its edges or another view decide;
- with a LiDAR iPhone, the heights measured directly (`add_depth`): each depth map, placed by
  the pose of its frame, gives the height of every point it sees, in 2 cm cells, averaged over
  the maps. Where it has measured, it wins over the plane sweep. The floor's height found by
  ARKit is checked against the points the LiDAR sees near it (`floor_offset`).

    fmap = FloorMap((-1.5, -1.5, 1.5, 1.5))
    fmap.add(image, pose, rover_xy=(x, y))   # the rover is a moving obstacle: left out
    fmap.add_depth(metres, k, pose)          # a LiDAR depth map and the pose of its frame
    picture = fmap.render()                  # BGR top view: floor, obstacles in red, unknown dark
    cells = fmap.grid(0.05)                  # -1 unknown / 0 free / 1 obstacle, row 0 = south
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

import cv2
import numpy as np
from pose_camera import Pose

HEIGHTS = (0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.36)  # m, the swept planes
CELL = 0.03  # m, the cells the map decides on (obstacle or floor); the team's grid is 5 cm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _team_colour_detection():
    """The team's colour segmentation and its filters, imported when first needed."""
    for sub in ('src/guidance_global_map', 'merging_map', 'sim'):
        path = os.path.join(ROOT, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    from fusion_reelle import part_ombre, sans_traits

    from guidance_global_map.segmentation import segment_obstacles

    return segment_obstacles, part_ombre, sans_traits


def to_cell_scale(image: np.ndarray, pose: Pose, res: float) -> tuple[np.ndarray, Pose]:
    """The picture halved (with a blur first) until one of its pixels is about one cell of
    the map on the floor, and the pose with the intrinsics to match. Without it, a floor with
    a grain finer than a cell (a carpet) is sampled one point per cell: the two views of a
    pair then show different noise, and the heights come out wrong."""
    gsd = float(pose.c[2]) / float(pose.k[0, 0])  # metres of floor per pixel
    halvings = int(np.floor(np.log2(max(res / gsd, 1.0))))
    k = pose.k.copy()
    for _ in range(halvings):
        image = cv2.pyrDown(image)
        k[0, 0] /= 2
        k[1, 1] /= 2
        k[0, 2] = (k[0, 2] + 0.5) / 2 - 0.5
        k[1, 2] = (k[1, 2] + 0.5) / 2 - 0.5
    if not halvings:
        return image, pose
    return image, replace(pose, k=k, width=image.shape[1], height=image.shape[0])


@dataclass
class Keyframe:
    pose: Pose
    gray: np.ndarray
    rover_xy: tuple[float, float] | None = None  # where the rover was in that view


class FloorMap:
    def __init__(
        self,
        bounds=(-1.5, -1.5, 1.5, 1.5),
        res: float = 0.005,
        heights=HEIGHTS,
        obstacle_m: float = 0.045,
        window: int = 7,
        min_ncc: float = 0.6,
        min_margin: float = 0.12,
        min_texture: float = 25.0,
        keyframe_step: float = 0.06,
        keyframes: int = 10,
        decay: float = 0.95,
        rover_radius: float = 0.22,
        rover_height: float = 0.26,
        view_aging: float = 0.03,
        cell_aging: float = 0.004,
        max_votes: float = 4.0,
        colour: bool = False,
        cell: float = CELL,
        follow: bool = False,
        depth_res: float = 0.02,
        min_obstacle_m: float = 0.04,
    ) -> None:
        """`res`: metres per cell. `obstacle_m`: a surface higher than this is an obstacle.
        `window`: cells of the matching window. `min_ncc`: how well two views must agree at
        the chosen height; `min_margin`: by how much better than at the other heights.
        `min_texture`: grey-level variance below which a window says nothing. `keyframe_step`:
        metres the camera moves between two kept views. `decay`: what one update leaves of the
        older votes (an obstacle taken away fades out). `follow`: the map is centred on the
        point the camera looks at in the first view after a reset or a new floor frame (the
        floor frame's origin is where the phone started tracking, maybe metres away); `bounds`
        then only gives the map's size."""
        self.bounds = tuple(float(v) for v in bounds)
        self.res = res
        x0, y0, x1, y1 = self.bounds
        self.cols, self.rows = round((x1 - x0) / res), round((y1 - y0) / res)
        self.heights = np.asarray(heights, np.float32)
        self.obstacle_m = obstacle_m
        self.window = window
        self.min_ncc, self.min_margin, self.min_texture = min_ncc, min_margin, min_texture
        self.keyframe_step, self.max_keyframes = keyframe_step, keyframes
        self.decay = decay
        self.rover_radius = rover_radius
        self.rover_size = 0.35  # m: a patch touching the rover and no larger is the rover
        self.rover_body = 0.10  # m: around its centre, what it drives over is floor
        self.rover_height = rover_height  # its top: where it leans to once laid flat
        # what each new view takes off the advantage of the views kept in the floor's
        # picture: 0.03 = after about 10 views, the current one wins wherever it sees well
        self.view_aging = view_aging
        # the same for what each cell is (obstacle or floor), much slower: the view that
        # saw a cell most from above keeps speaking for it (a view 0.2 more to the side takes
        # over after about 50 views, some 10 s); views from the side, which see tall things
        # lean and box edges spill over, take over only when nothing better comes
        self.cell_aging = cell_aging
        # a view says nothing of a cell seen farther off its vertical than this (distance over
        # height: 0.5 = about 27 degrees)
        self.max_side = 0.5
        # the votes of a cell are scaled down past this total: new evidence can still turn
        # it within a few pairs, instead of fighting all the votes of the past
        self.max_votes = max_votes
        # views of the air after which it may contradict what the D405 said of a cell
        self.beam_hold = 15
        # the team's colour detection on each view (off: heights only)
        self.colour_on = colour
        # The cells the map decides on: obstacle or floor. Smaller than the team's 5 cm (the
        # export, `grid(0.05)`, is still in 5 cm): a cell touched by an obstacle is an
        # obstacle, so its size is margin added to the rover's radius; 3 cm cells add 1.5 cm
        # on average instead of 2.5.
        self.colour_k = max(1, round(cell / res))
        self.cell = self.colour_k * res
        self.crows = -(-self.rows // self.colour_k)
        self.ccols = -(-self.cols // self.colour_k)
        self.follow = follow
        self.min_obstacle_m = min_obstacle_m
        self.depth_f = max(1, round(depth_res / res))  # map cells per depth cell, each way
        self.depth_res = self.depth_f * res
        self.drows = -(-self.rows // self.depth_f)
        self.dcols = -(-self.cols // self.depth_f)
        self.reset()

    def reset(self) -> None:
        self.mosaic = np.zeros((self.rows, self.cols, 3), np.uint8)
        self.view_score = np.full((self.rows, self.cols), np.inf, np.float32)
        self.votes = np.zeros((len(self.heights), self.rows, self.cols), np.float32)
        self.keyframes: list[Keyframe] = []
        self.session: int | None = None
        self.views = 0
        self.pairs = 0
        self.dheight = np.zeros((self.drows, self.dcols), np.float32)
        self.dweight = np.zeros((self.drows, self.dcols), np.float32)
        self.depth_maps = 0
        self.floor_offset = 0.0  # m: where the LiDAR sees the floor, against ARKit's floor
        # the team's colour detection, per decision cell: how far from straight below the view
        # that last spoke for each cell saw it, and how many views in a row saw an obstacle
        self.rover_now: tuple[float, float] | None = None  # the rover's centre (set_rover)
        self.colour_q = np.full((self.crows, self.ccols), np.inf, np.float32)
        self.colour_hits = np.zeros((self.crows, self.ccols), np.int8)
        # the same for the heights of the pairs of views
        self.height_q = np.full((self.crows, self.ccols), np.inf, np.float32)
        self.height_hits = np.zeros((self.crows, self.ccols), np.int8)
        # the rover's D405 (add_beam): -1 nothing said, 0 floor, 1 obstacle, and the view
        # count when it said it
        self.beam_state = np.full((self.crows, self.ccols), -1, np.int8)
        self.beam_view = np.zeros((self.crows, self.ccols), np.int64)
        self.beam_looks = 0
        self._centre_next = self.follow

    def recentre(self, x: float, y: float) -> None:
        """Move the map (same size) so that (x, y) is its centre; it starts empty."""
        x0, y0, x1, y1 = self.bounds
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        self.bounds = (x - hw, y - hh, x + hw, y + hh)
        session = self.session
        self.reset()
        self.session = session
        self._centre_next = False

    # -- frames of the grid ---------------------------------------------------------------

    def _grid_to_floor(self, c0: int, r0: int) -> np.ndarray:
        """3x3 from (col, row) of the grid, counted from (c0, r0), to (x, y, 1) on the floor."""
        x0, _, _, y1 = self.bounds
        res = self.res
        return np.array(
            [[res, 0, x0 + (c0 + 0.5) * res], [0, -res, y1 - (r0 + 0.5) * res], [0, 0, 1.0]]
        )

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        x0, _, _, y1 = self.bounds
        return np.column_stack([(xy[:, 0] - x0) / self.res, (y1 - xy[:, 1]) / self.res])

    def _roi(self, *poses: Pose, z: float = 0.0) -> tuple[int, int, int, int] | None:
        """The grid window (c0, r0, c1, r1) all these views see, or None."""
        box = None
        for p in poses:
            corners = p.footprint(z)
            if not np.isfinite(corners).all():
                return None  # a view that sees the horizon: too slanted to map
            cells = self._cells(corners)
            b = (*cells.min(axis=0), *cells.max(axis=0))
            box = b if box is None else (
                max(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), min(box[3], b[3])
            )  # fmt: skip
        c0, r0 = max(0, int(box[0])), max(0, int(box[1]))
        c1, r1 = min(self.cols, int(np.ceil(box[2]))), min(self.rows, int(np.ceil(box[3])))
        if c1 - c0 < 2 * self.window or r1 - r0 < 2 * self.window:
            return None
        return c0, r0, c1, r1

    def _warp(self, pose: Pose, image, z: float, roi) -> tuple[np.ndarray, np.ndarray]:
        """The picture laid on the plane at height z, on the grid window `roi`, and where it
        is inside the picture."""
        c0, r0, c1, r1 = roi
        m = pose.plane_homography(z) @ self._grid_to_floor(c0, r0)
        size = (c1 - c0, r1 - r0)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        out = cv2.warpPerspective(image, m, size, flags=flags, borderMode=cv2.BORDER_CONSTANT)
        inside = cv2.warpPerspective(
            np.full(image.shape[:2], 255, np.uint8), m, size, flags=flags, borderValue=0
        )
        return out, inside > 250

    # -- adding a view --------------------------------------------------------------------

    def _accept(self, pose: Pose) -> str | None:
        """None when this pose can feed the map (and the map is placed); else why not."""
        if pose.session != self.session:  # the phone's floor frame started again
            self.reset()
            self.session = pose.session
        if not pose.floor_known or pose.tracking != 'normal' or pose.c[2] < 0.3:
            return 'no floor, tracking not normal, or camera too low'
        if self._centre_next:  # where the camera looks now: the middle of the picture
            aim = pose.pixel_to_floor([[pose.k[0, 2], pose.k[1, 2]]])[0]
            if not np.isfinite(aim).all() or np.linalg.norm(aim - pose.c[:2]) > 2 * pose.c[2]:
                aim = pose.c[:2]
            self.recentre(float(aim[0]), float(aim[1]))
        return None

    def add_depth(self, metres: np.ndarray, k, pose: Pose, rover_xy=None, max_range=4.0) -> dict:
        """One LiDAR depth map (metres along the optical axis, 0 = no measure; `k` = fx, fy,
        cx, cy in its pixels) and the camera's pose for that frame. Every point it sees is
        placed on the floor; each 2 cm cell takes the mean height of its points, averaged with
        what earlier maps said (the 8 latest weigh the most: a moved box fades out)."""
        why = self._accept(pose)
        if why:
            return {'used': False, 'why': why}
        fx, fy, cx, cy = (float(v) for v in k)
        h, w = metres.shape
        v, u = np.mgrid[0:h, 0:w]
        d = metres.astype(np.float32)
        ok = np.isfinite(d) & (d > 0.15) & (d < max_range)
        if ok.sum() < 50:
            return {'used': False, 'why': 'almost no depth measured'}
        cam = np.column_stack([(u[ok] - cx) / fx * d[ok], (v[ok] - cy) / fy * d[ok], d[ok]])
        pts = cam @ pose.r.T + pose.c  # the floor frame
        # ARKit's floor plane may be off by a centimetre or two: the floor is where most of
        # the points near z = 0 are.
        near = pts[np.abs(pts[:, 2] - self.floor_offset) < 0.06, 2]
        if near.size > 200:
            self.floor_offset = 0.8 * self.floor_offset + 0.2 * float(np.median(near))
        z = pts[:, 2] - self.floor_offset
        x0, _, _, y1 = self.bounds
        col = np.floor((pts[:, 0] - x0) / self.depth_res).astype(np.int64)
        row = np.floor((y1 - pts[:, 1]) / self.depth_res).astype(np.int64)
        keep = (col >= 0) & (col < self.dcols) & (row >= 0) & (row < self.drows) & (z > -0.1)
        if rover_xy is not None:  # the rover moves: its own points say nothing of the floor
            keep &= np.hypot(pts[:, 0] - rover_xy[0], pts[:, 1] - rover_xy[1]) > self.rover_radius
        idx = row[keep] * self.dcols + col[keep]
        n = self.drows * self.dcols
        count = np.bincount(idx, minlength=n).reshape(self.drows, self.dcols)
        total = np.bincount(idx, weights=z[keep], minlength=n).reshape(self.drows, self.dcols)
        seen = count > 0
        mean = (total[seen] / count[seen]).astype(np.float32)
        wgt = self.dweight[seen]
        self.dheight[seen] = (self.dheight[seen] * wgt + mean) / (wgt + 1)
        self.dweight[seen] = np.minimum(wgt + 1, 8)
        self.depth_maps += 1
        return {'used': True, 'points': int(keep.sum()), 'cells': int(seen.sum())}

    def _depth_layer(self) -> tuple[np.ndarray, np.ndarray]:
        """The LiDAR heights and where they are known, at the map's own cell size."""
        f = self.depth_f
        up = np.repeat(np.repeat(self.dheight, f, axis=0), f, axis=1)[: self.rows, : self.cols]
        known = np.repeat(np.repeat(self.dweight >= 1, f, axis=0), f, axis=1)
        return up, known[: self.rows, : self.cols]

    def add(self, image: np.ndarray, pose: Pose, rover_xy=None) -> dict:
        """One picture and the camera's pose for it. `rover_xy`: where the rover is in that
        picture (floor frame), left out of the heights. Returns what this view did."""
        why = self._accept(pose)
        if why:
            return {'used': False, 'why': why}
        image, pose = to_cell_scale(image, pose, self.res)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        done = {'used': True, 'pairs': 0}
        self._lay_flat(image, pose, rover_xy)
        if self.colour_on:
            self._colour_view(image, pose, rover_xy)

        # Heights: against the kept views far enough from this one (a few tens of cm).
        h = float(pose.c[2])
        wanted = min(0.5, 0.3 * h)
        partners = sorted(
            (k for k in self.keyframes if 0.08 <= np.linalg.norm(k.pose.c - pose.c) <= 0.8),
            key=lambda k: abs(np.linalg.norm(k.pose.c - pose.c) - wanted),
        )[:2]
        if partners:
            self.votes *= self.decay
        for k in partners:
            if self._sweep(k, pose, gray, rover_xy):
                done['pairs'] += 1
                self.pairs += 1

        # Keep this view for the next ones when the camera has moved enough since the last.
        if (
            not self.keyframes
            or min(np.linalg.norm(k.pose.c - pose.c) for k in self.keyframes) >= self.keyframe_step
        ):
            self.keyframes.append(Keyframe(pose, gray, rover_xy))
            if len(self.keyframes) > self.max_keyframes:
                self.keyframes.pop(0)
        self.views += 1
        return done

    def rover_cells(self, pose: Pose, rover_xy, roi) -> np.ndarray:
        """Where the rover covers the floor in this view, on the grid window `roi`: a band of
        `rover_radius` around the segment from its place on the floor to where its top
        (`rover_height`) lands once laid flat (it leans away from the point under the
        camera)."""
        c0, r0, c1, r1 = roi
        x, y = rover_xy
        h = float(pose.c[2])
        lean = h / max(h - self.rover_height, 0.1)  # the top lands this much farther out
        tx = pose.c[0] + (x - pose.c[0]) * lean
        ty = pose.c[1] + (y - pose.c[1]) * lean
        cols = np.arange(c0, c1, dtype=np.float32)[None, :]
        rows = np.arange(r0, r1, dtype=np.float32)[:, None]
        (ax, ay), (bx, by) = self._cells(np.array([[x, y], [tx, ty]], float))
        dx, dy = bx - ax, by - ay
        t = np.clip(((cols - ax) * dx + (rows - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        d = np.hypot(cols - (ax + t * dx), rows - (ay + t * dy)) * self.res
        return d <= self.rover_radius

    def _lay_flat(self, image: np.ndarray, pose: Pose, rover_xy=None) -> None:
        """The picture of the floor: a place takes this view when it sees it about as well as
        the view it has (distance to the point under the camera, over the height); the kept
        views lose their advantage view after view, so that what the camera sees NOW soon
        wins: what moved or went away is redrawn, and the small drift of ARKit's frame does
        not pile up old outlines. The rover is never drawn into it: where it covers the floor
        in this view, the floor seen before is kept."""
        roi = self._roi(pose)
        if roi is None:
            return
        c0, r0, c1, r1 = roi
        top, inside = self._warp(pose, image, 0.0, roi)
        cols = np.arange(c0, c1, dtype=np.float32)
        rows = np.arange(r0, r1, dtype=np.float32)
        cx, cy = self._cells(pose.c[None, :2])[0]
        d = np.hypot(cols[None, :] - cx, rows[:, None] - cy) * self.res / float(pose.c[2])
        score = d.astype(np.float32)
        old = self.view_score[r0:r1, c0:c1]
        old += self.view_aging
        take = inside & (score < old)
        if rover_xy is not None:
            take &= ~self.rover_cells(pose, rover_xy, roi)
        old[take] = score[take]
        self.mosaic[r0:r1, c0:c1][take] = top[take]

    def _sweep(self, key: Keyframe, pose: Pose, gray: np.ndarray, rover_xy) -> bool:
        roi = self._roi(key.pose, pose)
        if roi is None:
            return False
        c0, r0, c1, r1 = roi
        win = (self.window, self.window)
        best = np.full((r1 - r0, c1 - c0), -2.0, np.float32)
        best_i = np.zeros(best.shape, np.int32)
        nccs = []
        texture = None
        for i, z in enumerate(self.heights):
            if min(pose.c[2], key.pose.c[2]) - z < 0.25:
                nccs.append(np.full(best.shape, -2.0, np.float32))
                continue
            a, ina = self._warp(pose, gray, float(z), roi)
            b, inb = self._warp(key.pose, key.gray, float(z), roi)
            a, b = a.astype(np.float32), b.astype(np.float32)
            ma, mb = cv2.boxFilter(a, -1, win), cv2.boxFilter(b, -1, win)
            va = cv2.boxFilter(a * a, -1, win) - ma * ma
            vb = cv2.boxFilter(b * b, -1, win) - mb * mb
            cov = cv2.boxFilter(a * b, -1, win) - ma * mb
            ncc = cov / np.sqrt(np.maximum(va * vb, 1e-6))
            valid = cv2.erode((ina & inb).astype(np.uint8), np.ones(win, np.uint8)) > 0
            ncc[~valid] = -2.0
            if i == 0:
                texture = np.minimum(va, vb)
            nccs.append(ncc)
            better = ncc > best
            best[better], best_i[better] = ncc[better], i
        stack = np.stack(nccs)
        # how much the chosen height stands out: against the heights two steps away or more
        n = len(self.heights)
        idx = np.arange(n)[:, None, None]
        far = np.abs(idx - best_i[None]) >= 2
        others = np.where(far, stack, -2.0).max(axis=0)
        margin = best - others
        good = (best >= self.min_ncc) & (margin >= self.min_margin)
        if texture is not None:
            good &= texture >= self.min_texture
        # the rover moves: where it stood in either view says nothing of the floor
        for view, where in ((pose, rover_xy), (key.pose, key.rover_xy)):
            if where is not None:
                good &= ~self.rover_cells(view, where, roi)
        if not good.any():
            return False
        weight = np.clip(margin, 0, 1) * good
        votes = self.votes[:, r0:r1, c0:c1]
        for i in range(n):
            sel = good & (best_i == i)
            votes[i][sel] += weight[sel]
        total = votes.sum(axis=0)
        over = total > self.max_votes  # bounded: the past cannot outvote the present for long
        if over.any():
            votes[:, over] *= self.max_votes / total[over]
        self._height_cells(good, best_i, roi, pose)
        return True

    def _height_cells(self, good, best_i, roi, pose: Pose) -> None:
        """What this pair says of each decision cell: an obstacle (enough of its points measured
        high), floor (enough measured at floor height), or nothing. As for the colours, what
        the camera sees again REPLACES what was said, the views seeing a cell most from above
        winning: votes added pair after pair would grow the edges of the boxes (where the two
        views of a pair disagree) until the passages close."""
        k = self.colour_k
        c0, r0, c1, r1 = roi
        ra, ca = -(-r0 // k) * k, -(-c0 // k) * k  # the whole cells inside the window
        rb, cb = (r1 // k) * k, (c1 // k) * k
        if rb - ra < k or cb - ca < k:
            return
        sub = (slice(ra - r0, rb - r0), slice(ca - c0, cb - c0))
        g, h = good[sub], self.heights[best_i[sub]]
        rows, cols = (rb - ra) // k, (cb - ca) // k

        def count(m):
            return m.reshape(rows, k, cols, k).sum(axis=(1, 3))

        # The matching window spreads every high patch by half its width (it still sees the
        # box top from just outside it): its holes filled first (a box top matches in places
        # only), it is shrunk back by as much, to its true size.
        up = (g & (h >= self.obstacle_m)).astype(np.uint8)
        half = self.window // 2
        if half:
            square = np.ones((2 * half + 1, 2 * half + 1), np.uint8)
            up = cv2.erode(cv2.morphologyEx(up, cv2.MORPH_CLOSE, square), square)
        high = count(up > 0)
        floor = count(g & (h < self.obstacle_m))
        n = k * k
        # high on a quarter of the cell at least, and more than at floor height: the edge of a
        # box top, where the matching window spills over, says nothing rather than "obstacle"
        says_obstacle = (high >= max(2, n // 4)) & (high >= 2 * floor)
        says_floor = (floor >= max(3, n // 5)) & (high <= max(1, n // 25))
        ys = (np.arange(ra // k, rb // k) + 0.5) * self.cell
        xs = (np.arange(ca // k, cb // k) + 0.5) * self.cell
        x0, _, _, y1 = self.bounds
        q = (np.hypot(x0 + xs[None, :] - pose.c[0], y1 - ys[:, None] - pose.c[1])
             / float(pose.c[2])).astype(np.float32)  # fmt: skip
        win = (slice(ra // k, rb // k), slice(ca // k, cb // k))
        kept, hits = self.height_q[win], self.height_hits[win]
        kept += self.cell_aging
        # only where this view looks down rather than across (`max_side`)
        take = (says_obstacle | says_floor) & (q <= kept) & (q <= self.max_side)
        kept[take] = q[take]
        step = np.where(says_obstacle, 1, -1).astype(np.int8)
        hits[take] = np.clip(hits[take] + step[take], 0, 3)
        self._air_overrules_beam(win, take, says_obstacle)

    def _air_overrules_beam(self, win, take, says_obstacle) -> None:
        """The air has looked again, well after the D405 spoke (`beam_hold` views later) and
        says otherwise: the D405's verdict ends there (the obstacle it saw has gone, or one has
        come where it saw floor). Until then, the D405 is believed over the air."""
        state, view = self.beam_state[win], self.beam_view[win]
        late = take & (self.views > view + self.beam_hold)
        state[late & ~says_obstacle & (state == 1)] = -1
        state[late & says_obstacle & (state == 0)] = -1

    def add_beam(self, obstacle_xy, floor_xy) -> dict:
        """One look of the rover's D405, placed on the floor (beam.BeamLook.on_floor): what
        it sees right ahead is believed over the air map (closer, metric, looking at it now).
        A cell it sees as an obstacle is one at once (added, or confirmed); a cell it sees as
        floor is cleared at once, and what the air had said of it is forgotten (the air must
        see an obstacle there twice again). Its verdict holds until the air, looking again
        `beam_hold` views later, says otherwise."""
        k = self.colour_k

        def cells(xy):
            xy = np.asarray(xy, float).reshape(-1, 2)
            if not len(xy):
                return None
            c = np.floor(self._cells(xy) / k).astype(np.int64)
            ok = (c[:, 0] >= 0) & (c[:, 0] < self.ccols) & (c[:, 1] >= 0) & (c[:, 1] < self.crows)
            return c[ok, 1], c[ok, 0]

        free, hit = cells(floor_xy), cells(obstacle_xy)
        if free is not None:
            self.beam_state[free] = 0
            self.colour_hits[free] = 0
            self.height_hits[free] = 0
            self.beam_view[free] = self.views
        if hit is not None:  # a cell with both: an obstacle (an edge, the safe side)
            self.beam_state[hit] = 1
            self.beam_view[hit] = self.views
        self.beam_looks += 1
        return {
            'obstacle_cells': 0 if hit is None else len(set(zip(*hit, strict=False))),
            'floor_cells': 0 if free is None else len(set(zip(*free, strict=False))),
        }

    def _up(self, cells: np.ndarray) -> np.ndarray:
        """decision cells -> the map's own cells."""
        k = self.colour_k
        up = np.repeat(np.repeat(cells, k, axis=0), k, axis=1)
        out = np.zeros((self.rows, self.cols), bool)
        out[: min(self.rows, up.shape[0]), : min(self.cols, up.shape[1])] = up[
            : self.rows, : self.cols
        ]
        return out

    # -- what the map says ----------------------------------------------------------------

    def height_map(self, min_votes: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """(height in metres, known) per cell: the LiDAR's height where it has measured, else
        the plane sweep's height with the most votes."""
        total = self.votes.sum(axis=0)
        best = self.votes.argmax(axis=0)
        height, known = self.heights[best], total >= min_votes
        if self.depth_maps:
            dh, dk = self._depth_layer()
            height = np.where(dk, dh, height)
            known = known | dk
        return height, known

    def obstacles(self, min_votes: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(height, known, obstacle) per cell: the decision cells two views in a row found high
        (`_height_cells`), the LiDAR's high cells, and the team's colour detection; known:
        where any of them has spoken."""
        height, known = self.height_map(min_votes)
        obstacle = self._up(self.height_hits >= 2)
        known = known | self._up(np.isfinite(self.height_q))
        if self.depth_maps:
            dh, dk = self._depth_layer()
            obstacle |= dk & (dh >= self.obstacle_m)
        colour = self._colour_layer()
        if colour is not None:  # what the team's colour detection sees counts too
            obstacle |= colour
            known = known | colour
        obstacle = self._clean(obstacle)  # the air's view of the rover and its specks
        if self.beam_looks:  # the rover's D405, believed over the air where it spoke; after the
            # cleaning: what it sees right in front of the rover is no part of the rover
            ahead, floor = self._up(self.beam_state == 1), self._up(self.beam_state == 0)
            obstacle = (obstacle | ahead) & ~floor
            known = known | ahead | floor
        return height, known, obstacle

    def _clean(self, obstacle: np.ndarray) -> np.ndarray:
        """Without what is the rover itself and without specks: a patch no larger than one
        decision cell (a bad match, a reflection) is dropped; so is any patch touching the
        rover's place and no larger than the rover (`rover_size`, the team's idea: free the
        blob under the tag), whatever its parts and cables look like from above."""
        if not obstacle.any():
            return obstacle
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            obstacle.astype(np.uint8), connectivity=8
        )
        k = self.colour_k
        drop = stats[:, cv2.CC_STAT_AREA] <= k * k
        disc = None
        if self.rover_now is not None:
            rc, rr = self._cells(np.asarray([self.rover_now], float))[0]
            r = self.rover_radius / self.res
            r0, r1 = max(0, int(rr - r)), min(self.rows, int(rr + r) + 1)
            c0, c1 = max(0, int(rc - r)), min(self.cols, int(rc + r) + 1)
            if r0 < r1 and c0 < c1:
                cols = np.arange(c0, c1)[None, :]
                rows = np.arange(r0, r1)[:, None]
                disc = np.hypot(cols - rc, rows - rr) <= r
                small = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
                small = small * self.res <= self.rover_size
                touching = np.unique(labels[r0:r1, c0:c1][disc])
                drop[touching[small[touching]]] = True
        drop[0] = False  # the background
        out = obstacle & ~drop[labels]
        if disc is not None:
            out[r0:r1, c0:c1] &= ~disc  # under the rover: the rover, not an obstacle
        return out

    def set_rover(self, x: float, y: float) -> None:
        """Where the rover's centre is now, on the floor. It is not an obstacle (`_clean`),
        and where it drives is floor: the cells under its body forget what was said of them
        (a view will say again whether something is there once it has gone)."""
        self.rover_now = (float(x), float(y))
        k = self.colour_k
        cx, cy = self._cells(np.asarray([self.rover_now], float))[0] / k
        r = self.rover_body / self.cell
        r0, r1 = max(0, int(cy - r)), min(self.crows, int(cy + r) + 1)
        c0, c1 = max(0, int(cx - r)), min(self.ccols, int(cx + r) + 1)
        if r0 >= r1 or c0 >= c1:
            return
        cols = np.arange(c0, c1)[None, :] + 0.5
        rows = np.arange(r0, r1)[:, None] + 0.5
        under = np.hypot(cols - cx, rows - cy) <= r
        for hits in (self.colour_hits, self.height_hits):
            hits[r0:r1, c0:c1][under] = 0

    # -- the team's colour detection ------------------------------------------------------

    def _colour_view(self, image: np.ndarray, pose: Pose, rover_xy=None) -> None:
        """The team's colour detection (guidance_global_map.segment_obstacles, then the shadow
        and thin-line filters of sim/fusion_reelle.py, with their defaults) on THIS view laid
        flat: what differs from the floor's colour around it is an obstacle, a shadow or a
        line taped on the floor is not. It sees what has no texture to measure a height on
        (a plain box, a dark bag); the heights see what has the floor's colour.

        Each decision cell keeps what the view that saw it most from above said, the kept views
        losing their advantage view after view; what the camera sees again is replaced, not
        added to (the team's --drone-cumul rule). Seen from the side, a tall box lands
        farther out once laid flat; adding views would grow it into the union of all its
        leanings and close the passages. A cell turns obstacle after two views say so and
        free after two views say not. The rover itself (`rover_xy`) is left out."""
        segment_obstacles, part_ombre, sans_traits = _team_colour_detection()
        k = self.colour_k
        roi = self._roi(pose)
        if roi is None:
            return
        c0, r0, c1, r1 = roi  # widened to whole decision cells
        c0, r0 = (c0 // k) * k, (r0 // k) * k
        c1, r1 = min(self.ccols * k, -(-c1 // k) * k), min(self.crows * k, -(-r1 // k) * k)
        if c1 - c0 < 4 * k or r1 - r0 < 4 * k:
            return
        roi = (c0, r0, c1, r1)
        top, inside = self._warp(pose, image, 0.0, roi)
        ignore = None if rover_xy is None else self.rover_cells(pose, rover_xy, roi)
        pixels = segment_obstacles(
            top,
            inside,
            ignore,
            color_threshold=25.0,
            lightness_weight=0.7,
            min_area_px=max(1, round(0.0005 / self.res**2)),
            floor_window_px=round(1.0 / self.res),
        )
        rows, cols = (r1 - r0) // k, (c1 - c0) // k

        def cells(a, how):
            return getattr(a.reshape(rows, k, cols, k), how)(axis=(1, 3))

        seen = cells(inside, 'mean') >= 0.6
        if ignore is not None:
            seen &= ~cells(ignore, 'any')  # under the rover: nothing to say about the floor
        m = cells(pixels, 'any').astype(np.int8)[::-1]  # the team's filters: row 0 = south
        if inside.any():
            filled = top.copy()
            filled[~inside] = np.median(top[inside], axis=0)  # outside the view: no shadow
            m[(m == 1) & (part_ombre(filled, self.res, m.shape, self.cell) > 0.6)] = 0
        m = sans_traits(m, 0.15 / self.cell)
        obstacle = m[::-1] == 1
        # how far from straight below each cell is seen in this view (distance / height)
        ys = (np.arange(r0 // k, r1 // k) + 0.5) * self.cell
        xs = (np.arange(c0 // k, c1 // k) + 0.5) * self.cell
        x0, _, _, y1 = self.bounds
        q = (np.hypot(x0 + xs[None, :] - pose.c[0], y1 - ys[:, None] - pose.c[1])
             / float(pose.c[2])).astype(np.float32)  # fmt: skip
        gr0, gc0 = r0 // k, c0 // k
        kept = self.colour_q[gr0 : gr0 + rows, gc0 : gc0 + cols]
        hits = self.colour_hits[gr0 : gr0 + rows, gc0 : gc0 + cols]
        kept += self.cell_aging
        take = seen & (q <= kept) & (q <= self.max_side)
        kept[take] = q[take]
        step = np.where(obstacle, 1, -1).astype(np.int8)
        hits[take] = np.clip(hits[take] + step[take], 0, 3)
        self._air_overrules_beam((slice(gr0, gr0 + rows), slice(gc0, gc0 + cols)), take, obstacle)

    def _colour_layer(self) -> np.ndarray | None:
        """The colour obstacles at the map's own cell size, or None without colour."""
        if not self.colour_on:
            return None
        k = self.colour_k
        cells = self.colour_hits >= 2
        up = np.repeat(np.repeat(cells, k, axis=0), k, axis=1)
        out = np.zeros((self.rows, self.cols), bool)
        out[: up.shape[0], : up.shape[1]] = up[: self.rows, : self.cols]
        return out

    def grid(self, cell: float = 0.05, min_votes: float = 0.6) -> dict:
        """Cells of `cell` metres: 1 = an obstacle in it, 0 = floor seen, -1 = unknown; row 0
        is the south (the team's tile convention). Once the colour detection has run, floor
        seen in the picture and found free by it counts as free."""
        height, known, high = self.obstacles(min_votes)
        k = max(1, round(cell / self.res))
        rows, cols = self.rows // k, self.cols // k
        h = height[: rows * k, : cols * k].reshape(rows, k, cols, k)
        kn = known[: rows * k, : cols * k].reshape(rows, k, cols, k)
        hi = high[: rows * k, : cols * k].reshape(rows, k, cols, k)
        obstacle = hi.sum(axis=(1, 3)) >= max(1, k * k // 8)
        seen = kn.sum(axis=(1, 3)) >= k * k // 4
        if self.colour_on:  # floor the colour detection has looked at counts as seen
            looked = np.isfinite(self.colour_q)
            up = np.repeat(np.repeat(looked, self.colour_k, axis=0), self.colour_k, axis=1)
            full = np.zeros((self.rows, self.cols), bool)
            full[: up.shape[0], : up.shape[1]] = up[: self.rows, : self.cols]
            seen |= full[: rows * k, : cols * k].reshape(rows, k, cols, k).mean(axis=(1, 3)) >= 0.5
        cells = np.where(obstacle, 1, np.where(seen, 0, -1)).astype(np.int8)[::-1]
        top = np.where(kn, h, 0).max(axis=(1, 3))[::-1]
        return {
            'res': cell,
            'x0': self.bounds[0],
            'y0': self.bounds[1],
            'cells': cells.tolist(),
            'height_m': np.round(top, 3).tolist(),
        }

    def render(self, min_votes: float = 0.6) -> np.ndarray:
        """The map as a picture: the floor as seen, obstacles tinted red (darker = higher),
        floor whose height is known slightly green, places never seen dark grey."""
        out = self.mosaic.copy()
        never = ~np.isfinite(self.view_score)
        out[never] = (40, 40, 40)
        height, known, obstacle = self.obstacles(min_votes)
        floor = known & ~obstacle & ~never
        out[floor] = (0.8 * out[floor] + 0.2 * np.array([90, 200, 90])).astype(np.uint8)
        if obstacle.any():
            level = np.clip(height[obstacle] / self.heights[-1], 0.2, 1.0)[:, None]
            red = np.array([40, 40, 255], np.float32)
            out[obstacle] = (0.35 * out[obstacle] + 0.65 * red * (1.2 - 0.5 * level)).clip(0, 255)
        return out

    def render_grid(self, cell: float = 0.05, min_votes: float = 0.6) -> np.ndarray:
        """The grid the planner uses, drawn like the team's drone map, at the picture's size
        (same pixels as `render`): free light, obstacle dark red, unknown grey, cell lines."""
        g = self.grid(cell, min_votes)
        cells = np.array(g['cells'], np.int8)[::-1]  # row 0 = north, as pictures
        k = max(1, round(cell / self.res))
        colours = np.zeros((*cells.shape, 3), np.uint8)
        colours[cells == -1] = (70, 70, 70)
        colours[cells == 0] = (225, 232, 225)
        colours[cells == 1] = (40, 40, 190)
        up = np.repeat(np.repeat(colours, k, axis=0), k, axis=1)
        out = np.full((self.rows, self.cols, 3), 70, np.uint8)
        out[: up.shape[0], : up.shape[1]] = up
        if k >= 4:  # the cell lines, faint
            out[::k, :] = (out[::k, :] * 0.85).astype(np.uint8)
            out[:, ::k] = (out[:, ::k] * 0.85).astype(np.uint8)
        return out

    def stats(self) -> dict:
        height, known, obstacle = self.obstacles()
        return {
            'views': self.views,
            'pairs': self.pairs,
            'keyframes': len(self.keyframes),
            'seen_m2': round(float(np.isfinite(self.view_score).sum()) * self.res**2, 2),
            'known_m2': round(float(known.sum()) * self.res**2, 2),
            'obstacle_m2': round(float(obstacle.sum()) * self.res**2, 3),
            'depth_maps': self.depth_maps,
            'floor_offset_cm': round(self.floor_offset * 100, 1),
        }
