"""The rover clears the drone's false obstacles (fusion_reelle --clear-drone).

The drone's map is made of colours seen from above: a blue tape line, a stain or a reflection
on the floor can look like an obstacle in every picture. In superposition mode the rover adds
obstacles and never erases the drone's, so such a false obstacle stays on the map and the
planner keeps going round it, or finds no way at all.

With this on, a drone obstacle cell is cleared when the rover's depth camera, looking at it
now and from close (between `near` and `reach` m ahead of the lens, within `half_fov_deg` of
its axis), sees it free together with every cell within `margin` around it, in `views` fresh
grids in a row. A cell the rover has not seen around it blocks the clearing as an obstacle
does: a real obstacle that the drone placed a few centimetres off is kept by the margin.

A cleared cell stays cleared, also on the drone's later pictures (there with one cell of
slack around it: the placement of the pictures jitters, so near a real obstacle the margin
is one cell less on them), until:
- the rover sees an obstacle within `margin` of it;
- the drone's picture shows, within `margin` of it (and the slack) and in `changed` pictures
  in a row, a piece of at least `grown` obstacle cells that were not there when it was
  cleared (one cell of slack): something was put there;
- reset(): the maps were reset, the drone camera calibrated again, or the switch turned off.

Cells are the maps' fine cells, absolute: (i, j) = (floor(y / res), floor(x / res)). No I/O
and no lock: fusion_reelle calls it under its own lock.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class DroneClears:
    def __init__(
        self,
        res: float = 0.05,
        reach: float = 0.5,
        near: float = 0.25,
        margin: float = 0.10,
        views: int = 3,
        half_fov_deg: float = 35.0,
        grown: int = 3,
        changed: int = 2,
    ) -> None:
        self.res = res
        self.reach, self.near = reach, near
        self.margin_cells = max(1, int(round(margin / res)))
        self.views = max(1, int(views))
        self.half_fov = math.radians(half_fov_deg)
        self.grown, self.changed = grown, changed
        self.reset()

    def reset(self) -> None:
        self.cleared: dict[tuple[int, int], int] = {}  # cell -> pictures in a row with news
        self.streak: dict[tuple[int, int], int] = {}  # cell -> fresh grids in a row clearable
        self.seen: set[tuple[int, int]] = set()  # drone obstacles around them when cleared

    def __len__(self) -> int:
        return len(self.cleared)

    # ---- helpers ---------------------------------------------------------------------------
    def _disc(self) -> np.ndarray:
        k = 2 * self.margin_cells + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @staticmethod
    def _mask(cells, i0: int, j0: int, h: int, w: int, pad: int = 0) -> np.ndarray:
        """`cells` as a boolean mask of the window whose first cell is (i0, j0), `pad` cells
        wider on each side."""
        mask = np.zeros((h + 2 * pad, w + 2 * pad), bool)
        for i, j in cells:
            a, b = i - i0 + pad, j - j0 + pad
            if 0 <= a < h + 2 * pad and 0 <= b < w + 2 * pad:
                mask[a, b] = True
        return mask

    def _with_slack(self, cells, i0: int, j0: int, h: int, w: int) -> np.ndarray:
        """`cells` and the cells next to them, on the window."""
        mask = self._mask(cells, i0, j0, h, w, pad=1).astype(np.uint8)
        return cv2.dilate(mask, np.ones((3, 3), np.uint8))[1:-1, 1:-1] == 1

    # ---- the rover's grid ------------------------------------------------------------------
    def observe(self, rover, drone, i0, j0, lens, heading, fresh=True):
        """One look of the rover. `rover` and `drone`: its grid and the drone's layer on the
        same window of the maps (-1/0/1), first cell (i0, j0). `lens`: (x, y) of the depth
        camera on the maps; `heading`: where it looks (rad); None when not known. `fresh`: the
        grid holds new pictures since the last call (the same memory read twice is not a
        second look). Returns (cells newly cleared, cells given back to the drone)."""
        h, w = rover.shape
        disc = self._disc()
        near_obstacle = cv2.dilate((rover == 1).astype(np.uint8), disc) == 1
        given_back = []
        for c in list(self.cleared):
            a, b = c[0] - i0, c[1] - j0
            if 0 <= a < h and 0 <= b < w and near_obstacle[a, b]:
                del self.cleared[c]
                self.streak.pop(c, None)
                given_back.append(c)
        if not fresh:
            return [], given_back
        if lens is None or heading is None:
            self.streak = {}
            return [], given_back
        free = (rover == 0).astype(np.uint8)
        all_free = cv2.erode(free, disc, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 1
        r = self.res
        ys, xs = np.mgrid[0:h, 0:w]
        dx = (j0 + xs + 0.5) * r - lens[0]
        dy = (i0 + ys + 0.5) * r - lens[1]
        dist = np.hypot(dx, dy)
        off = np.abs(np.mod(np.arctan2(dy, dx) - heading + np.pi, 2 * np.pi) - np.pi)
        looked_at = (dist >= self.near) & (dist <= self.reach) & (off <= self.half_fov)
        ii, jj = np.nonzero((drone == 1) & all_free & looked_at)
        now = set(zip((ii + i0).tolist(), (jj + j0).tolist(), strict=True))
        self.streak = {c: self.streak.get(c, 0) + 1 for c in now}
        new = [c for c, n in self.streak.items() if n >= self.views and c not in self.cleared]
        if new:
            # what the drone had around them: more there in a later picture is a change
            around = cv2.dilate(self._mask(new, i0, j0, h, w).astype(np.uint8), disc) == 1
            si, sj = np.nonzero(around & (drone == 1))
            self.seen.update(zip((si + i0).tolist(), (sj + j0).tolist(), strict=True))
            for c in new:
                self.cleared[c] = 0
                del self.streak[c]  # given back later, a cell needs its `views` looks again
        return new, given_back

    # ---- the drone's pictures --------------------------------------------------------------
    def check_picture(self, m, i0, j0):
        """A new picture of the drone placed on the maps (-1/0/1, first cell (i0, j0)), as it
        came, before apply(). Gives back the cleared cells near something new: obstacle cells
        that were not there when they were cleared (one cell of slack), `grown` of them at
        least in one piece (a speck is noise), in `changed` pictures in a row. Returns them."""
        if not self.cleared:
            return []
        h, w = m.shape
        before = self._with_slack(self.seen | set(self.cleared), i0, j0, h, w)
        news = ((m == 1) & ~before).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(news, connectivity=8)
        big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.grown]
        k = 2 * (self.margin_cells + 1) + 1  # the margin, and the slack
        near_news = (
            cv2.dilate(
                np.isin(labels, big).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            )
            == 1
        )
        given_back = []
        for c in list(self.cleared):
            a, b = c[0] - i0, c[1] - j0
            if not (0 <= a < h and 0 <= b < w) or m[a, b] == -1:
                continue  # not in this picture
            if near_news[a, b]:
                self.cleared[c] += 1
                if self.cleared[c] >= self.changed:
                    del self.cleared[c]
                    self.streak.pop(c, None)
                    given_back.append(c)
            else:
                self.cleared[c] = 0
        return given_back

    def apply(self, m, i0, j0, slack: bool = True) -> int:
        """Clears, in place, the drone's obstacles of `m` (-1/0/1, first cell (i0, j0)) on the
        cleared cells, and next to them with `slack` (a later picture of the drone: its
        placement jitters by a cell or so). Returns how many cells it changed."""
        if not self.cleared:
            return 0
        h, w = m.shape
        if slack:
            mask = self._with_slack(self.cleared, i0, j0, h, w)
        else:
            mask = self._mask(self.cleared, i0, j0, h, w)
        hit = mask & (m == 1)
        m[hit] = 0
        return int(hit.sum())

    def tiles_touched(self, n: int, z: int):
        """The keys (z, tx, ty) of the n x n tiles at level z (the finest) that hold a
        cleared cell or a cell next to one."""
        return {
            (z, (j + dj) // n, (i + di) // n)
            for i, j in self.cleared
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        }
