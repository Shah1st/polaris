"""Rolling obstacle grid around the rover, in the odometry frame, from the depth camera.

No I/O and standard library only, like rover_core: the rover server feeds it the grids of
`rover_mapper/d405_server.py` (GET /grid) and the controller asks it whether the path ahead is
clear (rover_core.Controller, `local_guard`).

Cells are 5 cm squares aligned with the odometry axes, the values of the map fusion:
1 obstacle, 0 floor seen free, -1 not seen. The grid keeps a square of `size` metres centred
on the rover.

Two rules keep one bad picture from deciding anything, and old news from staying:

* confirmation: a cell is an obstacle once `confirm` of its last three looks said so, and free
  once its last `confirm` looks said free; in between it keeps what it was (not seen, for a
  cell that is new). One picture with a speck of noise marks nothing.
* forgetting, by the clock: a cell not looked at for `hold` seconds counts as not seen, for
  every reader, whether or not pictures still arrive. Cells closer than `near` to the rover
  are kept longer (`hold_near`): the camera cannot see the floor right under its nose, so it
  cannot confirm or clear them, but they do not stay for ever either.
"""

from __future__ import annotations

import math
import threading

UNKNOWN, FREE, OBSTACLE = -1, 0, 1


class LocalMap:
    def __init__(
        self,
        size: float = 2.5,
        res: float = 0.05,
        hold: float = 10.0,
        near: float = 0.45,
        hold_near: float = 30.0,
        confirm: int = 2,
    ) -> None:
        self.size, self.res, self.hold, self.near = size, res, hold, near
        self.hold_near, self.confirm = hold_near, confirm
        # (i, j) -> [value, time of the last look, the last three looks as bits (1 = obstacle),
        #            number of looks so far (3 at most)]
        self.cells: dict[tuple[int, int], list] = {}
        self.t_last: float | None = None  # time of the last picture integrated
        self.frames = 0
        self.lock = threading.Lock()

    # -- writing ----------------------------------------------------------------------

    def integrate(self, cells, x0: float, y0: float, res: float, pose, t: float) -> None:
        """One picture's grid: cells[i][j] at (x0 + j*res, y0 + i*res) in the camera's own
        frame (x forward, y left, origin on the floor under the lens); `pose` = (x, y, heading)
        of that frame in the odometry frame when the picture was taken; `t` = now."""
        rows = len(cells)
        cols = len(cells[0]) if rows else 0
        sx, sy, sh = pose
        c, s = math.cos(sh), math.sin(sh)
        r = self.res
        new: dict[tuple[int, int], int] = {}  # what this picture says of each map cell
        if rows and cols:
            # Every map cell whose centre falls in a seen cell of the picture takes its value
            # (looked up from the map side: a rotated grid copied the other way leaves holes).
            corners = [
                (sx + c * px - s * py, sy + s * px + c * py)
                for px in (x0, x0 + cols * res)
                for py in (y0, y0 + rows * res)
            ]
            j_lo = math.floor(min(p[0] for p in corners) / r)
            j_hi = math.floor(max(p[0] for p in corners) / r)
            i_lo = math.floor(min(p[1] for p in corners) / r)
            i_hi = math.floor(max(p[1] for p in corners) / r)
            for i in range(i_lo, i_hi + 1):
                dy = (i + 0.5) * r - sy
                for j in range(j_lo, j_hi + 1):
                    dx = (j + 0.5) * r - sx
                    jj = math.floor((c * dx + s * dy - x0) / res)
                    ii = math.floor((-s * dx + c * dy - y0) / res)
                    if 0 <= ii < rows and 0 <= jj < cols:
                        v = cells[ii][jj]
                        if v != UNKNOWN:
                            new[(i, j)] = int(v)
            # ... and no obstacle is lost: each obstacle cell marks the map cell under its centre.
            for ii in range(rows):
                row = cells[ii]
                py = y0 + (ii + 0.5) * res
                for jj in range(cols):
                    if row[jj] == OBSTACLE:
                        px = x0 + (jj + 0.5) * res
                        key = (
                            math.floor((sy + s * px + c * py) / r),
                            math.floor((sx + c * px - s * py) / r),
                        )
                        new[key] = OBSTACLE
        with self.lock:
            self._forget((sx, sy), t)
            k = max(1, int(self.confirm))
            for key, seen in new.items():
                cell = self.cells.get(key)
                if cell is None:
                    cell = self.cells[key] = [UNKNOWN, t, 0, 0]
                cell[1] = t
                cell[2] = ((cell[2] << 1) | (1 if seen == OBSTACLE else 0)) & 0b111
                cell[3] = min(3, cell[3] + 1)
                k = min(k, 3)
                window = 1 if k == 1 else 3  # confirm 1: the last look alone decides
                looks = cell[2] & ((1 << min(cell[3], window)) - 1)
                if bin(looks).count('1') >= k:
                    cell[0] = OBSTACLE
                elif cell[3] >= k and cell[2] & ((1 << k) - 1) == 0:
                    cell[0] = FREE
            self.t_last = t
            self.frames += 1

    def _alive(self, key, cell, centre, t: float) -> bool:
        """The cell is inside the window and was looked at recently enough."""
        r, half = self.res, self.size / 2
        dx, dy = (key[1] + 0.5) * r - centre[0], (key[0] + 0.5) * r - centre[1]
        if abs(dx) > half or abs(dy) > half:
            return False
        limit = self.hold_near if math.hypot(dx, dy) <= self.near else self.hold
        return t - cell[1] <= limit

    def _forget(self, centre, t: float) -> None:
        dead = [k for k, cell in self.cells.items() if not self._alive(k, cell, centre, t)]
        for key in dead:
            del self.cells[key]

    def clear(self) -> None:
        with self.lock:
            self.cells.clear()
            self.t_last = None

    # -- reading ----------------------------------------------------------------------

    def age(self, t: float) -> float | None:
        """Seconds since the last picture, None before the first one."""
        return None if self.t_last is None else t - self.t_last

    def first_obstacle(self, points, half_width: float, max_dist: float, t: float):
        """(distance along `points`, (x, y) of the obstacle cell) for the first point of the
        polyline that comes within `half_width` of the centre of an obstacle cell, or None
        when the first `max_dist` metres are clear. `points` start at the rover and should be
        a few centimetres apart; `t` = now (cells too old to trust are not obstacles)."""
        if not points:
            return None
        r = self.res
        with self.lock:
            self._forget(points[0], t)
            obstacles = [
                ((j + 0.5) * r, (i + 0.5) * r)
                for (i, j), cell in self.cells.items()
                if cell[0] == OBSTACLE
            ]
        if not obstacles:
            return None
        dist, prev = 0.0, points[0]
        for p in points:
            dist += math.dist(prev, p)
            prev = p
            if dist > max_dist:
                return None
            for ox, oy in obstacles:
                if abs(ox - p[0]) <= half_width and abs(oy - p[1]) <= half_width:
                    if math.hypot(ox - p[0], oy - p[1]) <= half_width:
                        return dist, (ox, oy)
        return None

    def counts(self, centre, t: float) -> dict:
        with self.lock:
            self._forget(centre, t)
            values = [cell[0] for cell in self.cells.values()]
        return {'obstacle': values.count(OBSTACLE), 'free': values.count(FREE)}

    def snapshot(self, centre, t: float) -> dict:
        """The square around `centre` as rows of -1/0/1: cells[i][j] covers
        [x0 + j*res, x0 + (j+1)*res] x [y0 + i*res, y0 + (i+1)*res] in the odometry frame."""
        r = self.res
        n = int(round(self.size / r))
        j0 = math.floor(centre[0] / r) - n // 2
        i0 = math.floor(centre[1] / r) - n // 2
        with self.lock:
            self._forget(centre, t)
            rows = [
                [(self.cells.get((i0 + i, j0 + j)) or (UNKNOWN,))[0] for j in range(n)]
                for i in range(n)
            ]
        return {'res': r, 'x0': round(j0 * r, 4), 'y0': round(i0 * r, 4), 'cells': rows}
