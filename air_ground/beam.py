"""The rover's own close look: its D405 depth camera, a second stream next to the air camera.

The air camera maps the whole area but sees tall things lean, moves, and loses details; the
D405 on the front of the rover measures what is right ahead, in metres. The team's local
navigation (docs/local-navigation.md) runs it on the robot: rover_mapper/d405_server.py turns
each depth picture into 5 cm cells, rover/local_map.py keeps them around the rover in its
odometry frame (an obstacle after 2 pictures of 3, forgotten 10 s after its last look), and
the rover server gives them on GET /local_grid. This module reads that grid; the guide places
its cells on the floor through the air-ground link and FloorMap.add_beam believes them over
the air map:

- an obstacle it sees is one at once: added (one just put down in front, or missed from the
  air) or confirmed;
- floor it sees is cleared at once (a box taken away, a false obstacle of the air map).

    beam = BeamReader('http://<rover>:8080')      # a thread; keeps the newest grid
    grid = beam.latest(max_age=2.0)               # a LocalGrid, or None
    obstacles, floor = grid.on_floor(link)        # floor-frame points (N x 2) of its cells
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np


@dataclass
class LocalGrid:
    cells: np.ndarray  # int8 (rows, cols): -1 not seen / 0 free / 1 obstacle; cell [i][j]
    x0: float  # starts at (x0 + j * res, y0 + i * res) in the rover's ODOMETRY frame
    y0: float
    res: float
    pose: tuple[float, float, float] | None  # the rover's odometry pose (x, y, heading rad)
    age_s: float | None  # age of the newest picture in it
    frames: int  # pictures that fed it so far: a new value is a new look
    received: float  # time.monotonic() here

    def on_floor(self, link, max_dist: float = 1.5, sub: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """(obstacle points, floor points) (N, 2) in the floor frame: `sub` x `sub` points in
        each of its cells (the map's decision cells are smaller than its 5 cm: one point per
        cell would leave every other row of them untouched), through the air-ground link
        (guidance.OdomLink), within `max_dist` of the rover."""
        ii, jj = np.nonzero(self.cells >= 0)
        if not len(ii):
            return np.zeros((0, 2)), np.zeros((0, 2))
        x = self.x0 + (jj + 0.5) * self.res
        y = self.y0 + (ii + 0.5) * self.res
        state = self.cells[ii, jj]
        if self.pose is not None:
            near = np.hypot(x - self.pose[0], y - self.pose[1]) <= max_dist
            x, y, state = x[near], y[near], state[near]
        off = ((np.arange(sub) + 0.5) / sub - 0.5) * self.res
        ox, oy = (a.ravel() for a in np.meshgrid(off, off))
        x = (x[:, None] + ox).ravel()
        y = (y[:, None] + oy).ravel()
        state = np.repeat(state, len(ox))
        c, s = np.cos(link.theta), np.sin(link.theta)
        pts = np.column_stack([c * x - s * y + link.tx, s * x + c * y + link.ty])
        return pts[state == 1], pts[state == 0]


def parse_grid(d: dict, received: float) -> LocalGrid:
    cells = np.asarray(d.get('cells') or [], np.int8)
    if cells.ndim != 2:
        cells = np.full((0, 0), -1, np.int8)
    pose = d.get('pose')
    return LocalGrid(
        cells,
        float(d.get('x0') or 0.0),
        float(d.get('y0') or 0.0),
        float(d.get('res') or 0.05),
        None if pose is None else (float(pose[0]), float(pose[1]), float(pose[2])),
        None if d.get('age_s') is None else float(d['age_s']),
        int(d.get('frames') or 0),
        received,
    )


class BeamReader:
    """Asks the rover for its local grid a few times a second, in a thread (the request also
    keeps the robot's D405 feed running, which starts on demand)."""

    def __init__(self, rover_url: str, period: float = 0.25, timeout: float = 2.0,
                 start: bool = True) -> None:  # fmt: skip
        self.url = rover_url.rstrip('/') + '/local_grid'
        self.period, self.timeout = period, timeout
        self.error: str | None = 'not asked yet'
        self.count = 0
        self.last_s: float | None = None  # how long the last request took
        self._grid: LocalGrid | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def latest(self, max_age: float | None = None) -> LocalGrid | None:
        with self._lock:
            g = self._grid
        if g is None or g.age_s is None:
            return None
        if max_age is not None and g.age_s > max_age:
            return None
        return g

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                g = parse_grid(d, time.monotonic())
                with self._lock:
                    self._grid = g
                self.count += 1
                self.error = d.get('error') or (
                    None if g.age_s is not None else 'no D405 picture yet'
                )
            except urllib.error.HTTPError as e:
                self.error = f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:120]}'
            except (OSError, ValueError, KeyError) as e:
                self.error = f'{type(e).__name__}: {e}'
            self.last_s = time.monotonic() - t0
            self._stop.wait(max(0.0, self.period - self.last_s))
