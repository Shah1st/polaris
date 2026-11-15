"""The rover's own close look: the D405 depth camera, a second stream next to the air camera.

The air camera maps the whole area but sees tall things lean, moves, and loses details; the
D405 on the front of the rover measures what is right ahead, in metres, at a few frames a
second. Each of its looks is placed on the floor with the rover's pose at that moment, and
then (floor_map.FloorMap.add_beam):

- adds an obstacle the air did not see (or not yet: one just put down in front),
- confirms one it did see,
- clears one that is not there (a box taken away, a false obstacle of the air map).

Its detection is the team's: rover_mapper/perception_service.py --d405 (the floor found in
the depth itself, obstacles 4 to 40 cm above it, 5 cm cells, rover frame: x ahead, y to the
left). This module reads its GET /obstacles:

    beam = BeamReader('http://<rover>:8095')      # a thread; keeps the newest look
    look = beam.latest(max_age=1.0)               # a BeamLook, or None
    obstacles, free = look.on_floor(rover_pose)   # floor-frame points (N x 2) of its cells
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np


@dataclass
class BeamLook:
    cells: np.ndarray  # int8 (rows, cols): -1 unseen / 0 floor / 1 obstacle; [i][j] at
    x0: float  # x = x0 + j * res (ahead), y = y0 + i * res (to the left), in the
    y0: float  # D405's frame on the floor
    res: float
    age_s: float  # how old the look was when the service answered
    received: float  # time.monotonic() here when it arrived
    seq: int

    @property
    def taken(self) -> float:
        """When the D405 took it, on this computer's clock (time.monotonic())."""
        return self.received - self.age_s

    def on_floor(self, pose, offset=(0.0, 0.0), yaw_deg: float = 0.0, max_range: float = 1.0,
                 sub: int = 2) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        """(obstacle points, floor points), (N, 2) each, in the floor frame, for the rover at
        `pose` (x, y, heading in radians: guidance.Pose2). `offset`: where the D405's frame
        is on the rover (m, ahead and to the left of its centre); `yaw_deg`: where it looks,
        counter-clockwise from the rover's front. Only cells within `max_range` m ahead (the
        D405's error grows fast with distance). Each cell gives sub x sub points, so that a
        finer map is covered."""
        rows, cols = self.cells.shape
        ii, jj = np.nonzero(self.cells >= 0)
        if not len(ii):
            return np.zeros((0, 2)), np.zeros((0, 2))
        f = (np.arange(sub) + 0.5) / sub
        fi, fj = np.meshgrid(f, f, indexing='ij')
        x = self.x0 + (jj[:, None] + fj.ravel()[None, :]) * self.res  # ahead of the D405
        y = self.y0 + (ii[:, None] + fi.ravel()[None, :]) * self.res  # to its left
        state = np.repeat(self.cells[ii, jj][:, None], sub * sub, axis=1)
        keep = (x > 0) & (x <= max_range)
        x, y, state = x[keep], y[keep], state[keep]
        a = math.radians(yaw_deg)
        xr = offset[0] + x * math.cos(a) - y * math.sin(a)  # on the rover: ahead, left
        yr = offset[1] + x * math.sin(a) + y * math.cos(a)
        c, s = math.cos(pose.heading), math.sin(pose.heading)
        pts = np.column_stack([pose.x + c * xr - s * yr, pose.y + s * xr + c * yr])
        return pts[state == 1], pts[state == 0]


def parse_look(d: dict, received: float, seq: int) -> BeamLook:
    cells = np.asarray(d['cells'], np.int8)
    if cells.ndim != 2:
        raise ValueError('the D405 look has no grid')
    if d.get('floor') is False:  # no floor found in the depth: nothing to say
        cells = np.full(cells.shape, -1, np.int8)
    return BeamLook(cells, float(d['x0']), float(d['y0']), float(d['res']),
                    float(d.get('age_s') or 0.0), received, seq)  # fmt: skip


class BeamReader:
    """Asks the D405 perception for a fresh look a few times a second, in a thread."""

    def __init__(self, base_url: str, period: float = 0.2, timeout: float = 2.0,
                 start: bool = True) -> None:  # fmt: skip
        self.url = base_url.rstrip('/') + '/obstacles'
        self.period, self.timeout = period, timeout
        self.error: str | None = 'not asked yet'
        self.count = 0
        self.last_s: float | None = None  # how long the last request took
        self._look: BeamLook | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def latest(self, max_age: float | None = None) -> BeamLook | None:
        with self._lock:
            look = self._look
        if look is None or (max_age is not None and time.monotonic() - look.taken > max_age):
            return None
        return look

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                now = time.monotonic()
                self.count += 1
                look = parse_look(d, now, self.count)
                with self._lock:
                    self._look = look
                self.error = None
            except urllib.error.HTTPError as e:
                self.error = f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:120]}'
            except (OSError, ValueError, KeyError) as e:
                self.error = f'{type(e).__name__}: {e}'
            self.last_s = time.monotonic() - t0
            self._stop.wait(max(0.0, self.period - self.last_s))
