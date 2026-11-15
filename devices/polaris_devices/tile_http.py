"""The generic device: any process that serves its map as JSON over HTTP.

A new device needs no Python code in this repository: it serves

    GET /map  ->  200 {"t": <optional, ignored>, "res": 0.05, "x0": 1.20, "y0": 0.40,
                       "cells": [[-1, 0, 1, ...], ...]}
              or  503 when it has nothing fresh (the device counts as unavailable)

`cells[i][j]` is the cell at x = x0 + j * res, y = y0 + i * res in the MAP frame (row 0 =
south), 1 obstacle, 0 free and seen, -1 not seen. `res` is 0.05 m (or 0.1, 0.2, 0.4, 0.8:
0.05 times a power of two); x0, y0 are multiples of res. A device that already cuts tiles may
answer {"t": ..., "tiles": [{"z", "x", "y", "data"}, ...]} instead.

Tiles are stamped when they are received (time.monotonic(), the clock of every other adapter and
of the rover), not with the device's "t": a Unix "t" next to monotonic stamps would make an old
tile always win over a fresh one of the same role.

Optional, when the device can locate other devices (settings {"locates": true}):

    GET /pose ->  200 {"<device id>": [x, y, heading_rad_or_null], ...}

A zoom request re-reads /map and keeps the tiles over the requested area.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

from .registry import DeviceUnavailable


def get_json(url: str, timeout: float):
    """Decoded JSON, or None when the device does not answer or answers an error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b'null')
    except (OSError, ValueError):  # refused, timeout, HTTP 4xx/5xx (HTTPError), bad JSON
        return None


def level_of(res: float, carte) -> int:
    """Fusion level z whose cell is `res` m."""
    k = math.log2(res / carte.res_fine)
    z = carte.zmax - round(k)
    if abs(k - round(k)) > 1e-6 or not 0 <= z <= carte.zmax:
        raise ValueError(f'res {res} m is not {carte.res_fine} m times a power of two')
    return z


def messages_from_answer(answer: dict, carte) -> list[dict]:
    """Tile messages (source not set) from a /map answer, stamped when received."""
    from fusion_carte import decouper

    t = time.monotonic()  # one clock in the fusion: the device's "t" is ignored
    if 'tiles' in answer:
        return [
            {'t': t, 'z': int(m['z']), 'x': int(m['x']), 'y': int(m['y']), 'data': m['data']}
            for m in answer['tiles']
        ]
    z = level_of(float(answer['res']), carte)
    return decouper(answer['cells'], float(answer['x0']), float(answer['y0']), z, carte, '', t)


class TileHttpDevice:
    """MapSource over HTTP. Reads /map at most every `period` s."""

    def __init__(self, address: str, carte, timeout: float = 1.0, period: float = 0.5) -> None:
        self.url = address.rstrip('/')
        self.carte, self.timeout, self.period = carte, timeout, period
        self._answer, self._read_at = None, -math.inf

    def _read(self, force=False):
        now = time.monotonic()
        if force or now - self._read_at >= self.period:
            self._answer, self._read_at = get_json(self.url + '/map', self.timeout), now
        return self._answer

    def available(self) -> bool:
        return self._read() is not None

    def tiles(self) -> list[dict]:
        answer = self._read()
        return [] if answer is None else messages_from_answer(answer, self.carte)

    def zoom(self, tile) -> list[dict]:
        z, tx, ty = tile
        side = self.carte.taille_tuile(z)
        x0, y0 = tx * side, ty * side
        answer = self._read(force=True)
        out = []
        for msg in [] if answer is None else messages_from_answer(answer, self.carte):
            mx, my, mside = self.carte.rect_tuile((int(msg['z']), msg['x'], msg['y']))
            if mx < x0 + side and mx + mside > x0 and my < y0 + side and my + mside > y0:
                out.append(msg)
        return out


class TileHttpLocator(TileHttpDevice):
    """TileHttpDevice that also answers GET /pose."""

    def pose_of(self, device_id: str):
        poses = get_json(self.url + '/pose', self.timeout) or {}
        pose = poses.get(device_id)
        if pose is None:
            return None
        x, y, heading = (list(pose) + [None])[:3]
        return float(x), float(y), None if heading is None else float(heading)


def make(spec, fleet):
    if not spec.address:
        raise DeviceUnavailable(f'{spec.id}: tile-http needs "address"')
    cls = TileHttpLocator if spec.settings.get('locates') else TileHttpDevice
    return cls(
        spec.address,
        fleet.carte,
        timeout=float(spec.settings.get('timeout', 1.0)),
        period=float(spec.settings.get('period', 0.5)),
    )
