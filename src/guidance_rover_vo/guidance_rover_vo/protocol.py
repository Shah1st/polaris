"""Wire format of the rover visual-odometry stream (no ROS dependency).

The rover mapper (monocular 3D mapping on the rover camera, runs natively on
the host for GPU access) sends one message per mapping step over TCP:

    uint32 big-endian  header length H
    H bytes            UTF-8 JSON header:
                       {"v": 1, "stamp": s, "x": m, "y": m, "yaw": rad,
                        "lost": bool, "quality": 0..1, "map_version": int,
                        "n": number of points}
    n * 12 bytes       points, float32 little-endian (x, y, z) in base_link

Frames: x/y/yaw are base_link in the mapper's own odom frame (REP 103: x
forward at start, y left, z up, metres, floor at z = 0). While `lost` is true
the pose is frozen and must not be trusted. `map_version` changes when the
mapper re-expresses its whole frame (metric calibration in the first seconds),
so the pose may jump at that moment.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field

import numpy as np

PROTOCOL_VERSION = 1
_LEN = struct.Struct('>I')
MAX_HEADER = 1 << 16


@dataclass
class RoverState:
    stamp: float
    x: float
    y: float
    yaw: float
    lost: bool
    quality: float
    map_version: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


def encode(state: RoverState) -> bytes:
    pts = np.ascontiguousarray(state.points, dtype='<f4').reshape(-1, 3)
    header = json.dumps(
        {
            'v': PROTOCOL_VERSION,
            'stamp': state.stamp,
            'x': state.x,
            'y': state.y,
            'yaw': state.yaw,
            'lost': state.lost,
            'quality': state.quality,
            'map_version': state.map_version,
            'n': len(pts),
        }
    ).encode()
    return _LEN.pack(len(header)) + header + pts.tobytes()


def decode(buf: bytes) -> tuple[RoverState | None, int]:
    """Decode one message from the start of `buf`.

    Returns (state, bytes consumed), or (None, 0) if `buf` does not hold a
    complete message yet. Raises ValueError on a malformed or foreign stream.
    """
    if len(buf) < _LEN.size:
        return None, 0
    (h,) = _LEN.unpack_from(buf)
    if h > MAX_HEADER:
        raise ValueError(f'header length {h} too large, not a rover state stream')
    if len(buf) < _LEN.size + h:
        return None, 0
    head = json.loads(buf[_LEN.size : _LEN.size + h])
    if head.get('v') != PROTOCOL_VERSION:
        raise ValueError(f'unsupported protocol version {head.get("v")}')
    n = int(head['n'])
    end = _LEN.size + h + 12 * n
    if len(buf) < end:
        return None, 0
    pts = np.frombuffer(buf, dtype='<f4', count=3 * n, offset=_LEN.size + h).reshape(n, 3)
    state = RoverState(
        stamp=float(head['stamp']),
        x=float(head['x']),
        y=float(head['y']),
        yaw=float(head['yaw']),
        lost=bool(head['lost']),
        quality=float(head['quality']),
        map_version=int(head['map_version']),
        points=pts.astype(np.float32),
    )
    return state, end


class StreamReader:
    """Reads successive messages from a connected socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b''

    def read(self) -> RoverState:
        """Block until one full message arrived. Raises ConnectionError on EOF."""
        while True:
            state, used = decode(self._buf)
            if state is not None:
                self._buf = self._buf[used:]
                return state
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError('rover mapper closed the connection')
            self._buf += chunk
