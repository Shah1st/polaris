"""TCP stream of the rover state, for the ROS side (polaris `guidance_rover_vo`).

The mapper runs natively on the Mac (GPU); ROS runs in a Linux container
where DDS cannot reach the host, so the two talk over one TCP port. The
container connects to `host.docker.internal:<port>`.

Wire format, one message per mapper step (same as polaris
`guidance_rover_vo/protocol.py`, keep both in sync):

    uint32 big-endian  header length H
    H bytes            UTF-8 JSON header:
                       {"v": 1, "stamp": s, "x": m, "y": m, "yaw": rad,
                        "lost": bool, "quality": 0..1, "map_version": int,
                        "n": number of points}
    n * 12 bytes       points, float32 little-endian (x, y, z) in base_link
"""
from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np

from .export import RoverState

PROTOCOL_VERSION = 1


def encode(state: RoverState) -> bytes:
    pts = np.ascontiguousarray(state.points, dtype='<f4').reshape(-1, 3)
    header = json.dumps({
        'v': PROTOCOL_VERSION, 'stamp': state.stamp, 'x': state.x, 'y': state.y,
        'yaw': state.yaw, 'lost': state.lost, 'quality': state.quality,
        'map_version': state.map_version, 'n': len(pts),
    }).encode()
    return struct.pack('>I', len(header)) + header + pts.tobytes()


class StateServer:
    """Accepts any number of clients and sends each of them every new state.

    A slow or dead client is dropped rather than allowed to block the mapper.
    """

    def __init__(self, port: int, host: str = '0.0.0.0'):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def _accept(self) -> None:
        while True:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(0.2)
            print(f'Pont ROS : client connecté depuis {addr[0]}')
            with self._lock:
                self._clients.append(conn)

    def publish(self, state: RoverState) -> None:
        msg = encode(state)
        with self._lock:
            alive = []
            for c in self._clients:
                try:
                    c.sendall(msg)
                    alive.append(c)
                except OSError:
                    c.close()
            self._clients = alive

    def close(self) -> None:
        self._sock.close()
        with self._lock:
            for c in self._clients:
                c.close()
            self._clients = []
