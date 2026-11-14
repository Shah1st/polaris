"""TCP stream of the true rover state in the guidance_rover_vo wire format.

Lets ROS nodes (e.g. rover_vo_bridge_node with port:=7410) be tested on the simulation's
ground truth, without the rover mapper. Plain Python: the wire format module has no ROS import.
"""

import os
import socket
import sys
import threading

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'guidance_rover_vo')
)
from guidance_rover_vo.protocol import RoverState, encode  # noqa: E402

__all__ = ['RoverState', 'StateServer']


class StateServer:
    """Accepts any number of clients and sends each of them every new state.

    A slow or dead client is dropped rather than allowed to block the simulation.
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
            print(f'state stream: client connected from {addr[0]}', flush=True)
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
