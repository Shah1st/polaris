"""Stand-in for the rover mapper: serves made-up rover states on a TCP port,
in the guidance_rover_vo wire format, so the ROS bridge can be tested anywhere.

    python3 fake_mapper.py [port]

The rover drives a 1 m circle at 0.2 m/s with a small wall of points ahead.
"""
import math
import socket
import sys
import threading
import time

import numpy as np
from guidance_rover_vo.protocol import RoverState, encode

port = int(sys.argv[1]) if len(sys.argv) > 1 else 7400
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('0.0.0.0', port))
srv.listen(4)
clients = []
lock = threading.Lock()


def accept() -> None:
    while True:
        c, _ = srv.accept()
        with lock:
            clients.append(c)


threading.Thread(target=accept, daemon=True).start()
wall = np.stack(np.meshgrid(np.linspace(0.6, 0.6, 20), np.linspace(-0.3, 0.3, 20),
                            indexing='ij'), -1).reshape(-1, 2)
pts = np.c_[wall, np.linspace(0.0, 0.3, len(wall))].astype(np.float32)
t0 = time.time()
print(f'fake mapper on port {port}', flush=True)
while True:
    t = time.time() - t0
    a = 0.2 * t  # rad, 1 m radius at 0.2 m/s
    st = RoverState(stamp=time.time(), x=math.sin(a), y=1 - math.cos(a), yaw=a, lost=False,
                    quality=0.95, map_version=1, points=pts)
    msg = encode(st)
    with lock:
        alive = []
        for c in clients:
            try:
                c.sendall(msg)
                alive.append(c)
            except OSError:
                c.close()
        clients[:] = alive
    time.sleep(1 / 6)
