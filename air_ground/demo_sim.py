"""The control dashboard without any hardware, on this computer.

    python air_ground/demo_sim.py        then open http://127.0.0.1:8097

Starts the simulated rover of rover/rover_server.py (with its simulated camera), a fake phone
that films it from a camera that keeps moving (tests/fake_phone.py: the pictures and poses
CamStream sends in ARKit mode), and the guide with its dashboard. Click on the map: the
simulated rover drives there. Nothing here talks to the real rover or to a real phone.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'rover'))

import rover_server  # noqa: E402
import synthetic as syn  # noqa: E402
from fake_phone import HEIGHT, SIDE, make_phone  # noqa: E402
from guidance import RoverClient  # noqa: E402
from guide_rover import GuideService  # noqa: E402
from rover_core import Config  # noqa: E402
from rover_marker import RoverFinder  # noqa: E402


def serve(handler, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(('127.0.0.1', port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8097, help='the dashboard (8097)')
    ap.add_argument('--rover-port', type=int, default=8098, help='the simulated rover (8098)')
    ap.add_argument('--phone-port', type=int, default=8099, help='the fake phone (8099)')
    a = ap.parse_args()
    args = SimpleNamespace(sim_drift=0.0, width=640, height=480, fps=15.0, quality=70, camera='0',
                           no_camera=False, d405='off')  # fmt: skip
    config = os.path.join(tempfile.mkdtemp(prefix='polaris-demo-'), 'rover_config.json')
    app = rover_server.App(Config(), config, args, serial_port=None)
    app.set_mode('sim')
    rover = serve(rover_server.make_handler(app), a.rover_port)
    phone = serve(make_phone(app, syn.floor_texture()), a.phone_port)
    service = GuideService(
        f'http://127.0.0.1:{a.phone_port}',
        RoverClient(f'http://127.0.0.1:{a.rover_port}'),
        RoverFinder(0, SIDE, HEIGHT),
        bounds=(-1.0, -1.0, 1.0, 1.0),
        colour=False,  # the test floor is painted with coloured squares: not obstacles
        port=a.port,
    )
    service.start()
    print(f'SIMULATION, no hardware: dashboard http://127.0.0.1:{service.port}', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        phone.shutdown()
        rover.shutdown()
        app.close()


if __name__ == '__main__':
    main()
