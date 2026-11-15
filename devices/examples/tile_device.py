"""A complete Polaris device in 40 lines, standard library only: it serves its map over HTTP.

    python devices/examples/tile_device.py --port 8178 --x0 1.0 --y0 0.5
    # then declare it: {"id": "box", "role": "drone", "kind": "tile-http",
    #                   "address": "http://127.0.0.1:8178"}

This one "sees" a 1 x 0.6 m free area with a 20 cm box in the middle. A real device replaces
`grid()` with its sensor. Format: devices/README.md, "Serve tiles over HTTP".
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RES = 0.05  # m per cell: 0.05 times a power of two


def grid():
    """-1 not seen, 0 free and seen, 1 obstacle. Row 0 = south (smallest y)."""
    cells = [[0] * 20 for _ in range(12)]  # 1.0 m (x) by 0.6 m (y)
    for i in range(4, 8):
        for j in range(8, 12):
            cells[i][j] = 1
    return cells


def make_handler(x0, y0):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (http.server names it)
            if self.path != '/map':
                return self.send_error(404)
            body = {'t': time.time(), 'res': RES, 'x0': x0, 'y0': y0, 'cells': grid()}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8178)
    p.add_argument('--x0', type=float, default=1.0)  # map position of cell [0][0], metres
    p.add_argument('--y0', type=float, default=0.5)
    a = p.parse_args()
    ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(a.x0, a.y0)).serve_forever()
