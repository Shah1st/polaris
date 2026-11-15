"""Serve the mission page (/mission) on sample data, without robot, camera or mission.

    python teleop/demo_mission.py [--port 8150] [--fixe]

Starts on 127.0.0.1 only: a stand-in rover (mock_rover.py, port + 1) whose snapshots feed the
camera panel, a stand-in perception debug picture (port + 2), and teleop/server.py with
--suivi pointing at a document that this script rewrites every second (a rover driving to the
victim in the 2.4 m arena, with a stop to ask the drone), the same way lancer_overhead.py
--suivi writes it: to a temporary file, then renamed. With --fixe, the sample document
teleop/mission_exemple.json is served as it is. Open http://127.0.0.1:<port>/mission.

    python teleop/demo_mission.py --ecrire-exemple   # regenerate mission_exemple.json
"""

import argparse
import json
import math
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mock_rover
import server

HERE = os.path.dirname(os.path.abspath(__file__))
RES, N = 0.05, 48  # 2.4 m arena at the fusion's finest cell
VICTIME = (2.0, 2.05)
# Corners of the planned path around the drone's obstacle block.
COINS = [(0.3, 0.3), (0.85, 0.45), (1.55, 0.55), (1.85, 1.0), (1.95, 1.55), VICTIME]


def _grille() -> list[str]:
    rows = []
    for i in range(N):  # row 0 = south
        y = (i + 0.5) * RES
        row = []
        for j in range(N):
            x = (j + 0.5) * RES
            if 1.0 <= x <= 1.4 and 0.8 <= y <= 1.6:
                c = '1'  # seen by the drone
            elif 1.55 <= x <= 1.75 and 1.65 <= y <= 1.85:
                c = '2'  # seen by the rover only
            elif 0.2 <= x <= 0.65 and 1.8 <= y <= 2.3:
                c = '?'  # under a table: nobody saw it
            elif x < 0.1 or y < 0.1 or x > 2.3 or y > 2.3:
                c = '?'
            else:
                c = '0'
            row.append(c)
        rows.append(''.join(row))
    return rows


def _dense(points, step=0.05):
    out = [points[0]]
    for a, b in zip(points, points[1:], strict=False):
        n = max(1, round(math.dist(a, b) / step))
        out += [
            (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)
        ]
    return out


def exemple(k: int) -> dict:
    """The mission document after k seconds of the sample drive (loops)."""
    route = _dense(COINS)
    n = len(route)
    k %= n + 6  # a few seconds of 'arrive' before starting again
    i = min(k, n - 1)
    x, y = route[i]
    nx, ny = route[min(i + 1, n - 1)]
    cap = math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else math.pi / 2
    journal = [f'{j:3d} s  avance ({route[j][0]:.2f}, {route[j][1]:.2f})' for j in range(i + 1)]
    etat, action = 'en_cours', 'SUIVRE'
    raison = 'chemin libre vers la victime (A* sur la carte fusionnee)'
    if 20 <= i < 24:
        action = 'DEMANDER_DRONE'
        raison = 'obstacle vu par le rover pres de (1.65, 1.75) : le drone regarde de plus pres'
        journal.append(f'{i:3d} s  obstacle rover en (1.65, 1.75), zoom du drone demande')
    elif i >= n - 1:
        etat, action, raison = 'arrive', 'ARRIVE', 'a 0.00 m de la victime : mission accomplie'
        journal.append(f'{i:3d} s  ARRIVE pres de la victime')
    restant = math.dist((x, y), VICTIME)
    if etat == 'en_cours':
        raison += f' - reste {restant:.2f} m'
    return {
        't': time.time(),
        'etat': etat,
        'rover': [round(x, 3), round(y, 3), round(cap, 3)],
        'victime': list(VICTIME),
        'zone': [0.0, 0.0, 2.4, 2.4],
        'decision': {'action': action, 'raison': raison},
        'chemin': [[round(a, 3), round(b, 3)] for a, b in route[i:]],
        'traj': [[round(a, 3), round(b, 3)] for a, b in route[: i + 1]],
        'journal': journal[-30:],
        'grille': {'x0': 0.0, 'y0': 0.0, 'res': RES, 'lignes': _grille()},
    }


def ecrire(path: str, doc: dict) -> None:
    """Atomically, as lancer_overhead.py --suivi does: the page never reads half a file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def perception_debug(port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = mock_rover.picture('perception (exemple)')
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8150)
    ap.add_argument('--fixe', action='store_true', help='serve mission_exemple.json unchanged')
    ap.add_argument('--ecrire-exemple', action='store_true', help='write mission_exemple.json')
    args = ap.parse_args()
    exemple_path = os.path.join(HERE, 'mission_exemple.json')
    if args.ecrire_exemple:
        doc = exemple(14)
        doc['t'] = 1789000000.0  # fixed, so the file only changes with the code
        with open(exemple_path, 'w') as f:  # one line per field: readable, and diffs stay small
            f.write(
                '{\n' + ',\n'.join(f' {json.dumps(k)}: {json.dumps(v)}' for k, v in doc.items())
            )
            f.write('\n}\n')
        return
    rover = mock_rover.MockRover(args.port + 1)
    perception_debug(args.port + 2)
    if args.fixe:
        suivi = exemple_path
    else:
        suivi = os.path.join(tempfile.mkdtemp(prefix='mission-'), 'suivi.json')

        def animer():
            k = 0
            while True:
                ecrire(suivi, exemple(k))
                k += 1
                time.sleep(1.0)

        threading.Thread(target=animer, daemon=True).start()
    hub = server.CameraHub(rover.url + '/video_feed')
    handler = server.make_handler(
        server.Robot(rover.url),
        hub,
        8081,
        '',
        {},
        'http://127.0.0.1:9',  # no fusion
        suivi,
        f'http://127.0.0.1:{args.port + 2}/debug.jpg',
    )
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    srv.daemon_threads = True
    print(f'mission page: http://127.0.0.1:{args.port}/mission  (document: {suivi})', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
