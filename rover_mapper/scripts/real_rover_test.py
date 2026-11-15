"""Drive the real RaspRover through measured moves while the rover mapper maps its camera,
and compare the mapper's trajectory with the rover's own odometry (wheels + gyro).

The rover runs Raphaël's test server (polaris branch `rover-teleop`, rover/rover_server.py):
GET /state, POST /sequence, POST /stop, POST /reset_odom. The mapper is run.py started with
`--source <rover>/video_feed --bridge-port 7400`; this script reads its pose stream.

    python scripts/real_rover_test.py --rover http://IP:8080 --mapper 127.0.0.1:7400 \
        --seq "F50 L90 F50 L90 F50 L90 F50 L90"
    python scripts/real_rover_test.py --rover http://IP:8080 --check     # state only, no motion

Steps: F<cm> forward, B<cm> backward, L<deg> left turn, R<deg> right turn.
Safety: refuses to start if the rover link, gyro or battery are not OK or if the rover is
already moving; any error or Ctrl-C sends POST /stop; the whole run has a timeout. The
rover server itself stops 0.5 s after the last command and abandons a blocked move.
Writes out/real_rover/<time>/{log.json, traj.csv, traj.png}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import struct
import threading
import time
import urllib.request

import numpy as np


def http(base: str, path: str, body: dict | None = None, timeout: float = 3.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'},
                                 method='GET' if body is None else 'POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b'{}')


def parse_seq(text: str) -> list[dict]:
    steps = []
    for tok in text.split():
        m = re.fullmatch(r'([FBLR])(\d+(?:\.\d+)?)', tok.upper())
        if not m:
            raise SystemExit(f'bad step {tok!r}: use F50 B30 L90 R45')
        k, n = m.group(1), float(m.group(2))
        if k in 'FB' and n > 150:
            raise SystemExit(f'{tok}: moves are limited to 150 cm per step in this test')
        steps.append({'move': n / 100 * (1 if k == 'F' else -1)} if k in 'FB'
                     else {'turn': n * (1 if k == 'L' else -1)})
    return steps


class MapperStream:
    """Reads the mapper's pose stream (rovermap/bridge.py wire format) in a thread."""

    def __init__(self, hostport: str) -> None:
        host, port = hostport.rsplit(':', 1)
        self.sock = socket.create_connection((host, int(port)), timeout=5)
        self.sock.settimeout(1.0)
        self.samples: list[tuple[float, float, float, float, bool, float]] = []
        self.error: str | None = None
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _read(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n and not self._stop:
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError('mapper stream closed')
            buf += chunk
        return buf

    def _run(self) -> None:
        try:
            while not self._stop:
                (h,) = struct.unpack('>I', self._read(4))
                hdr = json.loads(self._read(h))
                self._read(12 * int(hdr.get('n', 0)))          # points: not needed here
                self.samples.append((time.time(), hdr['x'], hdr['y'], hdr['yaw'],
                                     bool(hdr['lost']), float(hdr.get('quality', 0))))
        except Exception as e:  # noqa: BLE001 - reported in the log
            self.error = str(e)

    def close(self) -> None:
        self._stop = True
        self.sock.close()


def preflight(st: dict, min_battery: float) -> list[str]:
    problems = []
    link = st.get('link', {})
    if not link.get('connected'):
        problems.append(f"ESP32 link not connected ({link.get('error')})")
    elif (link.get('rx_age') or 99) > 0.5:
        problems.append(f"no ESP32 feedback for {link.get('rx_age'):.1f} s")
    if not st.get('odom', {}).get('gyro_ready'):
        problems.append('gyro not ready (keep the rover still after start)')
    b = st.get('battery_v')
    if b is not None and b < min_battery:
        problems.append(f'battery low: {b:.2f} V < {min_battery} V')
    ctrl = st.get('ctrl', {})
    if ctrl.get('mode') not in ('idle', None) or ctrl.get('queued'):
        problems.append(f"rover busy (mode {ctrl.get('mode')})")
    return problems


def rel_to_start(x, y, yaw, x0, y0, yaw0):
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    dx, dy = x - x0, y - y0
    return c * dx - s * dy, s * dx + c * dy, (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi


def umeyama_2d(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ sgn @ vt
    var = (a ** 2).sum() / len(src)
    s = float((d * np.diag(sgn)).sum() / var) if var > 1e-9 else 1.0
    return s, r, md - s * r @ ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rover', required=True, help='rover server, e.g. http://192.168.1.20:8080')
    ap.add_argument('--mapper', default='127.0.0.1:7400', help='run.py --bridge-port stream')
    ap.add_argument('--seq', default='F50 L90 F50 L90 F50 L90 F50 L90')
    ap.add_argument('--check', action='store_true', help='print the rover state and stop there')
    ap.add_argument('--no-mapper', action='store_true', help='drive and log odometry only')
    ap.add_argument('--min-battery', type=float, default=10.5)
    ap.add_argument('--settle', type=float, default=3.0, help='s of stillness before moving')
    ap.add_argument('--timeout', type=float, default=180.0)
    args = ap.parse_args()

    st = http(args.rover, '/state')
    odo = st.get('odom', {})
    print(f"rover: link {st.get('link', {}).get('connected')}, battery {st.get('battery_v')} V, "
          f"gyro ready {odo.get('gyro_ready')}, mode {st.get('ctrl', {}).get('mode')}, "
          f"camera {st.get('camera')}, sim {st.get('sim')}")
    problems = preflight(st, args.min_battery)
    for p in problems:
        print('  NOT OK:', p)
    if args.check or problems:
        return

    steps = parse_seq(args.seq)
    mapper = None if args.no_mapper else MapperStream(args.mapper)
    out = time.strftime('out/real_rover/%Y%m%d_%H%M%S')
    os.makedirs(out, exist_ok=True)
    odom_log: list[tuple[float, float, float, float]] = []
    states: list[dict] = []
    t_start = time.time()
    try:
        http(args.rover, '/reset_odom', {})
        print(f'still for {args.settle:.0f} s (mapper initialises)...')
        time.sleep(args.settle)
        if mapper is not None:
            if not mapper.samples:
                raise RuntimeError(f'no pose from the mapper at {args.mapper} ({mapper.error})')
            m0 = mapper.samples[-1]
        t_go = time.time()
        http(args.rover, '/sequence', {'steps': steps})
        print('sequence sent:', args.seq)
        seen_busy = False
        while True:
            st = http(args.rover, '/state')
            t = time.time()
            o = st['odom']
            odom_log.append((t, o['x'], o['y'], math.radians(o['heading_deg'])))
            ctrl = st['ctrl']
            states.append({'t': t - t_go, 'mode': ctrl['mode'], 'v': o['v'],
                           'battery_v': st.get('battery_v')})
            busy = ctrl['mode'] != 'idle' or ctrl['queued'] or ctrl['current']
            seen_busy |= bool(busy)
            if seen_busy and not busy:
                break
            if not seen_busy and t - t_go > 3.0:
                raise RuntimeError('the rover never started the sequence')
            if t - t_start > args.timeout:
                raise RuntimeError('timeout')
            time.sleep(0.1)
        time.sleep(1.0)                                   # let the mapper catch up
        history = st['ctrl'].get('history', [])
    except BaseException as e:
        try:
            http(args.rover, '/stop', {})
            print('STOP sent')
        finally:
            if mapper is not None:
                mapper.close()
        if isinstance(e, KeyboardInterrupt):
            print('interrupted')
            return
        raise
    if mapper is not None:
        mapper.close()

    od = np.array(odom_log)
    path_odom = float(np.linalg.norm(np.diff(od[:, 1:3], axis=0), axis=1).sum())
    res = {'seq': args.seq, 'steps': steps, 'duration_s': float(od[-1, 0] - od[0, 0]),
           'odom_final': {'x': float(od[-1, 1]), 'y': float(od[-1, 2]),
                          'heading_deg': math.degrees(od[-1, 3])},
           'odom_path_m': path_odom, 'rover_history': history[-len(steps):]}
    rows = [('t', 'odom_x', 'odom_y', 'odom_yaw', 'map_x', 'map_y', 'map_yaw', 'map_lost')]
    if mapper is not None:
        ms = [s for s in mapper.samples if s[0] >= t_go - 0.5]
        mp = np.array([[s[0], *rel_to_start(s[1], s[2], s[3], m0[1], m0[2], m0[3])] for s in ms])
        lost = np.array([s[4] for s in ms])
        res['mapper'] = {'samples': len(ms), 'lost_fraction': float(lost.mean()) if len(ms) else None,
                         'stream_error': mapper.error}
        if len(mp) >= 3:
            # Mapper pose at each odometry time (nearest earlier sample: the mapper lags).
            idx = np.clip(np.searchsorted(mp[:, 0], od[:, 0]) - 1, 0, len(mp) - 1)
            mo = mp[idx]
            path_map = float(np.linalg.norm(np.diff(mp[:, 1:3], axis=0), axis=1).sum())
            s, r, t = umeyama_2d(mo[:, 1:3], od[:, 1:3])
            ali = (s * (r @ mo[:, 1:3].T)).T + t
            err_sim = np.linalg.norm(ali - od[:, 1:3], axis=1)
            err_raw = np.linalg.norm(mo[:, 1:3] - od[:, 1:3], axis=1)
            res['mapper'].update({
                'final': {'x': float(mp[-1, 1]), 'y': float(mp[-1, 2]),
                          'heading_deg': math.degrees(mp[-1, 3])},
                'path_m': path_map,
                'scale_map_over_odom': path_map / path_odom if path_odom > 1e-3 else None,
                'align_scale_odom_over_map': s,
                'error_after_similarity_rms_cm': float(np.sqrt((err_sim ** 2).mean()) * 100),
                'error_raw_rms_cm': float(np.sqrt((err_raw ** 2).mean()) * 100),
                'heading_final_diff_deg': math.degrees(
                    (mp[-1, 3] - od[-1, 3] + math.pi) % (2 * math.pi) - math.pi),
            })
            for k in range(len(od)):
                rows.append((od[k, 0] - t_go, *od[k, 1:4], *mo[k, 1:4], int(lost[min(idx[k], len(lost) - 1)])))
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(od[:, 1], od[:, 2], 'k-', label='rover odometry (wheels + gyro)')
                ax.plot(mp[:, 1], mp[:, 2], 'c-', alpha=0.6, label='mapper (raw)')
                ax.plot(ali[:, 0], ali[:, 1], 'm--', label=f'mapper x{s:.2f} (aligned)')
                ax.set_aspect('equal')
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=8)
                ax.set_title(args.seq)
                fig.savefig(os.path.join(out, 'traj.png'), dpi=120)
            except Exception as e:  # noqa: BLE001
                res['plot_error'] = str(e)
    with open(os.path.join(out, 'traj.csv'), 'w') as f:
        f.write('\n'.join(','.join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in r)
                          for r in rows) + '\n')
    json.dump({**res, 'states': states}, open(os.path.join(out, 'log.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print('saved in', out)


if __name__ == '__main__':
    main()
