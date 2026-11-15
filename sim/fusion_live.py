"""Live map fusion on the forest simulation: Paco's merging_map fed with the simulator's data.

    python sim/fusion_live.py [--sim http://127.0.0.1:5050] [--scene <dir with maps.npz>]
                              [--port 8094] [--period 1.0] [--range 2.0]

About once per `--period` seconds:

1. Drone: the simulator's `/grid/overhead.png` (what the drone sees from its survey altitude, -1
   under the canopy), cut by `decouper()` into drone tiles for `CarteFusion.recevoir()`. Fetched
   at start, then again every minute (it only changes if the simulator restarts).
2. Rover: one `/snapshot` (true pose, rover camera depth, simulated D405), turned into world
   points by the active source, classified per 5 cm cell (sim/fusion_feed.py), cut by
   `decouper()` into rover tiles for `recevoir()`. One source at a time, chosen with --source or
   from the page (POST /source); switching empties the rover layer of the map:
   - `camera`: the rover camera's EXACT depth from the simulator, cut at `--range` m;
   - `d405`: the simulated D405 (sensor model with noise and holes, 0.07-1 m);
   - `da3`: the rover mapper's DA3 estimate, not connected yet.
   None of them is the mapper's estimate today.
3. Decision: `CarteFusion.decider(rover, goal)` towards the scene's goal, again whenever the path
   ahead stops passing `chemin_valide(..., horizon=3.0)` or the rover is more than 1 m off the
   path (no more often than every `--min-gap` s), and at least every `--replan` s. If it asks
   the drone to look closer (DEMANDER_DRONE), the drone answers with the requested tiles from
   the same grid, as Paco's simulated drone does (demo_fusion.Simu.zoom_drone).
4. Pictures and a status, served on 127.0.0.1 only: GET /rover.png /drone.png /fused.png,
   /depth_d405.png (D405 sensor model next to the exact depth of the same view, one colour
   scale), /depth_camera.png (rover camera, exact depth), /status.json (decision, scan and D405
   figures, timings). POST /source {"source": "camera" | "d405"}.

The simulated rover drives its own route (run_world.py --auto): nothing is sent to it, the path
shown is what the fusion would do from where the rover is.
"""

import argparse
import io
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'merging_map'))

from fusion_carte import CarteFusion, decouper, reduire  # noqa: E402

import fusion_feed as ff  # noqa: E402

# The forest scene on the team's VM (read only: goal and start); --scene or FOREST_SCENE elsewhere.
DEFAULT_SCENE = os.environ.get('FOREST_SCENE', '/home/romaindcrl/rovermap/out/forest/A_false')
PICTURES = ('rover.png', 'drone.png', 'fused.png', 'depth_d405.png', 'depth_camera.png')


def http_get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def read_snapshot(sim: str) -> dict:
    """One `/snapshot` of the simulator (rover camera, simulated D405, true pose), as arrays."""
    with np.load(io.BytesIO(http_get(sim.rstrip('/') + '/snapshot'))) as z:
        return {k: z[k] for k in z.files}


# Rover point sources. Each turns the latest frame into a Scan (world points + rover pose); one
# is active at a time, chosen from the page. A source with its own feed (the mapper's bridge)
# may ignore the frame. Only the simulator ones exist yet.
class SimCameraDepth:
    """Exact depth of the rover camera (simulator truth), cut at a sensor range."""

    id = 'camera'
    label = 'caméra du rover : profondeur exacte du simulateur, portée bornée'
    available = True

    def __init__(self, max_range: float) -> None:
        self.max_range = max_range
        self.label = f'{self.label} à {max_range:g} m'

    def read(self, frame: dict) -> ff.Scan:
        return ff.scan_from_depth(
            frame['depth'],
            frame['K'],
            frame['T'],
            tuple(frame['rover']),
            float(frame['t']),
            self.max_range,
            ff.CAM_HEIGHT,
        )


class SimD405:
    """The simulated RealSense D405: its sensor model (noise, holes, 0.07-1 m), not the truth.

    It sees no drop edge: past an edge its pixels get no measurement, and no measurement is not
    taken as evidence of anything.
    """

    id = 'd405'
    label = 'D405 simulée (modèle de capteur, courte portée 0,07-1 m)'
    available = True

    def read(self, frame: dict) -> ff.Scan:
        return ff.scan_from_depth(
            frame['d405_depth'],
            frame['d405_K'],
            frame['d405_T'],
            tuple(frame['rover']),
            float(frame['t']),
            float(frame['d405_range'][1]),
            float(frame['d405_mount'][2]),  # camera height: its mount above the ground
        )


class MapperDA3:
    """Placeholder: the rover mapper's DA3 estimate, through its bridge (guidance_rover_vo
    protocol: pose and points in base_link). To be plugged in once the forest mapper is fixed."""

    id = 'da3'
    label = 'estimation DA3 du mappeur (à venir)'
    available = False

    def read(self, frame: dict) -> ff.Scan:
        raise NotImplementedError('DA3 mapper source not connected yet')


class SimDrone:
    """The drone's grid from the simulator (`/grid/overhead.png`) and its origin, from /state."""

    def __init__(self, sim: str) -> None:
        self.sim = sim.rstrip('/')

    def read(self) -> tuple[np.ndarray, tuple[float, float]]:
        state = json.loads(http_get(self.sim + '/state'))
        origin = tuple(float(v) for v in state['grid']['origin'])
        if abs(state['grid']['res'] - ff.RES) > 1e-9:
            raise ValueError(f'grid cell {state["grid"]["res"]} m, the fusion expects {ff.RES}')
        return ff.drone_grid_from_png(http_get(self.sim + '/grid/overhead.png')), origin


def remaining_path(path, pos, skip: float = 0.3):
    """The part of `path` still ahead of a rover at `pos`, and how far the rover is from the path.

    The ahead part starts at the path point closest to `pos`, without the points within `skip` m
    of the rover: the simulated rover drives its own route, often closer to a tree than the
    fusion's safety margin, and its own surroundings must not count as "path blocked" (Paco's
    mission avoids the same trap with its `sortie` distance in premier_point_bloque).
    """
    if not path:
        return [], math.inf
    d = [math.hypot(x - pos[0], y - pos[1]) for x, y in path]
    k = int(np.argmin(d))
    ahead = [tuple(p) for p in path[k:] if math.hypot(p[0] - pos[0], p[1] - pos[1]) > skip]
    return ahead, d[k]


class LiveFusion:
    def __init__(self, args) -> None:
        self.args = args
        scene = np.load(os.path.join(args.scene, 'maps.npz'))
        self.goal = tuple(args.goal) if args.goal else tuple(float(v) for v in scene['goal'])
        self.start = tuple(float(v) for v in scene['start'])
        self.sources = {
            src.id: src for src in (SimCameraDepth(args.range), SimD405(), MapperDA3())
        }
        self.source_id = self.wanted_source = args.source
        self.drone_src = SimDrone(args.sim)
        self.lock = threading.Lock()
        self.pictures: dict[str, tuple[bytes, float, int]] = {}
        self.status: dict = {'etat': 'démarrage'}
        self.seq = 0
        self.reset()

    def reset(self) -> None:
        self.carte = CarteFusion()
        self.drone = None  # (matrix, origin, hash)
        self.drone_base = None  # drone picture without the markers
        self.t_drone = -math.inf
        self.decision, self.t_decision, self.n_decisions = None, -math.inf, 0
        self.why = self.why_next = ''
        self.trail: list[tuple[float, float]] = []
        self.last_t = None
        self.zooms: list = []

    # ---- feeding ----------------------------------------------------------------------------
    def feed_drone(self, t: float) -> None:
        m, origin = self.drone_src.read()
        key = hash(m.tobytes())
        if self.drone is not None and self.drone[2] == key:
            return
        level = self.args.drone_level
        f = 2 ** (self.carte.zmax - level)
        coarse = reduire(m, f)
        for msg in decouper(coarse, origin[0], origin[1], level, self.carte, 'drone', t):
            self.carte.recevoir(msg)
        self.drone = (m, origin, key)
        self.drone_base = None
        self.decision, self.why_next = None, 'nouvelle carte du drone'

    def answer_zoom(self, keys, t: float) -> None:
        """The drone flies lower over the requested tiles: same grid, at the requested level."""
        m, origin, _ = self.drone
        for z in sorted({k[0] for k in keys}):
            coarse = reduire(m, 2 ** (self.carte.zmax - z))
            for msg in decouper(coarse, origin[0], origin[1], z, self.carte, 'drone', t):
                if (msg['z'], msg['x'], msg['y']) in keys:
                    self.carte.recevoir(msg)
        self.zooms.extend(keys)

    def set_source(self, source_id: str) -> None:
        """Asked from the page; applied by the loop at its next cycle."""
        src = self.sources.get(source_id)
        if src is None or not src.available:
            raise ValueError(f'source not available: {source_id!r}')
        self.wanted_source = source_id

    def switch_source(self) -> None:
        """Change the rover's point source. The rover layer of the fusion is emptied (through
        Paco's restaurer_rover with an empty snapshot) so the map shows what this source alone
        brings; the drone layer stays."""
        print(f'rover source: {self.source_id} -> {self.wanted_source}', flush=True)
        self.source_id = self.wanted_source
        self.carte.restaurer_rover(({}, {}))
        self.decision, self.why_next = None, 'nouvelle source des points du rover'

    # ---- one cycle --------------------------------------------------------------------------
    def step(self) -> None:
        timing = {}
        t0 = time.monotonic()
        frame = read_snapshot(self.args.sim)
        timing['snapshot_s'] = time.monotonic() - t0
        if self.wanted_source != self.source_id:
            self.switch_source()
        scan = self.sources[self.source_id].read(frame)
        timing['capteur_s'] = time.monotonic() - t0
        if self.last_t is not None and scan.t < self.last_t - 1.0:
            print(
                f'simulator time went back ({self.last_t:.1f} -> {scan.t:.1f}): reset', flush=True
            )
            self.reset()
        self.last_t = scan.t
        if time.monotonic() - self.t_drone > 60.0 or self.drone is None:
            self.feed_drone(scan.t)
            self.t_drone = time.monotonic()

        t1 = time.monotonic()
        local = ff.classify_scan(scan)
        msgs = decouper(
            local.data, local.x0, local.y0, self.carte.zmax, self.carte, 'rover', scan.t
        )
        for msg in msgs:
            self.carte.recevoir(msg)
        timing['rover_s'] = time.monotonic() - t1
        pos = scan.pose[:2]
        if not self.trail or math.dist(self.trail[-1], pos) > 0.1:
            self.trail.append(pos)
            self.trail = self.trail[-3000:]

        t2 = time.monotonic()
        grid = self.maybe_decide(pos, scan.t)
        timing['decision_s'] = time.monotonic() - t2

        t3 = time.monotonic()
        self.render(scan, local, grid)
        depth = self.render_depth(frame)
        timing['images_s'] = time.monotonic() - t3
        timing['total_s'] = time.monotonic() - t0
        self.publish_status(scan, local, len(msgs), timing, depth)

    def maybe_decide(self, pos, t: float) -> None:
        dec = self.decision
        why = ''
        if dec is None:
            why = self.why_next or 'première décision'
        elif time.monotonic() - self.t_decision < self.args.min_gap:
            return None  # the simulated rover does not follow the path: do not replan every cycle
        elif time.monotonic() - self.t_decision >= self.args.replan:
            why = f'périodique ({self.args.replan:g} s)'
        elif dec['action'] == 'DEMANDER_DRONE':
            why = 'le drone a répondu'
        else:
            rest, off = remaining_path(dec.get('chemin') or [], pos)
            if off > 1.0:
                why = 'rover à plus de 1 m du chemin'
            elif len(rest) >= 2 and not self.carte.chemin_valide(rest, horizon=3.0):
                why = 'obstacle sur les 3 m de chemin devant (chemin_valide)'
        if not why:
            return None
        dec = self.carte.decider(pos, self.goal)
        self.decision, self.t_decision, self.why = dec, time.monotonic(), why
        self.why_next = ''
        self.n_decisions += 1
        if dec['action'] == 'DEMANDER_DRONE':
            self.answer_zoom(dec['tuiles'], t)
        print(
            time.strftime('%H:%M:%S'),
            f'decision #{self.n_decisions} {dec["action"]} ({why})',
            f'at ({pos[0]:.1f}, {pos[1]:.1f})',
            f'path {len(dec.get("chemin") or [])} pts',
            flush=True,
        )
        # the grid the decision was made on is the current map, unless the drone just added tiles
        return dec['grille'] if dec['action'] != 'DEMANDER_DRONE' else None

    # ---- output -----------------------------------------------------------------------------
    def render(self, scan: ff.Scan, local: ff.LocalGrid, fresh_grid=None) -> None:
        g = (
            fresh_grid
            if fresh_grid is not None
            else self.carte.grille(*self.carte.zone_survolee())
        )
        gx0, gy0 = g.j0 * g.res, g.i0 * g.res
        half = 4.0
        x, y, _ = scan.pose
        window = (x - half, y - half, x + half, y + half)
        # the rover's own layer, around the rover only
        i0 = max(0, math.floor((y - half) / g.res) - g.i0)
        j0 = max(0, math.floor((x - half) / g.res) - g.j0)
        n = round(2 * half / g.res) + 2
        sub = (slice(i0, i0 + n), slice(j0, j0 + n))
        rover_val = np.where(g.src[sub] == 2, g.val[sub], ff.UNSEEN)

        rover_img, _ = ff.render_rover_view(
            scan, local, (rover_val, gx0 + j0 * g.res, gy0 + i0 * g.res), half=half
        )
        m, origin, _ = self.drone
        x1, y1 = origin[0] + m.shape[1] * ff.RES, origin[1] + m.shape[0] * ff.RES
        view = ff.View(origin[0], origin[1], x1, y1, self.args.map_ppm)
        if self.drone_base is None:  # the drone grid only changes with a new survey
            self.drone_base = ff.render_drone_view(m, origin, view)
        drone_img = self.drone_base.copy()
        ff.draw_markers(drone_img, view, scan.pose, self.goal, self.start, window)
        dec = self.decision or {}
        tiles = dec.get('tuiles') or dec.get('a_verifier') or []
        fused_img = ff.render_fused_view(
            g.val,
            g.src,
            gx0,
            gy0,
            view,
            scan.pose,
            self.goal,
            self.start,
            window,
            path=dec.get('chemin'),
            trail=self.trail,
            tiles=tiles,
            tile_rect=self.carte.rect_tuile,
            point=dec.get('point') or dec.get('cible'),
        )
        pngs = [ff.encode_png(img) for img in (rover_img, drone_img, fused_img)]
        now = time.time()
        with self.lock:
            self.seq += 1
            for name, png in zip(PICTURES[:3], pngs, strict=True):
                self.pictures[name] = (png, now, self.seq)

    def render_depth(self, frame: dict) -> dict:
        """Depth comparison pictures from the same snapshot, and the D405 figures."""
        d_range = tuple(float(v) for v in frame['d405_range'])
        stats = ff.depth_stats(frame['d405_depth'], frame['d405_depth_true'], d_range)
        pair = ff.render_depth_pair(
            frame['d405_depth'],
            frame['d405_depth_true'],
            0.0,
            d_range[1],
            ('D405 : modele capteur', 'meme vue : profondeur exacte'),
            ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
        cam = frame['depth']
        cam_max = 8.0
        single = ff.render_depth_single(
            cam, 0.0, cam_max, 'camera rover : profondeur exacte', ticks=(0, 2, 4, 6, 8)
        )
        within = np.isfinite(cam) & (cam > 0) & (cam <= self.args.range)
        stats['camera_rover_dans_portee_pct'] = round(100.0 * within.mean(), 1)
        stats['camera_rover_echelle_m'] = cam_max
        stats['d405_portee_m'] = list(d_range)
        pngs = ff.encode_png(pair, 128), ff.encode_png(single, 128)
        now = time.time()
        with self.lock:
            self.pictures['depth_d405.png'] = (pngs[0], now, self.seq)
            self.pictures['depth_camera.png'] = (pngs[1], now, self.seq)
        return stats

    def publish_status(self, scan, local, n_tiles, timing, depth) -> None:
        dec = self.decision or {}
        path = dec.get('chemin') or []
        length = sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1))
        status = {
            'etat': 'ok',
            'genere_a': time.time(),
            'seq': self.seq,
            't_sim': scan.t,
            'rover': {'x': scan.pose[0], 'y': scan.pose[1], 'cap': scan.pose[2]},
            'objectif': self.goal,
            'source_rover': self.sources[self.source_id].label,
            'source_active': self.source_id,
            'sources': [
                {'id': s.id, 'label': s.label, 'disponible': s.available}
                for s in self.sources.values()
            ],
            'profondeur': depth,
            'portee_m': scan.sensor_range,
            'scan': local.stats | {'tuiles_envoyees': n_tiles},
            'drone': {
                'source': 'grille du drone du simulateur (/grid/overhead.png), survol initial',
                'niveau_z': self.args.drone_level,
                'tuiles': len(self.carte.tuiles['drone']),
                'zooms_demandes': [list(k) for k in self.zooms],
            },
            'tuiles_rover': len(self.carte.tuiles['rover']),
            'decision': {
                'action': dec.get('action'),
                'pourquoi': self.why,
                'numero': self.n_decisions,
                'age_s': time.monotonic() - self.t_decision,
                'longueur_chemin_m': round(length, 1),
                'a_verifier': [list(k) for k in dec.get('a_verifier') or []],
                'tuiles': [list(k) for k in dec.get('tuiles') or []],
                'distance_restante_m': dec.get('distance_restante'),
                'raison': dec.get('raison'),
            },
            'durees': {k: round(v, 3) for k, v in timing.items()},
        }
        with self.lock:
            self.status = status

    def error(self, text: str) -> None:
        with self.lock:
            self.status = dict(self.status, etat='erreur', erreur=text, erreur_a=time.time())


def make_handler(live: LiveFusion):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            name = self.path.split('?')[0].lstrip('/')
            with live.lock:
                pic = live.pictures.get(name)
                status = dict(live.status)
            if name == 'status.json':
                if 'genere_a' in status:
                    status['age_s'] = round(time.time() - status['genere_a'], 2)
                body, ctype, age, seq = json.dumps(status).encode(), 'application/json', 0.0, 0
            elif pic is not None:
                body, ctype = pic[0], 'image/png'
                age, seq = time.time() - pic[1], pic[2]
            else:
                self.send_error(404 if name not in PICTURES else 503)
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Age-Ms', str(round(1000 * age)))
            self.send_header('X-Seq', str(seq))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split('?')[0] != '/source':
                self.send_error(404)
                return
            n = min(int(self.headers.get('Content-Length') or 0), 1024)
            try:
                live.set_source(str(json.loads(self.rfile.read(n) or b'{}').get('source')))
                code, body = 200, {'ok': True, 'source': live.wanted_source}
            except (ValueError, AttributeError) as e:
                code, body = 400, {'error': str(e)}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sim', default='http://127.0.0.1:5050')
    ap.add_argument('--scene', default=DEFAULT_SCENE, help='scene dir (maps.npz: goal, start)')
    ap.add_argument('--goal', type=float, nargs=2, default=None, help='goal x y (m)')
    ap.add_argument('--port', type=int, default=8094)
    ap.add_argument('--period', type=float, default=1.0, help='seconds between cycles')
    ap.add_argument('--range', type=float, default=2.0, help='rover sensor range (m)')
    ap.add_argument('--source', default='camera', choices=('camera', 'd405'), help='rover points')
    ap.add_argument('--replan', type=float, default=10.0, help='decide at least this often (s)')
    ap.add_argument('--min-gap', type=float, default=3.0, help='and at most this often (s)')
    ap.add_argument('--drone-level', type=int, default=4, help='fusion level of the drone tiles')
    ap.add_argument('--map-ppm', type=float, default=8.0, help='pixels per metre, map pictures')
    args = ap.parse_args()

    live = LiveFusion(args)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(live))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'fusion live: http://127.0.0.1:{args.port}/status.json  sim {args.sim}', flush=True)
    n = 0
    while True:
        t0 = time.monotonic()
        try:
            live.step()
            n += 1
            if n % 30 == 1:
                print(time.strftime('%H:%M:%S'), json.dumps(live.status['durees']), flush=True)
        except Exception as e:  # simulator restarting, network hiccup: keep going
            live.error(f'{type(e).__name__}: {e}'[:300])
            print(time.strftime('%H:%M:%S'), 'error:', traceback.format_exc(limit=3), flush=True)
            time.sleep(2.0)
        time.sleep(max(0.0, args.period - (time.monotonic() - t0)))


if __name__ == '__main__':
    main()
