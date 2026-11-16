"""Forest search-and-rescue demo video: drones and rovers look for a victim nobody has located.

    MUJOCO_GL=egl python sim/mission/demo_video.py TERRAIN_DIR OUT.mp4 [--rovers 2] [--drones 1]
                                                   [--seed 3] [--speed 5] [--fps 20]

TERRAIN_DIR comes from `terrain.py` (any 3D model). What is real and what is simulated:
- REAL: the map fusion and the planner are merging_map's (`CarteFusion.recevoir`, `decider`,
  `chemin_valide`), fed with tiles tagged per device, and the pictures use the dashboard's colours.
- SIMULATED: the sensors. A drone "lidar" reveals the true map in a disc under the drone (nothing
  under tree crowns); a rover "depth camera" reveals it up to 5 m in a 100 degree cone, stopped by
  obstacles. The victim is "detected" by range and line of sight. Robots move kinematically.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'merging_map'))

import fusion_feed as ff  # noqa: E402
from fusion_carte import CarteFusion, decouper, reduire  # noqa: E402
from terrain import Terrain, mjcf_assets  # noqa: E402

ROVER_MESHES = os.environ.get('ROVER_MESHES', os.path.expanduser('~/rovermap/sim/assets/rasp_rover/meshes'))
DRONE_DIR = os.environ.get('DRONE_DIR', os.path.expanduser('~/polaris-mission/assets/skydio_x2/assets'))
BASE_Z = 0.04 - 0.00675
VIS = 2.2  # robots are drawn this many times larger than life, to be visible in a 40 m forest
ROVER_RANGE, ROVER_FOV, ROVER_SPEED = 5.0, math.radians(100), 0.55
DRONE_RADIUS, DRONE_SPEED, DRONE_ALT = 6.0, 1.3, 9.0
TEAM = [(214, 120, 42), (72, 73, 227), (60, 170, 60), (200, 60, 200)]  # BGR per rover


def rover_xml(k: int) -> tuple[str, str]:
    pt = (-0.02571, -0.0010148, BASE_Z + 0.097648)
    parts = (
        ('base_link', (0, 0, BASE_Z), (0, 0, 0), '.86 .86 .9 1'),
        ('base_lidar_link', (0.0316958, -0.0010148, BASE_Z + 0.097648), (0, 0, 90), '.1 .1 .1 1'),
        ('pt_base_link', pt, (0, 0, 0), '.12 .12 .12 1'),
        ('pt_link1', (pt[0] - 0.010151, pt[1], pt[2] + 0.0383), (0, 0, 0), '.16 .16 .16 1'),
        ('pt_link2', (pt[0] - 0.010151, pt[1] + 0.02, pt[2] + 0.0383 + 0.043003), (90, 0, 0), '.16 .16 .16 1'),
        ('left_up_wheel_link', (0.0494657, 0.03801, BASE_Z + 0.00675), (0, 0, 0), '.05 .05 .05 1'),
        ('left_down_wheel_link', (-0.0475324, 0.03801, BASE_Z + 0.00675), (0, 0, 0), '.05 .05 .05 1'),
        ('right_up_wheel_link', (0.0494657, -0.03801, BASE_Z + 0.00675), (0, 0, 0), '.05 .05 .05 1'),
        ('right_down_wheel_link', (-0.0475324, -0.03801, BASE_Z + 0.00675), (0, 0, 0), '.05 .05 .05 1'),
    )
    assets, geoms = [], []
    for mesh, pos, euler, rgba in parts:
        if k == 0:
            assets.append(f'<mesh name="rv_{mesh}" file="{os.path.join(ROVER_MESHES, mesh + ".stl")}" '
                          f'scale="{VIS} {VIS} {VIS}"/>')
        p = [v * VIS for v in pos]
        geoms.append(f'<geom type="mesh" mesh="rv_{mesh}" pos="{p[0]} {p[1]} {p[2]}" '
                     f'euler="{euler[0]} {euler[1]} {euler[2]}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    b, g, r = (c / 255 for c in TEAM[k % len(TEAM)])
    geoms.append(f'<geom type="sphere" size="{0.03 * VIS}" pos="0 0 {0.26 * VIS}" rgba="{r} {g} {b} 1" '
                 'contype="0" conaffinity="0"/>')
    body = f'<body name="rover{k}" mocap="true" pos="0 0 0">{"".join(geoms)}</body>'
    return ''.join(assets), body


def build_model(terrain: Terrain, n_rovers: int, n_drones: int):
    import mujoco

    t_assets, t_geoms = mjcf_assets(terrain.dir, terrain.man)
    assets, bodies = [t_assets], []
    for k in range(n_rovers):
        a, b = rover_xml(k)
        assets.append(a)
        bodies.append(b)
    assets.append(f'<texture name="x2" type="2d" file="{DRONE_DIR}/X2_lowpoly_texture_SpinningProps_1024.png"/>'
                  '<material name="x2" texture="x2"/>'
                  f'<mesh name="x2" file="{DRONE_DIR}/X2_lowpoly.obj" scale="{0.01 * VIS} {0.01 * VIS} {0.01 * VIS}"/>')
    for k in range(n_drones):
        bodies.append(f'<body name="drone{k}" mocap="true" pos="0 0 10">'
                      '<geom type="mesh" mesh="x2" material="x2" quat="0 0 1 1" contype="0" conaffinity="0"/></body>')
    bodies.append('<body name="victim" mocap="true" pos="0 0 0">'
                  '<geom type="capsule" fromto="-.35 0 .14 .25 0 .14" size=".14" rgba=".95 .45 .1 1" contype="0" conaffinity="0"/>'
                  '<geom type="sphere" pos=".5 0 .13" size=".12" rgba=".92 .76 .62 1" contype="0" conaffinity="0"/>'
                  '<geom type="capsule" fromto="-.4 .07 .09 -1.1 .13 .09" size=".085" rgba=".15 .2 .45 1" contype="0" conaffinity="0"/>'
                  '<geom type="capsule" fromto="-.4 -.07 .09 -1.1 -.2 .09" size=".085" rgba=".15 .2 .45 1" contype="0" conaffinity="0"/>'
                  '</body>')
    xml = f"""
<mujoco>
  <statistic extent="26" center="0 0 4"/>
  <visual><global offwidth="3840" offheight="2160"/><quality shadowsize="16384" offsamples="8"/>
    <headlight ambient=".42 .42 .45" diffuse=".25 .25 .25" specular="0 0 0"/>
    <map znear="0.002" zfar="12"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".50 .72 .93" rgb2=".93 .96 .99" width="64" height="512"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 40" dir="-.45 .3 -1" diffuse=".85 .8 .7" castshadow="true"/>
    {t_geoms}
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def yaw_quat(yaw: float):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


class Demo:
    def __init__(self, args) -> None:
        self.args = args
        self.T = T = Terrain(args.terrain)
        self.rng = np.random.default_rng(args.seed)
        self.W, self.H = T.shape[1] * T.res, T.shape[0] * T.res
        self.k = int(round(T.res / 0.05))  # truth cells -> fusion level (0.1 m = level 3)
        self.level = 4 - int(round(math.log2(self.k)))
        canopy_all = (T.top_filled - T.ground_filled) > 1.0
        self.truth = T.blocked.astype(np.int8)
        self.drone_truth = np.where(canopy_all & np.isfinite(T.ground), -1, self.truth).astype(np.int8)
        self.canopy_all = canopy_all
        self.carte = CarteFusion(niveau_plan=3, rayon_robot=0.16, marge_securite=0.06)
        starts = T.safe_points(self.rng, args.rovers, 4.0)
        d = T.path_distances(*starts[0])
        far = np.isfinite(d) & (d > 0.6 * d[np.isfinite(d)].max())
        mask = np.kron(far, np.ones((4, 4), bool))[: T.shape[0], : T.shape[1]]
        full = np.zeros(T.shape, bool)
        full[: mask.shape[0], : mask.shape[1]] = mask
        self.victim = T.safe_points(self.rng, 1, 1.0, avoid=starts, avoid_gap=12.0, where=full)[0]
        self.rovers = [{'x': x, 'y': y, 'yaw': self.rng.uniform(-3, 3), 'path': [], 'goal': None, 'trail': [(x, y)],
                        'seen': np.zeros(T.shape, bool), 'replan': 0.0, 'dist': 0.0, 'state': 'search'}
                       for x, y in starts]
        lanes = np.arange(T.y0 + 5, T.y1 - 4, 8.0)
        self.drones = []
        for k in range(args.drones):
            mine = lanes[k :: args.drones]
            wps = []
            for n, y in enumerate(mine):
                xs = (T.x0 + 5, T.x1 - 5) if n % 2 == 0 else (T.x1 - 5, T.x0 + 5)
                wps += [(xs[0], y), (xs[1], y)]
            self.drones.append({'x': wps[0][0], 'y': wps[0][1], 'wps': wps, 'i': 1, 'seen': np.zeros(T.shape, bool),
                                'state': 'survey'})
        self.t, self.found, self.done_t, self.events = 0.0, None, None, []
        self.log(f'Mission start: {args.rovers} rover(s), {args.drones} drone(s). Victim position unknown.')
        ii, jj = np.mgrid[0 : T.shape[0], 0 : T.shape[1]]
        self.cx, self.cy = T.x0 + (jj + 0.5) * T.res, T.y0 + (ii + 0.5) * T.res

    def log(self, text: str) -> None:
        self.events.append((self.t, text))
        print(f'[{self.t:6.1f} s] {text}', flush=True)

    def m(self, x, y):  # world -> map frame (origin at the terrain's south-west corner)
        return (x - self.T.x0, y - self.T.y0)

    def w(self, p):
        return (p[0] + self.T.x0, p[1] + self.T.y0)

    # -- sensors (simulated) ------------------------------------------------------------------
    def rover_look(self, r) -> np.ndarray:
        T = self.T
        ang = r['yaw'] + np.linspace(-ROVER_FOV / 2, ROVER_FOV / 2, 90)
        dist = np.arange(0.1, ROVER_RANGE, T.res * 0.7)
        px = r['x'] + np.cos(ang)[:, None] * dist[None]
        py = r['y'] + np.sin(ang)[:, None] * dist[None]
        i, j = T.cell(px, py)
        hit = T.blocked[i, j]
        before = np.cumsum(hit, axis=1) - hit == 0  # up to and including the first obstacle
        now = np.zeros(T.shape, bool)
        now[i[before], j[before]] = True
        r['seen'] |= now
        return now

    def publish(self) -> None:
        for k, d in enumerate(self.drones):
            m = reduire(np.where(d['seen'], self.drone_truth, -1).astype(np.int8), 2)
            for msg in decouper(m, 0.0, 0.0, self.level - 1, self.carte, f'drone:d{k + 1}', self.t):
                self.carte.recevoir(msg)
        for k, r in enumerate(self.rovers):
            m = np.where(r['seen'], self.truth, -1).astype(np.int8)
            for msg in decouper(m, 0.0, 0.0, self.level, self.carte, f'rover:r{k + 1}', self.t):
                self.carte.recevoir(msg)

    # -- behaviour ----------------------------------------------------------------------------
    def explore_goal(self, r):
        seen = np.zeros(self.T.shape, bool)
        for o in self.rovers:
            seen |= o['seen']
        todo = self.T.safe & ~seen
        cands = self.T.safe_points(self.rng, 25, 2.0, where=todo & self.canopy_all) or \
            self.T.safe_points(self.rng, 25, 2.0, where=todo)
        taken = [o['goal'] for o in self.rovers if o is not r and o['goal']]
        best = None
        for c in cands:
            cost = math.hypot(c[0] - r['x'], c[1] - r['y']) - 0.6 * min(
                [math.hypot(c[0] - g[0], c[1] - g[1]) for g in taken] or [0])
            if best is None or cost < best[0]:
                best = (cost, c)
        return best[1] if best else None

    def plan(self, r, k) -> None:
        if r['goal'] is None:
            return
        dec = self.carte.decider(self.m(r['x'], r['y']), self.m(*r['goal']), zone=(0, 0, self.W, self.H),
                                 tolerance=0.6, drone_disponible=False)
        path = [self.w(p) for p in dec.get('chemin') or []]
        r['path'], r['replan'], r['action'] = path[1:], self.t + 4.0, dec['action']
        if dec['action'] not in ('SUIVRE',) and r['state'] == 'search':
            r['goal'] = None  # unreachable for now: pick another place to look

    def step(self, dt: float) -> None:
        T = self.T
        self.t += dt
        for d in self.drones:
            tx, ty = (self.victim if d['state'] == 'overwatch' else d['wps'][d['i']])
            dx, dy = tx - d['x'], ty - d['y']
            dist = math.hypot(dx, dy)
            if dist < DRONE_SPEED * dt:
                d['x'], d['y'] = tx, ty
                if d['state'] == 'survey':
                    d['i'] = (d['i'] + 1) % len(d['wps'])
            else:
                d['x'] += dx / dist * DRONE_SPEED * dt
                d['y'] += dy / dist * DRONE_SPEED * dt
            d['seen'] |= np.hypot(self.cx - d['x'], self.cy - d['y']) < DRONE_RADIUS
        looks = [self.rover_look(r) for r in self.rovers]
        self.publish()
        if self.found is None:
            vi, vj = T.cell(*self.victim)
            for k, d in enumerate(self.drones):
                if math.hypot(d['x'] - self.victim[0], d['y'] - self.victim[1]) < 4.5 and not self.canopy_all[vi, vj]:
                    self.found = f'drone D{k + 1} (lidar + camera, open ground)'
                    d['state'] = 'overwatch'
            for k, now in enumerate(looks):
                if now[max(vi - 3, 0) : vi + 4, max(vj - 3, 0) : vj + 4].any():
                    self.found = f'rover R{k + 1} (depth camera, under the trees)'
            if self.found:
                self.log(f'VICTIM FOUND by {self.found}')
                best = min(range(len(self.rovers)), key=lambda q: math.hypot(
                    self.rovers[q]['x'] - self.victim[0], self.rovers[q]['y'] - self.victim[1]))
                for q, r in enumerate(self.rovers):
                    r['state'] = 'rescue' if q == best else 'hold'
                    r['goal'] = self.victim if q == best else None
                    r['path'], r['replan'] = [], 0.0
                self.log(f'Rover R{best + 1} is sent to the victim on the merged map')
                for d in self.drones:
                    d['state'] = 'overwatch'
        for k, r in enumerate(self.rovers):
            if r['state'] == 'hold' or r['state'] == 'arrived':
                continue
            if r['state'] == 'rescue' and math.hypot(r['x'] - self.victim[0], r['y'] - self.victim[1]) < 0.9:
                r['state'], r['path'] = 'arrived', []
                self.done_t = self.t
                self.log(f'Rover R{k + 1} REACHED THE VICTIM after {r["dist"]:.0f} m')
                continue
            if r['state'] == 'search' and (r['goal'] is None or r['seen'][T.cell(*r['goal'])] or not r['path']):
                if r['goal'] is None or r['seen'][T.cell(*r['goal'])]:
                    r['goal'] = self.explore_goal(r)
                    r['replan'] = 0.0
            ahead = [self.m(r['x'], r['y'])] + [self.m(*p) for p in r['path']]
            if self.t >= r['replan'] or not r['path'] or not self.carte.chemin_valide(ahead, horizon=2.5):
                self.plan(r, k)
            move = ROVER_SPEED * dt
            while move > 1e-6 and r['path']:
                tx, ty = r['path'][0]
                dx, dy = tx - r['x'], ty - r['y']
                dist = math.hypot(dx, dy)
                want = math.atan2(dy, dx)
                turn = (want - r['yaw'] + math.pi) % (2 * math.pi) - math.pi
                r['yaw'] += max(-2.0 * dt, min(2.0 * dt, turn))
                if abs(turn) > 0.9:
                    break  # turn on the spot first
                s = min(move, dist)
                nx, ny = r['x'] + dx / max(dist, 1e-6) * s, r['y'] + dy / max(dist, 1e-6) * s
                if T.is_blocked(nx, ny):
                    r['path'], r['replan'] = [], 0.0  # would touch something: stop and replan
                    break
                r['x'], r['y'], r['dist'] = nx, ny, r['dist'] + s
                move -= s
                if s >= dist - 1e-6:
                    r['path'].pop(0)
            if math.hypot(r['x'] - r['trail'][-1][0], r['y'] - r['trail'][-1][1]) > 0.15:
                r['trail'].append((r['x'], r['y']))

    # -- pictures -----------------------------------------------------------------------------
    def px(self, x, y, size):
        return int((x - self.T.x0) / self.W * size), int(size - 1 - (y - self.T.y0) / self.H * size)

    def map_picture(self, kind: str, size: int) -> np.ndarray:
        if kind == 'fused':
            g = self.carte.grille(0, 0, self.W, self.H)
            cls = ff.fused_classes(g.val, g.src)
        elif kind == 'drone':
            cls = ff.drone_classes(self.carte.couche('drone', 0, 0, self.W, self.H))
        else:
            v = self.carte.couche('rover', 0, 0, self.W, self.H)
            cls = np.full(v.shape, ff.K_UNKNOWN, np.uint8)
            cls[v == 0] = ff.K_ROVER_FREE
            cls[v == 1] = ff.K_ROVER_OBST
        img = cv2.resize(ff.PALETTE[cls][::-1], (size, size), interpolation=cv2.INTER_AREA)
        if kind != 'drone':
            for k, r in enumerate(self.rovers):
                pts = np.array([self.px(x, y, size) for x, y in r['trail']], np.int32)
                cv2.polylines(img, [pts], False, TEAM[k % 4], 1, cv2.LINE_AA)
                if kind == 'fused' and r['path']:
                    pts = np.array([self.px(r['x'], r['y'], size)] + [self.px(x, y, size) for x, y in r['path']], np.int32)
                    cv2.polylines(img, [pts], False, ff.PATH_C, 2, cv2.LINE_AA)
        if kind != 'rover':
            for d in self.drones:
                cv2.circle(img, self.px(d['x'], d['y'], size), int(DRONE_RADIUS / self.W * size), ff.WINDOW_C, 1, cv2.LINE_AA)
                cv2.drawMarker(img, self.px(d['x'], d['y'], size), ff.WINDOW_C, cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
        if kind != 'drone':
            for k, r in enumerate(self.rovers):
                p = self.px(r['x'], r['y'], size)
                cv2.circle(img, p, 5, TEAM[k % 4], -1, cv2.LINE_AA)
                cv2.circle(img, p, 5, ff.INK_C, 1, cv2.LINE_AA)
                cv2.line(img, p, (int(p[0] + 11 * math.cos(r['yaw'])), int(p[1] - 11 * math.sin(r['yaw']))), ff.INK_C, 2, cv2.LINE_AA)
        if self.found and kind == 'fused':
            ff.draw_star(img, _V(self, size), 0, 0) if False else cv2.drawMarker(
                img, self.px(*self.victim, size), ff.GOAL_C, cv2.MARKER_STAR, 22, 3, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), ff.INK_C, 1)
        return img


def _V(*_):
    return None


def text(img, s, org, scale=0.6, colour=ff.INK_C, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def main() -> None:
    import imageio_ffmpeg
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('terrain')
    ap.add_argument('out')
    ap.add_argument('--rovers', type=int, default=2)
    ap.add_argument('--drones', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--speed', type=float, default=5.0, help='simulated seconds per second of video')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--max-time', type=float, default=420.0, help='simulated seconds')
    args = ap.parse_args()

    demo = Demo(args)
    T = demo.T
    model, data = build_model(T, args.rovers, args.drones)
    ren = mujoco.Renderer(model, 2160, 3840)  # 3x supersampling, reduced below: smooth edges
    cam = mujoco.MjvCamera()
    mid = {n: model.body(n).mocapid[0] for n in [f'rover{k}' for k in range(args.rovers)] +
           [f'drone{k}' for k in range(args.drones)] + ['victim']}
    vz = float(T.ground_at(*demo.victim))
    data.mocap_pos[mid['victim']] = [*demo.victim, vz]
    data.mocap_quat[mid['victim']] = yaw_quat(1.0)
    writer = imageio_ffmpeg.write_frames(args.out, (1920, 1080), fps=args.fps, pix_fmt_in='bgr24', quality=9,
                                         macro_block_size=8)
    writer.send(None)
    dt, frame, hold = args.speed / args.fps, 0, 0
    look = np.array([0.0, 0.0, float(np.nanmean(T.ground))])
    while demo.t < args.max_time:
        demo.step(dt)
        for k, r in enumerate(demo.rovers):
            data.mocap_pos[mid[f'rover{k}']] = [r['x'], r['y'], float(T.ground_at(r['x'], r['y']))]
            data.mocap_quat[mid[f'rover{k}']] = yaw_quat(r['yaw'])
        for k, d in enumerate(demo.drones):
            z = float(T.ground_at(d['x'], d['y'])) + DRONE_ALT + 0.15 * math.sin(demo.t * 0.8 + k)
            data.mocap_pos[mid[f'drone{k}']] = [d['x'], d['y'], z]
        mujoco.mj_forward(model, data)
        # Camera: wide orbit, then chase views of the rovers; on the victim at the end.
        vt = frame / args.fps
        shot = int(vt // 9) % 3
        if demo.done_t is not None:
            target, dist, el = np.array([*demo.victim, vz]), 7.0, -24.0
        elif shot == 0:
            target, dist, el = np.array([0.0, 0.0, look[2]]), 46.0, -38.0
        else:
            r = demo.rovers[(shot - 1) % len(demo.rovers)] if demo.found is None else next(
                (q for q in demo.rovers if q['state'] == 'rescue'), demo.rovers[0])
            target, dist, el = np.array([r['x'], r['y'], float(T.ground_at(r['x'], r['y'])) + 0.4]), 6.5, -22.0
        look = look + 0.12 * (target - look) if np.linalg.norm(target - look) < 8 else target
        cam.lookat[:] = look
        cam.distance, cam.elevation, cam.azimuth = dist, el, 35 + vt * 4.0
        ren.update_scene(data, cam)
        scn = ren.scene
        try:  # overlays: planned paths, the drones' lidar cones, the victim's beacon
            def line(a, b, rgba, width=0.03):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array(a, float), np.array(b, float))
                scn.ngeom += 1

            for k, r in enumerate(demo.rovers):
                pts = [(r['x'], r['y'])] + r['path'][:: 3]
                b_, g_, r_ = (c / 255 for c in TEAM[k % 4])
                for a, b in zip(pts[:-1], pts[1:]):
                    line([*a, float(T.ground_at(*a)) + 0.12], [*b, float(T.ground_at(*b)) + 0.12], [r_, g_, b_, 0.9], 0.035)
            for k, d in enumerate(demo.drones):
                top = data.mocap_pos[mid[f'drone{k}']]
                for a in np.linspace(0, 2 * math.pi, 18, endpoint=False) + demo.t * 0.9:
                    gx, gy = d['x'] + DRONE_RADIUS * math.cos(a), d['y'] + DRONE_RADIUS * math.sin(a)
                    line(top, [gx, gy, float(T.top_at(gx, gy))], [0.1, 0.75, 0.95, 0.35], 0.012)
            if demo.found:
                line([*demo.victim, vz], [*demo.victim, vz + 9], [0.9, 0.2, 0.2, 0.45], 0.12)
        except Exception as e:  # noqa: BLE001 - overlays are cosmetic
            if frame == 0:
                print('overlay skipped:', e)
        view = cv2.resize(cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR), (1280, 720), interpolation=cv2.INTER_AREA)

        img = np.full((1080, 1920, 3), ff.FREE_C, np.uint8)
        text(img, 'POLARIS', (24, 46), 1.1, ff.INK_C, 3)
        text(img, 'forest search & rescue  |  drones + rovers, one merged map  |  simulation', (190, 44), 0.72, (90, 90, 90), 1)
        img[66:786, 20:1300] = view
        cv2.rectangle(img, (20, 66), (1300, 786), ff.INK_C, 1)
        fused = demo.map_picture('fused', 580)
        img[66:646, 1320:1900] = fused
        text(img, 'MERGED MAP (merging_map.CarteFusion)', (1320, 56), 0.6, ff.INK_C, 2)
        small = 285
        img[700 : 700 + small, 1320 : 1320 + small] = demo.map_picture('drone', small)
        img[700 : 700 + small, 1615 : 1615 + small] = demo.map_picture('rover', small)
        text(img, 'DRONE LAYER (lidar)', (1320, 690), 0.55, ff.INK_C, 1)
        text(img, 'ROVER LAYER (depth, 5 m)', (1615, 690), 0.55, ff.INK_C, 1)
        y = 1010
        for name, c in (('unknown', ff.UNKNOWN_C), ('free', ff.FREE_C), ('free, seen by a rover', ff.ROVER_FREE_C),
                        ('obstacle (drone)', ff.DRONE_OBST_C), ('obstacle (rover)', ff.ROVER_OBST_C), ('path', ff.PATH_C)):
            x = 1320 + [0, 140, 0, 290, 290, 480][['unknown', 'free', 'free, seen by a rover', 'obstacle (drone)',
                                                 'obstacle (rover)', 'path'].index(name)]
            yy = y + (28 if name in ('free, seen by a rover', 'obstacle (rover)', 'path') else 0)
            cv2.rectangle(img, (x, yy - 12), (x + 16, yy + 4), c, -1)
            cv2.rectangle(img, (x, yy - 12), (x + 16, yy + 4), ff.INK_C, 1)
            text(img, name, (x + 24, yy + 2), 0.48)
        state = 'SEARCHING: victim position unknown' if demo.found is None else (
            'VICTIM REACHED' if demo.done_t is not None else 'VICTIM FOUND: rover on its way')
        colour = ff.PATH_C if demo.found is None else ff.GOAL_C
        text(img, state, (24, 830), 0.95, colour, 2)
        text(img, f'mission time {int(demo.t // 60):02d}:{int(demo.t % 60):02d}   (video x{args.speed:g})', (24, 866), 0.62)
        for k, r in enumerate(demo.rovers):
            cv2.circle(img, (36, 902 + 30 * k), 8, TEAM[k % 4], -1, cv2.LINE_AA)
            text(img, f'R{k + 1}  {r["state"]:8s} {r["dist"]:5.1f} m driven', (54, 908 + 30 * k), 0.6)
        known = (demo.carte.grille(0, 0, demo.W, demo.H).val != -1).mean() if frame % 10 == 0 else getattr(demo, '_known', 0)
        demo._known = known
        text(img, f'map known: {100 * known:.0f} %', (24, 908 + 30 * len(demo.rovers)), 0.6)
        for n, (t_, s) in enumerate(demo.events[-5:]):
            text(img, f'{int(t_ // 60):02d}:{int(t_ % 60):02d}  {s}', (470, 866 + 30 * n), 0.58,
                 ff.GOAL_C if 'VICTIM' in s else ff.INK_C, 1)
        text(img, 'Map fusion and path planning: the real code. Sensors and motion: simulated.', (24, 1064), 0.5, (110, 110, 110))
        writer.send(np.ascontiguousarray(img))
        frame += 1
        if frame == 40:
            cv2.imwrite(os.path.splitext(args.out)[0] + '_first.jpg', img)
        if demo.done_t is not None:
            hold += 1
            if hold > args.fps * 5:
                break
    cv2.imwrite(os.path.splitext(args.out)[0] + '_last.jpg', img)
    writer.close()
    print(f'{args.out}: {frame} frames, {frame / args.fps:.0f} s of video, mission {demo.t:.0f} s')


if __name__ == '__main__':
    main()
