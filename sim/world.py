"""Shared simulation world: the RaspRover and a drone in a furnished scene (MuJoCo).

For the whole team: the drone side (overhead camera), the drone/rover
coordination, and the rover side (camera for the 3D mapper). Same scene as the
Blender renders (exported from Blender; the scene build scripts are not in this repository,
the result comes from `python sim/fetch_scene.py`).

    from world import RoverWorld
    w = RoverWorld("sim/scenes/apartment")
    w.step(v=0.2, omega=0.0)                 # rover command: m/s, rad/s (dt = 0.1 s)
    rgb, depth, K, T = w.rover_camera()      # what the rover sees, exact depth, OpenCV pose
    img, K, T = w.drone_camera()             # what the drone sees (downward)
    w.set_drone(x, y, z, yaw)
    x, y, yaw = w.rover_pose                 # ground truth
    occ = w.grid("overhead") / w.grid("rover")   # true occupancy maps (see grid())
    d = w.d405_camera()                      # simulated RealSense D405 (dict, see below)

Simulated RealSense D405 ("profondeur courte portée idéale, extrinsèques
exactes"): a second camera rigidly mounted on the rover, by default at the
front of the chassis (D405_MOUNT: 9 cm ahead of the body centre, 6 cm above
the floor, looking forward). THE MOUNT POINT IS AN ASSUMPTION, to be confirmed
on the real robot (constructor argument d405_mount). 848x480, fovy 58° (MuJoCo
has square pixels: 58° vertical gives 88.8° horizontal; Intel: 87°±3 x 58°±1).
Depth is aligned to its own colour image by construction (same camera).
Depth model (our assumption, not Intel's spec): valid from 0.07 m to
`d405_max_range` (1.0 m), zero elsewhere; stereo-like noise sigma(z) =
z^2 * 0.08 px / (f_px * 0.018 m baseline) (~2.6 mm at 0.5 m, ~1 cm at 1 m),
spatially correlated over ~4 px; pixels on depth discontinuities and 1 % of
random pixels are dropped (0), like stereo holes.

Rover motion: planar differential drive. Collisions and ranges come from the
exact occupancy of the scene at rover height (2-30 cm): the rover fits under a
table but stops at its legs, which MuJoCo's convex-hull mesh contacts cannot do.
"""

from __future__ import annotations

import json
import math
import os

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESHES = os.path.join(HERE, 'assets', 'rasp_rover', 'meshes')
BASE_Z = 0.04 - 0.00675
CAM_IN_BASE = (0.011825, -0.001, 0.178382)
ROVER_HALF = (0.093, 0.068)  # body footprint half sizes (m)
TRACK = 0.13
TEXSCALE = {'floor': 0.5, 'wall': 0.6, 'ceiling': 0.5}  # same tiling as the Blender materials
# Simulated RealSense D405, mount in the rover body frame (x fwd, y left, z up
# from the floor under the body centre). ASSUMED position: confirm on the robot.
D405_MOUNT = (0.09, 0.0, 0.06)
D405_SIZE = (848, 480)
D405_FOVY = 58.0
D405_RANGE = (0.07, 1.0)  # metres, valid depth (outside: 0 = no measurement)
D405_BASELINE = 0.018  # metres (Intel D405 stereo baseline)
D405_SUBPIXEL = 0.08  # disparity noise, pixels RMS (assumption)


def _png(path: str) -> str:
    """MuJoCo reads PNG textures only: convert (once) anything else."""
    if path.lower().endswith('.png'):
        return path
    out = os.path.splitext(path)[0] + '.png'
    if not os.path.exists(out):
        cv2.imwrite(out, cv2.imread(path))
    return out


def _xml(
    scene_dir: str,
    man: dict,
    rover_fovy: float,
    drone_fovy: float,
    d405_mount=D405_MOUNT,
    d405_fovy: float = D405_FOVY,
) -> str:
    assets, geoms = [], []
    for i, it in enumerate(man['items']):
        mesh = f'm{i}'
        assets.append(
            f'<mesh name="{mesh}" file="{os.path.join(scene_dir, "mujoco", it["obj"])}"/>'
        )
        mat = f'mat{i}'
        rep = TEXSCALE.get(it['kind'], 1.0)
        if it['texture']:
            tex = _png(os.path.join(scene_dir, 'mujoco', it['texture']))
            assets.append(f'<texture name="t{i}" type="2d" file="{tex}"/>')
            assets.append(
                f'<material name="{mat}" texture="t{i}" texrepeat="{rep} {rep}" specular="0.15" shininess="0.2"/>'
            )
        else:
            r, g, b, a = it['rgba']
            em = ' emission="1"' if it['emissive'] else ''
            assets.append(f'<material name="{mat}" rgba="{r} {g} {b} {a}"{em} specular="0.1"/>')
        group = 4 if it['kind'] == 'ceiling' else 2
        geoms.append(
            f'<geom name="{it["name"]}" type="mesh" mesh="{mesh}" material="{mat}" '
            f'contype="0" conaffinity="0" group="{group}"/>'
        )
    rover_parts = []
    pt = (-0.02571, -0.0010148, BASE_Z + 0.097648)
    for mesh, pos, euler, rgba in (
        ('base_link', (0, 0, BASE_Z), (0, 0, 0), '.82 .82 .85 1'),
        ('base_lidar_link', (0.0316958, -0.0010148, BASE_Z + 0.097648), (0, 0, 90), '.1 .1 .1 1'),
        ('pt_base_link', pt, (0, 0, 0), '.1 .1 .1 1'),
        ('pt_link1', (pt[0] - 0.010151, pt[1], pt[2] + 0.0383), (0, 0, 0), '.15 .15 .15 1'),
        (
            'pt_link2',
            (pt[0] - 0.010151, pt[1] + 0.02, pt[2] + 0.0383 + 0.043003),
            (90, 0, 0),
            '.15 .15 .15 1',
        ),
        ('left_up_wheel_link', (0.0494657, 0.03801, BASE_Z + 0.00675), (0, 0, 0), '.05 .05 .05 1'),
        (
            'left_down_wheel_link',
            (-0.0475324, 0.03801, BASE_Z + 0.00675),
            (0, 0, 0),
            '.05 .05 .05 1',
        ),
        (
            'right_up_wheel_link',
            (0.0494657, -0.03801, BASE_Z + 0.00675),
            (0, 0, 0),
            '.05 .05 .05 1',
        ),
        (
            'right_down_wheel_link',
            (-0.0475324, -0.03801, BASE_Z + 0.00675),
            (0, 0, 0),
            '.05 .05 .05 1',
        ),
    ):
        assets.append(f'<mesh name="rv_{mesh}" file="{os.path.join(MESHES, mesh + ".stl")}"/>')
        rover_parts.append(
            f'<geom type="mesh" mesh="rv_{mesh}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'euler="{euler[0]} {euler[1]} {euler[2]}" rgba="{rgba}" contype="0" conaffinity="0" group="1"/>'
        )
    lights = ''.join(
        f'<light pos="{x} {y} {h - 0.1}" dir="0 0 -1" diffuse=".45 .43 .4" castshadow="{str(i == 0).lower()}" '
        f'attenuation="1 0 0" cutoff="80"/>'
        for i, (x, y, h) in enumerate(
            man.get('lights', [(-2.2, -0.5, 2.6), (0.9, -0.3, 2.6), (2.6, 1.3, 2.6)])
        )
    )
    return f"""
<mujoco model="rover_world">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient=".42 .41 .4" diffuse=".25 .25 .25" specular="0 0 0"/>
    <map znear="0.005" zfar="80"/>
  </visual>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {lights}
    {''.join(geoms)}
    <body name="rover" mocap="true" pos="0 0 0">
      {''.join(rover_parts)}
      <camera name="rover_cam" pos="{CAM_IN_BASE[0]} {CAM_IN_BASE[1]} {BASE_Z + CAM_IN_BASE[2]}" euler="90 -90 0" fovy="{rover_fovy}"/>
      <camera name="chase" pos="-0.9 0 0.55" xyaxes="0 -1 0 0.5 0 0.866" fovy="60"/>
      <camera name="d405_cam" pos="{d405_mount[0]} {d405_mount[1]} {d405_mount[2]}" euler="90 -90 0" fovy="{d405_fovy}"/>
    </body>
    <body name="drone" mocap="true" pos="0 0 2.2">
      <geom type="box" size=".09 .09 .025" rgba=".15 .15 .17 1" contype="0" conaffinity="0" group="1"/>
      <geom type="cylinder" size=".07 .005" pos=".13 .13 .03" rgba=".6 .6 .65 .6" contype="0" conaffinity="0" group="1"/>
      <geom type="cylinder" size=".07 .005" pos="-.13 .13 .03" rgba=".6 .6 .65 .6" contype="0" conaffinity="0" group="1"/>
      <geom type="cylinder" size=".07 .005" pos=".13 -.13 .03" rgba=".6 .6 .65 .6" contype="0" conaffinity="0" group="1"/>
      <geom type="cylinder" size=".07 .005" pos="-.13 -.13 .03" rgba=".6 .6 .65 .6" contype="0" conaffinity="0" group="1"/>
      <camera name="drone_cam" pos="0 0 -0.04" euler="0 0 -90" fovy="{drone_fovy}"/>
    </body>
  </worldbody>
</mujoco>"""


class RoverWorld:
    def __init__(
        self,
        scene_dir: str,
        rover_size=(640, 480),
        rover_fovy=50.0,
        drone_size=(960, 720),
        drone_fovy=90.0,
        start=None,
        d405_mount=D405_MOUNT,
        d405_size=D405_SIZE,
        d405_fovy=D405_FOVY,
        d405_max_range=D405_RANGE[1],
        d405_seed=0,
    ):
        self.scene_dir = scene_dir
        man = json.load(open(os.path.join(scene_dir, 'mujoco', 'manifest.json')))
        self.d405_mount = tuple(float(v) for v in d405_mount)
        self.d405_size, self.d405_fovy = tuple(d405_size), float(d405_fovy)
        self.d405_range = (D405_RANGE[0], float(d405_max_range))
        self._d405_rng = np.random.default_rng(d405_seed)
        self._d405_r = self._d405_rd = None  # renderers, created on first use
        self.model = mujoco.MjModel.from_xml_string(
            _xml(scene_dir, man, rover_fovy, drone_fovy, self.d405_mount, self.d405_fovy)
        )
        self.data = mujoco.MjData(self.model)
        self.rover_size, self.drone_size = rover_size, drone_size
        self.rover_fovy, self.drone_fovy = rover_fovy, drone_fovy
        self._rover_mocap = self.model.body('rover').mocapid[0]
        self._drone_mocap = self.model.body('drone').mocapid[0]
        self._r = mujoco.Renderer(self.model, height=rover_size[1], width=rover_size[0])
        self._rd = mujoco.Renderer(self.model, height=rover_size[1], width=rover_size[0])
        self._rd.enable_depth_rendering()
        self._dr = mujoco.Renderer(self.model, height=drone_size[1], width=drone_size[0])
        occ = np.load(os.path.join(scene_dir, 'occupancy.npz'))
        self.res, self.room = float(occ['res']), tuple(float(v) for v in occ['room'])
        self._occ_rover = occ['occ']
        self._occ_overhead = None
        self.t = 0.0
        self.collided = False
        if start is None:
            tp = os.path.join(scene_dir, 'trajectory.json')
            start = (
                json.load(open(tp))['trajectory'][0][1:] if os.path.exists(tp) else (0.0, 0.0, 0.0)
            )
        self.set_rover(*start)
        self.set_drone(0.0, 0.0, 2.2, 0.0)

    # -- rover -------------------------------------------------------------
    @property
    def rover_pose(self) -> tuple[float, float, float]:
        return self._pose

    def set_rover(self, x, y, yaw):
        self._pose = (float(x), float(y), float(yaw))
        self.data.mocap_pos[self._rover_mocap] = (x, y, 0.0)
        self.data.mocap_quat[self._rover_mocap] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)

    def footprint_free(self, x, y, yaw, margin=0.0) -> bool:
        """True if the rover body footprint at (x, y, yaw) touches nothing at its height."""
        hx, hy = ROVER_HALF[0] + margin, ROVER_HALF[1] + margin
        c, s = math.cos(yaw), math.sin(yaw)
        xs, ys = np.meshgrid(
            np.arange(-hx, hx + 1e-9, self.res / 2), np.arange(-hy, hy + 1e-9, self.res / 2)
        )
        px, py = x + c * xs - s * ys, y + s * xs + c * ys
        i = ((px + self.room[0] / 2) / self.res).astype(int)
        j = ((py + self.room[1] / 2) / self.res).astype(int)
        nx, ny = self._occ_rover.shape
        if (i < 0).any() or (j < 0).any() or (i >= nx).any() or (j >= ny).any():
            return False
        return not self._occ_rover[i, j].any()

    def step(self, v: float, omega: float, dt: float = 0.1) -> bool:
        """Drive for dt seconds (body-frame speed v m/s, yaw rate omega rad/s).
        Returns False and stays put if the move would hit something."""
        x, y, yaw = self._pose
        n = max(1, int(round(dt / 0.02)))
        for _ in range(n):
            ny_ = yaw + omega * dt / n
            nx_ = x + v * math.cos(ny_) * dt / n
            nyy = y + v * math.sin(ny_) * dt / n
            if not self.footprint_free(nx_, nyy, ny_):
                self.collided = True
                self.set_rover(x, y, yaw)
                self.t += dt
                return False
            x, y, yaw = nx_, nyy, ny_
        self.collided = False
        self.t += dt
        self.set_rover(x, y, yaw)
        return True

    def step_wheels(self, left: float, right: float, dt: float = 0.1) -> bool:
        """Same as the real RaspRover command {"T":1,"L":left,"R":right} (wheel speeds, m/s)."""
        return self.step((left + right) / 2, (right - left) / TRACK, dt)

    def ranges(self, n_bins=24, fov=math.pi, max_range=2.5) -> np.ndarray:
        """Distance to the nearest obstacle at rover height per sector (front fov)."""
        x, y, yaw = self._pose
        out = np.full(n_bins, max_range)
        steps = np.arange(0.0, max_range, self.res / 2)
        for k, a in enumerate(np.linspace(-fov / 2, fov / 2, n_bins)):
            px, py = x + steps * math.cos(yaw + a), y + steps * math.sin(yaw + a)
            i = ((px + self.room[0] / 2) / self.res).astype(int)
            j = ((py + self.room[1] / 2) / self.res).astype(int)
            ok = (
                (i >= 0)
                & (j >= 0)
                & (i < self._occ_rover.shape[0])
                & (j < self._occ_rover.shape[1])
            )
            hit = np.zeros_like(ok)
            hit[ok] = self._occ_rover[i[ok], j[ok]]
            hit |= ~ok
            if hit.any():
                out[k] = steps[int(np.argmax(hit))]
        return out

    # -- cameras -------------------------------------------------------------
    @staticmethod
    def _K(w, h, fovy):
        f = (h / 2) / math.tan(math.radians(fovy) / 2)
        return np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])

    def _cam_pose_cv(self, name) -> np.ndarray:
        cid = self.model.camera(name).id
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.data.cam_xpos[cid]
        return T

    def _opt(self, hide_ceiling: bool):
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        if hide_ceiling:
            opt.geomgroup[4] = 0
        return opt

    def rover_camera(self):
        """(rgb uint8 HxWx3, depth float32 metres along the optical axis, K 3x3,
        camera->world 4x4 with OpenCV axes)."""
        opt = self._opt(False)
        self._r.update_scene(self.data, camera='rover_cam', scene_option=opt)
        rgb = self._r.render()
        self._rd.update_scene(self.data, camera='rover_cam', scene_option=opt)
        depth = self._rd.render()
        return (
            rgb,
            depth,
            self._K(*self.rover_size, self.rover_fovy),
            self._cam_pose_cv('rover_cam'),
        )

    def d405_camera(self, noise: bool = True) -> dict:
        """Simulated RealSense D405 at the same instant as rover_camera().
        Returns {"rgb": uint8 HxWx3, "depth": float32 metres, sensor model
        (0 = no measurement), "depth_true": float32 exact depth (full range),
        "K": 3x3, "T": camera->world 4x4 OpenCV axes}. Depth is pixel-aligned
        with "rgb". See the module docstring for the noise model."""
        w, h = self.d405_size
        if self._d405_r is None:
            self._d405_r = mujoco.Renderer(self.model, height=h, width=w)
            self._d405_rd = mujoco.Renderer(self.model, height=h, width=w)
            self._d405_rd.enable_depth_rendering()
        opt = self._opt(False)
        # The scene-wide near plane (~6 cm) would clip what is right in front
        # of a 7 cm sensor: use 1 cm for this camera only.
        znear = self.model.vis.map.znear
        self.model.vis.map.znear = 0.01 / self.model.stat.extent
        try:
            self._d405_r.update_scene(self.data, camera='d405_cam', scene_option=opt)
            rgb = self._d405_r.render()
            self._d405_rd.update_scene(self.data, camera='d405_cam', scene_option=opt)
            true = self._d405_rd.render().astype(np.float32)
        finally:
            self.model.vis.map.znear = znear
        K = self._K(w, h, self.d405_fovy)
        depth = (
            self.d405_noise(true, K)
            if noise
            else np.where(
                (true >= self.d405_range[0]) & (true <= self.d405_range[1]), true, 0
            ).astype(np.float32)
        )
        return {
            'rgb': rgb,
            'depth': depth,
            'depth_true': true,
            'K': K,
            'T': self._cam_pose_cv('d405_cam'),
        }

    def d405_noise(self, true: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Exact depth -> simulated D405 measurement (our model, not Intel's)."""
        rng = self._d405_rng
        h, w = true.shape
        # Stereo depth noise grows with z^2; correlated over the matching window.
        field = cv2.resize(
            rng.standard_normal((h // 4 + 1, w // 4 + 1)).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        field /= field.std() + 1e-9
        sigma = true**2 * D405_SUBPIXEL / (K[0, 0] * D405_BASELINE)
        depth = true + sigma * field
        lo, hi = self.d405_range
        ok = (true >= lo) & (true <= hi)
        # Holes at depth discontinuities (occlusion shadows / flying pixels).
        gx = np.abs(cv2.Sobel(true, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(true, cv2.CV_32F, 0, 1, ksize=3))
        edge = (gx + gy) / 8.0 > 0.05 * np.maximum(true, 1e-3)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ok &= ~edge & (rng.random((h, w)) > 0.01)
        return np.where(ok, depth, 0.0).astype(np.float32)

    def chase_camera(self):
        self._r.update_scene(self.data, camera='chase', scene_option=self._opt(False))
        return self._r.render()

    def set_drone(self, x, y, z, yaw=0.0):
        self.data.mocap_pos[self._drone_mocap] = (x, y, z)
        self.data.mocap_quat[self._drone_mocap] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)

    def drone_camera(self, hide_ceiling=True):
        """Downward view from the drone: (rgb, K, camera->world 4x4 OpenCV axes).
        A world point p projects to pixel K @ (R^T (p - t)) / z."""
        self._dr.update_scene(self.data, camera='drone_cam', scene_option=self._opt(hide_ceiling))
        return (
            self._dr.render(),
            self._K(*self.drone_size, self.drone_fovy),
            self._cam_pose_cv('drone_cam'),
        )

    @staticmethod
    def project(K, T, p):
        """Pixel (u, v) of world point p in a camera (K, camera->world T), or None if behind."""
        c = T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])
        if c[2] <= 0:
            return None
        return float(K[0, 0] * c[0] / c[2] + K[0, 2]), float(K[1, 1] * c[1] / c[2] + K[1, 2])

    # -- ground truth maps ---------------------------------------------------------
    def grid(self, kind='rover'):
        """True occupancy (bool array indexed [i_x, j_y], cell = self.res metres,
        cell (0, 0) at world (-room_w/2, -room_d/2)).
        "rover":    what blocks the rover (furniture between 2 and 30 cm).
        "overhead": everything seen from above (table tops included). The two
                    differ under tables and chairs: the rover can drive there."""
        if kind == 'rover':
            return self._occ_rover
        if self._occ_overhead is None:
            self._occ_overhead = self._overhead_footprint()
        return self._occ_overhead

    def _overhead_footprint(self):
        man = json.load(open(os.path.join(self.scene_dir, 'mujoco', 'manifest.json')))
        nx, ny = self._occ_rover.shape
        img = np.zeros((ny, nx), np.uint8)
        for it in man['items']:
            if it['kind'] != 'furniture':
                continue
            verts, faces = [], []
            for line in open(os.path.join(self.scene_dir, 'mujoco', it['obj'])):
                if line.startswith('v '):
                    verts.append(line.split()[1:4])
                elif line.startswith('f '):
                    idx = [int(a.split('/')[0]) - 1 for a in line.split()[1:]]
                    faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
            if not verts or not faces:
                continue
            V = np.array(verts, float)
            tri = np.array(faces)
            tri = tri[V[tri, 2].max(1) >= 0.02]  # ignore what lies on the floor (rugs)
            uv = np.round(
                np.stack(
                    [
                        (V[:, 0] + self.room[0] / 2) / self.res,
                        (V[:, 1] + self.room[1] / 2) / self.res,
                    ],
                    1,
                )
            ).astype(np.int32)
            P = uv[tri]  # (n, 3, 2) triangle corners in pixels
            # Most triangles fall inside one cell: set those pixels at once, draw the others.
            one = (P == P[:, :1]).all(axis=(1, 2))
            u, v = P[one, 0, 0], P[one, 0, 1]
            ok = (u >= 0) & (u < nx) & (v >= 0) & (v < ny)
            img[v[ok], u[ok]] = 1
            for poly in P[~one]:
                cv2.fillConvexPoly(img, poly, 1)
        return img.T.astype(bool) | self._occ_rover
