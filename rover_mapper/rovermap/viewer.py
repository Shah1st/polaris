"""Web 3D viewer (viser): the map, the rover, and a car-style chase camera."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import viser
import viser.transforms as vtf

from .mapper import Keyframe, Mapper, TrackResult
from .source import MODE_RGB, SENSOR_MODES

# Approximate RaspRover dimensions, metres.
BODY_L, BODY_W, BODY_H = 0.19, 0.11, 0.06
WHEEL_R, WHEEL_W = 0.04, 0.035
TRACK = 0.15          # distance between left and right wheel centres
WHEELBASE = 0.11
CAM_FORWARD = 0.07    # camera lens ahead of the body centre

MODES = ("3e personne", "Vue du dessus", "Vue robot (1re personne)", "Libre")
RENDERS = ("Points", "Splats (gaussiennes)")


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def splats_from_points(points, colors, normals, radii, tangent=0.75, thickness=0.12, opacity=0.9):
    """One flat Gaussian disc per map point, lying on the surface.

    Disc radius ~ the pixel footprint on the surface (`radii`), so neighbouring
    discs overlap into a continuous surface; thin along the normal. No
    training: an instant, smoother-looking alternative to points.
    """
    n = normals.astype(np.float32)
    rt = (tangent * radii).astype(np.float32)[:, None, None]
    rn = (thickness * radii).astype(np.float32)[:, None, None]
    nnT = n[:, :, None] * n[:, None, :]
    cov = rt ** 2 * (np.eye(3, dtype=np.float32)[None] - nnT) + rn ** 2 * nnT
    rgb = colors.astype(np.float32) / 255.0
    op = np.full((len(points), 1), opacity, np.float32)
    return points.astype(np.float32), cov, rgb, op


class Viewer:
    def __init__(self, mapper: Mapper, port: int = 8080, robot_scale: float = 1.0,
                 render: str = "splats", sensor_mode: str = MODE_RGB, d405_available: bool = False,
                 d405_mount: tuple[float, float, float] | None = None):
        self.mapper = mapper
        self._sensor0 = sensor_mode
        self.d405_available = d405_available
        self.d405_mount = d405_mount
        self.sensor_mode_requested: str | None = None   # set by the menu, applied by run.py
        self.cam_forward = CAM_FORWARD                  # lens ahead of the body centre (m)
        self._render0 = RENDERS[1] if render == "splats" else RENDERS[0]
        self.server = viser.ViserServer(port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
        self.robot_scale = robot_scale
        self._clouds: dict[int, viser.PointCloudHandle] = {}
        self._map_version = -1
        self._traj: list[np.ndarray] = []
        self._traj_dirty = False
        self._pos: np.ndarray | None = None   # smoothed robot position (ground)
        self._yaw = 0.0
        self._cam_pos: np.ndarray | None = None
        self._cam_yaw = 0.0
        self._n = 0
        self.reset_requested = False
        self.reconnect_requested = False
        self._lock = threading.Lock()
        self._tgt: tuple[np.ndarray, float, np.ndarray] | None = None
        self._tgt_vel: tuple[np.ndarray, float] = (np.zeros(3), 0.0)
        self._tgt_time = 0.0
        self._cam_c2w = np.eye(4)
        self._src = None  # camera source for the live picture
        self._build_scene()
        self._build_gui()
        # Free look with the trackpad/mouse: as soon as someone moves the view
        # themselves, stop the automatic camera (mode "Libre").
        self._sent_eye: dict[int, np.ndarray] = {}

        @self.server.on_client_connect
        def _(client) -> None:
            @client.camera.on_update
            def _(cam) -> None:
                sent = self._sent_eye.get(client.client_id)
                if sent is not None and self.mode.value != MODES[3]:
                    if np.linalg.norm(np.asarray(cam.position) - sent) > 0.03:
                        self.mode.value = MODES[3]

        threading.Thread(target=self._animate, daemon=True).start()
        threading.Thread(target=self._stream_live, daemon=True).start()

    live_hz = 12.0   # panel refresh; lower it on a slow link (each picture crosses the tunnel)

    def attach_source(self, src) -> None:
        """Show this source's frames in the panel at their own rate (12 Hz),
        not only when the mapper finishes a step."""
        self._src = src

    def _stream_live(self) -> None:
        last_seq = -1
        while True:
            time.sleep(1 / self.live_hz)
            src = self._src
            if src is None:
                continue
            try:
                seq, frame = src.latest()
                if frame is None or seq == last_seq:
                    continue
                last_seq = seq
                f = 320 / frame.shape[1]
                self.live.image = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            except Exception as e:
                print("live image:", e)

    # -- scene ------------------------------------------------------------
    def _build_scene(self) -> None:
        sc = self.server.scene
        sc.add_grid("/floor", width=40.0, height=40.0, plane="xy", cell_size=0.25,
                    section_size=1.0, infinite_grid=True, fade_distance=15.0,
                    cell_color=(70, 70, 70), section_color=(110, 110, 110))
        k = self.robot_scale
        self.robot = sc.add_frame("/robot", show_axes=False)
        sc.add_box("/robot/body", color=(235, 235, 240),
                   dimensions=(BODY_L * k, BODY_W * k, BODY_H * k),
                   position=(0.0, 0.0, (WHEEL_R + 0.01) * k))
        sc.add_box("/robot/top", color=(40, 40, 45),
                   dimensions=(BODY_L * 0.7 * k, BODY_W * 0.9 * k, 0.03 * k),
                   position=(-0.01 * k, 0.0, (WHEEL_R + 0.01 + BODY_H / 2 + 0.015) * k))
        # Front marker so the heading is obvious from above.
        sc.add_box("/robot/nose", color=(255, 140, 0),
                   dimensions=(0.02 * k, BODY_W * 0.6 * k, 0.02 * k),
                   position=((BODY_L / 2 + 0.005) * k, 0.0, (WHEEL_R + 0.01) * k))
        # Wheel axis must be along the robot's Y axis: rotate viser's cylinder
        # (axis along its local Y after this) by 90 deg about X.
        wheel_q = vtf.SO3.from_x_radians(np.pi / 2).wxyz
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            sc.add_cylinder(f"/robot/wheel{i}", radius=WHEEL_R * k, height=WHEEL_W * k,
                            color=(25, 25, 25), wxyz=wheel_q,
                            position=(sx * WHEELBASE / 2 * k, sy * TRACK / 2 * k, WHEEL_R * k))
        self.frustum = sc.add_camera_frustum("/camera", fov=1.0, aspect=4 / 3, scale=0.08,
                                             color=(255, 140, 0), thickness=0.003)

    def _build_gui(self) -> None:
        g = self.server.gui
        self.status = g.add_markdown("**En attente de la caméra…**")
        self.live = g.add_image(np.zeros((90, 120, 3), np.uint8), label="Caméra", format="jpeg")
        self.mode = g.add_dropdown(
            "Vue", MODES, initial_value=MODES[0],
            hint="Trackpad : cliquer-glisser pour tourner, deux doigts pour zoomer, "
                 "Maj + cliquer-glisser pour se déplacer. Bouger la vue passe en mode Libre.")
        self.sensors = g.add_dropdown(
            "Capteurs", SENSOR_MODES, initial_value=self._sensor0, disabled=not self.d405_available,
            hint="Caméra du robot seule, caméra de profondeur D405 seule, ou les deux "
                 "(profondeur D405 reprojetée dans l'image du robot). Changer réinitialise la carte.")
        mount = self.d405_mount
        where = (f"{100 * mount[0]:.0f} cm devant le centre du robot, {100 * mount[2]:.0f} cm du sol"
                 if mount is not None else "inconnue")
        self.sensor_note = g.add_markdown(
            (f"<small>D405 simulée : profondeur courte portée idéale, extrinsèques exactes. "
             f"Position de montage **supposée** ({where}), à confirmer sur le robot.</small>")
            if self.d405_available else
            "<small>D405 : disponible seulement avec le simulateur (`--sim`). "
            "La D405 réelle n'est pas encore branchée.</small>")
        self.metrics = g.add_markdown("")

        @self.sensors.on_update
        def _(_e) -> None:
            self.sensor_mode_requested = self.sensors.value
            self.status.content = f"**Capteurs : {self.sensors.value}** — carte réinitialisée"
        self.render = g.add_dropdown("Rendu", RENDERS, initial_value=self._render0)
        self.dist = g.add_slider("Distance caméra (m)", 0.2, 5.0, 0.05, 0.9)
        self.point_size = g.add_slider("Taille des points (mm)", 1.0, 30.0, 0.5, 8.0)

        @self.render.on_update
        def _(_e) -> None:
            # Redraw the whole map in the chosen style.
            for h in self._clouds.values():
                h.remove()
            self._clouds.clear()
            for kf in self.mapper.keyframes:
                self._add_cloud(kf)
        self.show_traj = g.add_checkbox("Trajectoire", True)
        reconnect = g.add_button("Reconnecter la caméra", icon="refresh")
        save = g.add_button("Exporter la carte (.ply)")
        reset = g.add_button("Réinitialiser la carte", color="red")

        @reconnect.on_click
        def _(_e) -> None:
            self.reconnect_requested = True
            self.status.content = "**Reconnexion de la caméra…**"

        @self.point_size.on_update
        def _(_e) -> None:
            for h in self._clouds.values():
                h.point_size = self.point_size.value / 1000.0

        @save.on_click
        def _(_e) -> None:
            path = time.strftime("out/map_%Y%m%d_%H%M%S.ply")
            import os
            os.makedirs("out", exist_ok=True)
            n = self.mapper.save_ply(path)
            self.server.gui.add_markdown(f"Carte exportée : `{path}` ({n:,} points)")

        @reset.on_click
        def _(_e) -> None:
            self.reset_requested = True

    # -- updates ----------------------------------------------------------
    def _add_cloud(self, kf: Keyframe) -> None:
        if len(kf.points) == 0:
            return
        name = f"/map/kf_{kf.idx:05d}"
        if self.render.value == RENDERS[1] and len(kf.normals) == len(kf.points):
            c, cov, rgb, op = splats_from_points(kf.points, kf.colors, kf.normals, kf.radii)
            self._clouds[kf.idx] = self.server.scene.add_gaussian_splats(name, c, cov, rgb, op)
        else:
            self._clouds[kf.idx] = self.server.scene.add_point_cloud(
                name, kf.points, kf.colors,
                point_size=self.point_size.value / 1000.0, point_shape="rounded",
                precision="float32")

    def clear(self) -> None:
        for h in self._clouds.values():
            h.remove()
        self._clouds.clear()
        self._traj.clear()
        with self._lock:
            self._pos = self._cam_pos = self._tgt = None
        self.server.scene.add_line_segments("/trajectory", np.zeros((1, 2, 3)), (0, 0, 0), visible=False)

    def update(self, r: TrackResult, fps: float) -> None:
        self._n += 1
        # Map: either everything changed (re-calibration / reset) or one new keyframe.
        if r.map_version != self._map_version:
            # Re-express the trajectory with the world changes since last time
            # (metric calibration); a pure rebuild (refinement) keeps it as is.
            moves = [x for x in self.mapper.transforms if x[0] > self._map_version]
            self._map_version = r.map_version
            for h in self._clouds.values():
                h.remove()
            self._clouds.clear()
            if moves and self._traj:
                p = np.array(self._traj)
                for _, s, R, t in moves:
                    p = s * p @ R.T + t
                p[:, 2] = 0.005  # stays on the floor
                self._traj = list(p)
                self._traj_dirty = True
                with self._lock:
                    self._pos = self._cam_pos = self._tgt = None
            for kf in self.mapper.keyframes:
                self._add_cloud(kf)
        elif r.new_keyframe is not None:
            self._add_cloud(r.new_keyframe)

        # The mapper only gives a pose ~3 times per second: store it as a
        # *target*; the 30 Hz animation thread glides everything towards it.
        R, t = r.c2w[:3, :3], r.c2w[:3, 3]
        fwd = R[:, 2]
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pos = np.array([t[0], t[1], 0.0]) - heading * self.cam_forward * self.robot_scale
        # The pose describes the frame taken when the mapper started, r.dt ago.
        now = time.time() - r.dt
        with self._lock:
            # Velocity from the last two poses, used to keep the robot moving
            # between mapper updates instead of waiting for the next one.
            vel, yaw_rate = np.zeros(3), 0.0
            if self._tgt is not None and not r.lost and self._pos is not None:
                dt_ = now - self._tgt_time
                if 0.02 < dt_ < 1.0:
                    vel = (pos - self._tgt[0]) / dt_
                    yaw_rate = ((yaw - self._tgt[1] + np.pi) % (2 * np.pi) - np.pi) / dt_
                    speed = np.linalg.norm(vel)
                    if speed > 1.5:  # implausible jump: don't extrapolate it
                        vel, yaw_rate = np.zeros(3), 0.0
            self._tgt = (pos, yaw, r.c2w.copy())
            self._tgt_vel = (vel, yaw_rate)
            self._tgt_time = now
            if self._pos is None:  # first pose or after a map reset: jump
                self._pos, self._yaw = pos.copy(), yaw
                self._cam_c2w = r.c2w.copy()

        h, w = r.image.shape[:2]
        self.frustum.fov = float(2 * np.arctan2(h / 2, r.K[1, 1]))
        self.frustum.aspect = w / h
        if self._n % 2 == 0:
            self.frustum.image = r.image
        if self._src is None:
            self.live.image = r.image

        # Trajectory (from the raw pose, so it is not lagging).
        moved = not self._traj or np.linalg.norm(self._traj[-1][:2] - pos[:2]) > 0.02
        if moved or self._traj_dirty:
            self._traj_dirty = False
            if moved:
                self._traj.append(pos + np.array([0.0, 0.0, 0.005]))
            if len(self._traj) >= 2:
                p = np.array(self._traj)
                self.server.scene.add_line_segments(
                    "/trajectory", np.stack([p[:-1], p[1:]], 1), (0, 200, 255),
                    thickness=0.01, visible=self.show_traj.value)

        n_pts = sum(len(k.points) for k in self.mapper.keyframes)
        mp = self.mapper
        if getattr(mp, "sensor_ratios", None):
            scale = (f"métrique (capteur de profondeur, {len(mp.sensor_ratios)} images)" if mp.sensor_scaled
                     else f"provisoire (capteur {len(mp.sensor_ratios)}/{mp.calib_frames})")
        elif mp.scale_final:
            scale = f"métrique (sol mesuré sur {len(mp.ground_heights)} images)"
        elif mp.scale_calibrated:
            scale = f"provisoire (sol {len(mp.ground_heights)}/{mp.calib_frames})"
        else:
            scale = "⚠️ non calibrée (sol non vu)"
        if r.lost:
            track = "🔴 perdu — recherche d'un endroit connu"
        else:
            track = "🟢 OK" if r.quality > 0.5 else "🟠 faible"
        self.status.content = (
            (f"**Capteurs** {self.sensors.value}\n\n" if self.d405_available else "") +
            f"**Suivi** {track} · **{fps:.1f} Hz**\n\n"
            f"Position x={pos[0]:+.2f} m, y={pos[1]:+.2f} m, cap {np.degrees(yaw):+.0f}°\n\n"
            f"{len(self.mapper.keyframes)} images clés · {n_pts:,} points\n\nÉchelle : {scale}"
            + (f"\n\nObjets mobiles masqués : {100 * mp.last_moving_share:.0f} % de l'image"
               if getattr(mp, "masker", None) is not None else ""))

    def show_metrics(self, m: dict | None) -> None:
        """Live comparison with the simulator's ground truth (TruthTracker.metrics())."""
        if m is None:
            self.metrics.content = ""
            return
        lines = ["**Mesures vs vérité terrain** (simulation, indicatif)"]
        if m.get("ready"):
            if m.get("scale_recent") is not None:
                sr = m["scale_recent"]
                lines += [
                    f"- 2 derniers mètres : erreur de position (après alignement) "
                    f"**{100 * m['ate_recent']:.1f} cm**",
                    f"- 2 derniers mètres : erreur d'échelle **{100 * sr:+.0f} %** "
                    f"(carte trop {'grande' if sr > 0 else 'petite'})",
                ]
            se = m["scale_err"]
            lines += [
                f"- Depuis le départ (dérive comprise) : position {100 * m['ate']:.1f} cm "
                f"(max {100 * m['ate_max']:.1f}), échelle {100 * se:+.0f} %",
                f"- Sans correction d'échelle : {100 * m['ate_noscale']:.1f} cm",
            ]
        else:
            lines.append(f"- Position / échelle : en attente (au moins 0,5 m et 10 images suivies ; "
                         f"{m.get('path', 0.0):.2f} m, {m['frames'] - m['lost']} images)")
        if m.get("depth_err") is not None:
            lines.append(f"- Profondeur : écart médian {100 * m['depth_err']:.1f} % (images récentes)")
        lines.append(f"- Images perdues : **{100 * m['lost_rate']:.0f} %** ({m['lost']}/{m['frames']})"
                     + (f", ignorées {m['none']}" if m.get("none") else ""))
        self.metrics.content = "\n".join(lines)

    # -- 30 Hz animation ---------------------------------------------------
    def _animate(self) -> None:
        period = 1.0 / 30.0
        last = time.time()
        while True:
            time.sleep(period)
            now = time.time()
            dt, last = now - last, now
            try:
                self._animate_step(dt)
            except Exception as e:  # never let the render loop die
                print("animation:", e)

    def _animate_step(self, dt: float) -> None:
        with self._lock:
            if self._tgt is None or self._pos is None:
                return
            pos_t, yaw_t, c2w_t = self._tgt
            # Predict where the robot is *now* from its last velocity (at most
            # a quarter second ahead), then approach that quickly: little lag,
            # no jumps.
            vel, yaw_rate = self._tgt_vel
            ahead = min(time.time() - self._tgt_time, 0.4)
            pos_t = pos_t + vel * ahead
            yaw_t = yaw_t + yaw_rate * ahead
            a = 1.0 - np.exp(-dt / 0.08)
            self._pos = self._pos + a * (pos_t - self._pos)
            self._yaw += a * ((yaw_t - self._yaw + np.pi) % (2 * np.pi) - np.pi)
            c = self._cam_c2w
            c[:3, 3] += a * (c2w_t[:3, 3] - c[:3, 3])
            dR = vtf.SO3.from_matrix(c[:3, :3].T @ c2w_t[:3, :3]).log()
            c[:3, :3] = c[:3, :3] @ vtf.SO3.exp(a * dR).as_matrix()
            pos, yaw, cam = self._pos.copy(), self._yaw, c.copy()

        self.robot.position = pos
        self.robot.wxyz = yaw_quat(yaw)
        self.frustum.position = cam[:3, 3]
        self.frustum.wxyz = vtf.SO3.from_matrix(cam[:3, :3]).wxyz

        mode = self.mode.value
        if mode == MODES[3]:
            return
        # The chase camera lags the robot a little more, like a car's 3D view.
        b = 1.0 - np.exp(-dt / 0.15)
        if self._cam_pos is None:
            self._cam_pos, self._cam_yaw = pos.copy(), yaw
        else:
            self._cam_pos = self._cam_pos + b * (pos - self._cam_pos)
            self._cam_yaw += b * ((yaw - self._cam_yaw + np.pi) % (2 * np.pi) - np.pi)
        head = np.array([np.cos(self._cam_yaw), np.sin(self._cam_yaw), 0.0])
        d = self.dist.value
        up = np.array([0.0, 0.0, 1.0])
        if mode == MODES[0]:
            eye = self._cam_pos - head * d + up * (0.55 * d)
            target = self._cam_pos + head * (0.35 * d) + up * 0.05
        elif mode == MODES[1]:
            eye = self._cam_pos + up * (2.5 * d) - head * 0.01
            target = self._cam_pos
            up = head
        else:
            eye = cam[:3, 3]
            target = eye + cam[:3, 2]
        for client in self.server.get_clients().values():
            self._sent_eye[client.client_id] = np.asarray(eye, float)
            with client.atomic():
                client.camera.up_direction = up
                client.camera.position = eye
                client.camera.look_at = target
