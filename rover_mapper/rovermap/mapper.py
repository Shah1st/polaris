"""Live monocular 3D mapping on top of Depth Anything 3.

The idea, kept deliberately simple:

* A handful of *keyframes* form the map. Each stores its camera pose, depth map
  and a coloured point cloud, all expressed in one world frame (Z up, metres,
  floor at z = 0).
* For every incoming camera frame we run DA3 on [a few reference keyframes +
  the new frame]. DA3 returns depth and camera poses for all of them, in its
  own arbitrary frame and scale.
* Because the reference keyframes already have a world pose and a world-scale
  depth map, we can solve for the similarity transform (scale, rotation,
  translation) from DA3's frame to the world frame, and so get the world pose
  of the new frame. That pose is where the robot is right now.
* When the robot has moved enough, the new frame becomes a keyframe and its
  points are added to the map.
* Metric scale and the "up" direction come from fitting the floor plane in the
  first frames and using the known height of the camera above the floor.
"""
from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
# DA3 imports pycolmap only for its COLMAP exporter; the wheel ships a second
# OpenMP runtime that crashes next to torch's on macOS. We never export COLMAP.
sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------
def to_c2w(ext_w2c: np.ndarray) -> np.ndarray:
    """(N,3,4) or (N,4,4) world->camera  ->  (N,4,4) camera->world."""
    n = ext_w2c.shape[0]
    w2c = np.tile(np.eye(4), (n, 1, 1))
    w2c[:, :3, :4] = ext_w2c[:, :3, :4]
    return np.linalg.inv(w2c)


def project_to_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


from .floor import backproject, fit_ground  # noqa: E402, F401 - numpy only, re-exported


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Keyframe:
    idx: int
    rgb: np.ndarray        # original frame fed to the network
    c2w: np.ndarray        # (4,4) camera->world
    K: np.ndarray          # (3,3) at network resolution
    depth: np.ndarray      # (h,w) in world units
    valid: np.ndarray      # (h,w) bool, trustworthy depth
    image: np.ndarray      # (h,w,3) uint8 at network resolution
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    thumb: np.ndarray | None = None  # tiny image signature, to recognise the place
    # Per point, for Gaussian-splat rendering: surface normal (world) and the
    # size of the pixel footprint on the surface (m).
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    last_used: int = 0     # step it last served as a reference (max_kf_data)

    @property
    def has_data(self) -> bool:
        """False once its images were released (max_kf_data): still in the
        map (pose, points), no longer usable as a reference."""
        return self.rgb is not None


def texture(rgb: np.ndarray) -> float:
    """Amount of detail (Laplacian variance at 336 px wide). Measured: blank,
    blown-out or lens-covered frames ~1-2; real iPhone frames 100-2000; TUM
    desk frames > 270."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = 336 / g.shape[1]
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def thumbnail(rgb: np.ndarray) -> np.ndarray:
    """Normalised 40x30 grey image: a cheap global signature for place recognition."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(cv2.resize(g, (40, 30), interpolation=cv2.INTER_AREA), (3, 3), 0)
    v = g.astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Sky-coloured pixels (bright, pale, not warmer than blue), grown by one
    pixel so the pale fringe around leaves goes too. DA3-BASE gives the sky a
    finite depth (a few metres once scaled), which paints white blobs in the
    trees. Colour only, checked on the simulated forest against true depth
    > 100 m: precision 0.96 / recall 0.85 before growing, 0.70 / 0.96 after
    (what it adds are leaf edges against the sky, mostly > 10 m away)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r, b = rgb[..., 0].astype(np.int16), rgb[..., 2].astype(np.int16)
    sky = (hsv[..., 2] > 150) & (hsv[..., 1] < 110) & (b >= r - 5)
    return cv2.dilate(sky.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def net_intrinsics(K_src: np.ndarray, src_hw: tuple[int, int], net_hw: tuple[int, int]) -> np.ndarray:
    """Known camera intrinsics, rescaled to the network's image size."""
    sy, sx = net_hw[0] / src_hw[0], net_hw[1] / src_hw[1]
    K = np.asarray(K_src, np.float64).copy()
    K[0] *= sx
    K[1] *= sy
    return K


class _InTurn:
    """Wraps DA3's InputProcessor so it handles the few images of one call one
    after the other (it otherwise starts a fresh thread pool every call)."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, *a, **k):
        k["sequential"] = True
        return self.proc(*a, **k)


@dataclass
class TrackResult:
    c2w: np.ndarray            # current camera pose in the world
    K: np.ndarray
    image: np.ndarray          # frame at network resolution
    depth: np.ndarray
    new_keyframe: Keyframe | None
    quality: float             # 0 (bad) .. 1 (good) agreement with the map
    dt: float                  # seconds spent in this step
    map_version: int           # bumps when the whole map was re-expressed
    lost: bool = False         # pose not trusted: looking for a known place
    overlap: float = 1.0       # 0..1 geometric agreement with the closest keyframe
    valid: np.ndarray | None = None  # trusted depth pixels, moving objects removed


class Mapper:
    def __init__(
        self,
        model_name: str = "depth-anything/DA3-BASE",
        process_res: int = 336,
        cam_height: float = 0.10,
        n_recent: int = 2,
        n_near: int = 1,
        kf_translation: float = 0.07,   # fraction of the scene depth
        kf_rotation: float = 12.0,      # degrees
        point_stride: int = 3,
        max_depth: float = 6.0,         # metres, points further away are dropped
        voxel: float = 0.015,           # metres, map de-duplication cell
        conf_percentile: float = 25.0,
        calib_frames: int = 6,          # floor fits averaged to fix the metric scale
        min_quality: float = 0.3,       # below: pose rejected, robot considered lost
        min_overlap: float = 0.3,
        # Lost frames before carrying on from the last good pose (0 = never).
        # Staying frozen made the live test unusable as soon as the camera
        # turned to unmapped ground; the cost is that a camera really moved to
        # an unrelated place gets mapped as if it were next to the last pose.
        lost_patience: int = 10,
        # Depths (m) trusted from a depth sensor. RealSense D405: ideal 7-50 cm.
        sensor_range: tuple[float, float] = (0.07, 0.8),
        min_texture: float = 15.0,      # below: frame ignored (blank / blown out)
        min_kf_texture: float = 50.0,   # below: never becomes a keyframe
        mask_moving: bool = True,       # keep people / animals / vehicles out of the map
        # Rover mode: the camera rides at a fixed height with no roll/pitch
        # change. Poses are kept on that plane and every keyframe's scale is
        # re-anchored on the floor, which stops scale drift accumulating.
        ground_robot: bool = False,
        # Outdoors. Shortest distance (m) between keyframes: with the camera
        # 10 cm above the ground the median depth is ~1 m, so the relative rule
        # above made a keyframe every 7 cm. Measured in the forest
        # (scripts/vm/eval_forest.py): 0.15 m gives ~25 % fewer keyframes and
        # slightly less squashed far depth (6-10 m: 0.65 -> 0.70 of the truth)
        # at the same accuracy; 0.3 m was fragile and 0.5 m clearly worse. The
        # tiny, squashed map came mostly from scale drift (see sensor_anchor).
        kf_min_dist: float = 0.0,
        mask_sky: bool = False,         # keep sky-coloured pixels out of the map (points only)
        # Depth sensor: also re-anchor every new keyframe's scale on it, not
        # only the first `calib_frames`. Otherwise the scale drifts along the
        # route (measured in the forest: map depth fell to ~0.5 of the truth).
        sensor_anchor: bool = False,
        # Also drop sky pixels from the depth used for tracking (overlap test,
        # scale ratios), not only from the map.
        mask_sky_tracking: bool = False,
        # While lost, move the pose on at the last measured speed instead of
        # freezing it: a rover keeps driving, and each frozen spell (up to
        # lost_patience frames) otherwise shortens the mapped route.
        lost_extrapolate: bool = False,
        # Largest believable move (m) per processed frame, 0 = no check. In the
        # forest, recovering from "lost" sometimes matched a look-alike place
        # and the pose jumped 2-18 m in one frame (true move ~0.1 m), which
        # then wrecked the map. A rejected jump counts as lost. Measured with
        # lost_extrapolate, over 5 stretches: an allowance growing like the
        # frames lost (needed for a frozen guess) still let recoveries jump
        # 0.5-2.6 m; sqrt(frames lost) around the predicted position: mean
        # ATE 0.81 -> 0.60 m, more frames flagged lost (12 -> 17 %); a fixed
        # one locked tracking out (lost 31 %, ATE 1.25 m).
        max_jump: float = 0.0,
        # > 0: torch CPU threads, and DA3's image preprocessing done in turn
        # instead of in a new 8-thread pool per call. Same outputs; on the VM
        # (8 vCPU shared with the simulator) inference went 315 -> 230 ms and
        # CPU use down ~28 % with 1 thread. 0 = leave torch's default.
        cpu_threads: int = 0,
        # Keyframes that keep their images and depth (~2.5 MB each); beyond,
        # the least recently used ones release them (they stay in the map as
        # points, and stop being references). 0 = keep all. A long live run
        # adds ~50 keyframes a minute and otherwise runs out of memory.
        max_kf_data: int = 0,
        device: str | None = None,
    ):
        from depth_anything_3.api import DepthAnything3

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device).eval()
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
            self.model.input_processor = _InTurn(self.model.input_processor)
        self.process_res = process_res
        self.cam_height = cam_height
        self.n_recent, self.n_near = n_recent, n_near
        self.kf_translation, self.kf_rotation = kf_translation, kf_rotation
        self.point_stride = point_stride
        self.max_depth = max_depth
        self.voxel = voxel
        self.conf_percentile = conf_percentile
        self.calib_frames = max(1, calib_frames)
        self.min_quality, self.min_overlap = min_quality, min_overlap
        self.lost_patience = lost_patience
        self.sensor_range = tuple(sensor_range)
        self.min_texture, self.min_kf_texture = min_texture, min_kf_texture
        self.ground_robot = ground_robot
        self.kf_min_dist = kf_min_dist
        self.mask_sky = mask_sky
        self.sensor_anchor = sensor_anchor
        self.mask_sky_tracking = mask_sky_tracking
        self.lost_extrapolate = lost_extrapolate
        self.max_jump = max_jump
        self.max_kf_data = max(max_kf_data, 60) if max_kf_data > 0 else 0  # never during calibration
        self.masker = None
        if mask_moving:
            from .dynamic import MovingMasker
            self.masker = MovingMasker(self.device)
        self.last_moving_share = 0.0  # share of the image masked as moving (for display)
        self.rng = np.random.default_rng(0)
        self.last_align = (0.0, 0.0, 1.0)
        self.reset()

    def _moving(self, rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        """Pixels of things that move on their own (people…), at resolution hw."""
        if self.masker is None:
            return np.zeros(hw, bool)
        dyn = self.masker(rgb, hw)
        self.last_moving_share = float(dyn.mean())
        return dyn

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        self.keyframes: list[Keyframe] = []
        self.lost = False
        self._lost_n = 0
        # Ground-robot mode: camera orientation relative to a level, yaw-zero
        # body, and camera height, fixed at calibration.
        self._mount: np.ndarray | None = None
        self._cam_z: float | None = None
        self.floor_rescales: list[float] = []
        self.scale_calibrated = False
        # Camera heights above the floor measured on keyframes (world units)
        # and the keyframes they came from; the median fixes the final scale.
        self.ground_heights: list[float] = []
        self.calib_kfs: list[Keyframe] = []
        # Depth sensor (RealSense…): sensor/map depth ratios on keyframes.
        # When present, they set the metric scale instead of the floor height.
        self.sensor_ratios: list[float] = []
        self.sensor_scaled = False
        self.sensor_rescales: list[float] = []
        # (map_version after, s, R, t) for every re-expression of the world,
        # so a viewer can move what it drew instead of wiping it.
        self.transforms: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        self.map_version = getattr(self, "map_version", 0) + 1
        self._voxels = np.zeros(0, np.int64)
        self._sharp_hist: list[float] = []
        self.last_c2w = np.eye(4)
        self._vel = np.zeros(3)   # world translation per tracked frame (lost_extrapolate)
        self._step_n = 0
        self._prev_ok = False

    # -- network ----------------------------------------------------------
    def _infer(self, images: list[np.ndarray]):
        p = self.model.inference(images, process_res=self.process_res)
        n = len(images)
        depth = np.asarray(p.depth, np.float32)
        conf = getattr(p, "conf", None)
        conf = np.ones_like(depth) if conf is None else np.asarray(conf, np.float32)
        ext = getattr(p, "extrinsics", None)
        c2w = np.tile(np.eye(4), (n, 1, 1)) if ext is None else to_c2w(np.asarray(ext, np.float64))
        K = np.asarray(p.intrinsics, np.float64)
        imgs = np.asarray(p.processed_images)
        return depth, conf, c2w, K, imgs

    def _valid_mask(self, depth: np.ndarray, conf: np.ndarray,
                    image: np.ndarray | None = None, thr: float | None = None) -> np.ndarray:
        if thr is None:  # callers checking one view several times pass it in
            thr = np.percentile(conf, self.conf_percentile)
        valid = (conf >= thr) & np.isfinite(depth) & (depth > 1e-6)
        if self.mask_sky_tracking and image is not None:
            valid &= ~sky_mask(image)
        # Drop "flying pixels" on depth discontinuities.
        gx = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3))
        valid &= (gx + gy) / 8.0 < 0.04 * depth
        return valid

    # -- map --------------------------------------------------------------
    def _make_points(self, kf: Keyframe) -> None:
        s = self.point_stride
        pts_c = backproject(kf.depth, kf.K, s)
        mask = kf.valid[::s, ::s] & (kf.depth[::s, ::s] < self.max_depth)
        if self.mask_sky:  # map only: tracking keeps every pixel (measured: masking there did not help)
            mask &= ~sky_mask(kf.image)[::s, ::s]
        P = pts_c[mask] @ kf.c2w[:3, :3].T + kf.c2w[:3, 3]
        C = kf.image[::s, ::s][mask]
        # Skip points landing in cells that older keyframes already filled:
        # avoids doubled walls and keeps the cloud size bounded when revisiting.
        cells = np.floor(P / self.voxel).astype(np.int64) + (1 << 20)
        keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
        keys_u, first = np.unique(keys, return_index=True)
        # _voxels stays sorted: binary search + insert instead of isin/union1d,
        # which re-sorted the whole map at every keyframe (~0.2 s at 3 M cells).
        idx = np.searchsorted(self._voxels, keys_u)
        fresh = ~((idx < len(self._voxels)) & (self._voxels[np.minimum(idx, len(self._voxels) - 1)] == keys_u)
                  if len(self._voxels) else np.zeros(len(keys_u), bool))
        self._voxels = np.insert(self._voxels, idx[fresh], keys_u[fresh])
        sel = first[fresh]
        kf.points = P[sel].astype(np.float32)
        kf.colors = C[sel].astype(np.uint8)
        # Surface normals from the point grid (camera frame), facing the camera.
        du = np.gradient(pts_c, axis=1)
        dv = np.gradient(pts_c, axis=0)
        N = np.cross(du, dv)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
        N = np.where((N * pts_c).sum(-1, keepdims=True) > 0, -N, N)
        kf.normals = (N[mask] @ kf.c2w[:3, :3].T)[sel].astype(np.float32)
        kf.radii = (kf.depth[::s, ::s][mask] * s / kf.K[0, 0])[sel].astype(np.float32)

    def _rebuild_points(self) -> None:
        self._voxels = np.zeros(0, np.int64)
        for kf in self.keyframes:
            if kf.has_data:
                self._make_points(kf)
            elif len(kf.points):  # images released: keep its points, just claim their cells
                cells = np.floor(kf.points / self.voxel).astype(np.int64) + (1 << 20)
                keys = np.unique((cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2])
                self._voxels = np.union1d(self._voxels, keys)
        self.map_version += 1

    def _apply_similarity(self, s: float, R: np.ndarray, t: np.ndarray) -> None:
        """Re-express the whole map: X_new = s * R @ X_old + t."""
        for kf in self.keyframes:
            c2w = np.eye(4)
            c2w[:3, :3] = R @ kf.c2w[:3, :3]
            c2w[:3, 3] = s * R @ kf.c2w[:3, 3] + t
            kf.c2w = c2w
            if kf.has_data:
                kf.depth = kf.depth * s
            else:
                kf.points = (s * kf.points @ R.T + t).astype(np.float32)
                kf.normals = (kf.normals @ R.T).astype(np.float32)
                kf.radii = (kf.radii * s).astype(np.float32)
        self._vel = s * R @ self._vel
        self._rebuild_points()
        self.transforms.append((self.map_version, s, R.copy(), np.asarray(t, float).copy()))

    def _try_calibrate(self, kf: Keyframe) -> bool:
        """Use the floor seen by `kf` to make the world Z-up, metric, floor at z=0."""
        g = fit_ground(kf.depth, kf.K, kf.valid, self.rng)
        if g is None:
            return False
        n_cam, h = g  # floor normal (camera frame) and camera height, current units
        # With a depth sensor the scale is the sensor's job: only level the map.
        s = 1.0 if self.sensor_ratios else self.cam_height / h
        up = kf.c2w[:3, :3] @ n_cam                       # floor normal, current world
        fwd = kf.c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])  # viewing direction
        x = fwd - (fwd @ up) * up
        if np.linalg.norm(x) < 1e-6:
            return False
        x /= np.linalg.norm(x)
        y = np.cross(up, x)
        R = np.stack([x, y, up])          # rows: new axes in old coordinates
        # Put this keyframe's camera at (0, 0, cam_height): new world origin is
        # the point on the floor straight below it, x pointing forward.
        t = np.array([0.0, 0.0, h * s]) - s * R @ kf.c2w[:3, 3]
        self._apply_similarity(s, R, t)
        self.scale_calibrated = True
        self.ground_heights = [self.cam_height]
        self.calib_kfs = [kf]
        if self.ground_robot:
            Rc = kf.c2w[:3, :3]
            yaw0 = float(np.arctan2(Rc[1, 2], Rc[0, 2]))
            self._mount = _rot_z(-yaw0) @ Rc
            self._cam_z = float(kf.c2w[2, 3])
        return True

    # -- ground-robot constraints -------------------------------------------
    def _planar(self, c2w: np.ndarray) -> np.ndarray:
        """Keep only x, y and heading; height and tilt come from calibration."""
        if self._mount is None:
            return c2w
        fwd = c2w[:3, 2]
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        out = c2w.copy()
        out[:3, :3] = _rot_z(yaw) @ self._mount
        out[2, 3] = self._cam_z
        return out

    def _floor_anchor(self, depth: np.ndarray, K: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Rescale a new keyframe's depth so the floor sits at the known camera
        height. Each keyframe is then metric on its own, so tracking errors in
        scale do not pile up from one keyframe to the next."""
        if self._cam_z is None:
            return depth
        g = fit_ground(depth, K, valid, self.rng)
        if g is None:
            return depth
        h = g[1]
        if not 0.5 * self._cam_z < h < 2.0 * self._cam_z:
            return depth
        k = float(np.clip(self._cam_z / h, 0.8, 1.25))
        self.floor_rescales.append(k)
        return depth * k

    @property
    def scale_final(self) -> bool:
        if self.sensor_ratios:
            return self.sensor_scaled
        return self.scale_calibrated and len(self.ground_heights) >= self.calib_frames

    def _sensor_scale(self, kf: Keyframe, sensor_depth: np.ndarray) -> bool:
        """Compare a depth sensor's metres with the map's depth on the same
        pixels. After `calib_frames` keyframes, rescale the whole map by the
        median ratio. Returns True if the map was rescaled.

        Tested on TUM: the map's depth error is the same at all distances
        (within ~3%), so a short-range sensor fixes the scale of the room too.
        """
        h, w = kf.depth.shape
        sd = cv2.resize(sensor_depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        ok = kf.valid & (sd > self.sensor_range[0]) & (sd < self.sensor_range[1])
        if ok.sum() < 200:
            return False
        self.sensor_ratios.append(float(np.median(sd[ok] / kf.depth[ok])))
        if self.sensor_scaled or len(self.sensor_ratios) < self.calib_frames:
            return False
        s = float(np.median(self.sensor_ratios))
        self.sensor_scaled = True
        if abs(s - 1.0) < 0.005:
            return False
        # Origin is on the floor: scaling about it keeps the floor at z = 0.
        self._apply_similarity(s, np.eye(3), np.zeros(3))
        self.sensor_ratios = [r / s for r in self.sensor_ratios]
        return True

    def _sensor_anchor(self, depth: np.ndarray, valid: np.ndarray, sensor_depth: np.ndarray) -> np.ndarray:
        """Rescale a new keyframe's depth so it agrees with the depth sensor on
        the pixels both see, as _floor_anchor does with the floor. The step is
        capped so one bad sensor frame (occlusion between the two cameras)
        cannot throw the map off; repeated keyframes still converge."""
        h, w = depth.shape
        sd = cv2.resize(sensor_depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        ok = valid & (sd > self.sensor_range[0]) & (sd < self.sensor_range[1])
        if ok.sum() < 200:
            return depth
        k = float(np.clip(np.median(sd[ok] / depth[ok]), 0.7, 1.4))
        self.sensor_rescales.append(k)
        return depth * k

    def _refine_scale(self, kf: Keyframe) -> bool:
        """Add one more floor measurement; once enough, rescale the map so the
        *median* camera height equals cam_height (one noisy fit no longer sets
        the scale of everything). Returns True if the map was rescaled."""
        g = fit_ground(kf.depth, kf.K, kf.valid, self.rng)
        if g is None:
            return False
        h = g[1]
        # Reject fits that clearly found another plane (table top, wall base…).
        if not 0.5 * self.cam_height < h < 2.0 * self.cam_height:
            return False
        self.ground_heights.append(h)
        self.calib_kfs.append(kf)
        if len(self.ground_heights) < self.calib_frames:
            return False
        s = self.cam_height / float(np.median(self.ground_heights))
        if abs(s - 1.0) < 0.005:
            return False
        # The world origin is on the floor, so scaling about it keeps z=0 the floor.
        self._apply_similarity(s, np.eye(3), np.zeros(3))
        self.ground_heights = [h * s for h in self.ground_heights]
        return True

    def _add_keyframe(self, rgb, c2w, K, depth, valid, image) -> Keyframe:
        # Copies: `image`, `K` (and `depth` of a first frame) are views into the
        # network's whole batch, which a keyframe would otherwise keep alive
        # (measured: ~6 MB per keyframe instead of ~2.5 MB).
        kf = Keyframe(len(self.keyframes), rgb, c2w, np.array(K), np.array(depth), valid, np.array(image),
                      thumb=thumbnail(rgb), last_used=self._step_n)
        self.keyframes.append(kf)
        self._make_points(kf)
        if self.max_kf_data:
            held = [k for k in self.keyframes if k.has_data]
            for k in sorted(held, key=lambda k: k.last_used)[: max(0, len(held) - self.max_kf_data)]:
                k.rgb = k.depth = k.valid = k.image = None
        return kf

    def _select_refs(self) -> list[Keyframe]:
        """Keyframes closest to where we were a moment ago, looking the same way.

        Chosen by place, not by creation time: when the robot drives through
        an area it has already mapped, it is located against that map instead
        of against wherever mapping last stopped."""
        kfs = [k for k in self.keyframes if k.has_data]
        n = self.n_recent + self.n_near
        if len(kfs) <= n:
            return list(kfs)
        pos = self.last_c2w[:3, 3]
        fwd = self.last_c2w[:3, 2]
        last = self.keyframes[-1]
        scale = max(1e-6, float(np.median(last.depth[last.valid])) if last.valid.any() else 1.0)
        cost = [np.linalg.norm(k.c2w[:3, 3] - pos) / scale + (1.0 - k.c2w[:3, 2] @ fwd) for k in kfs]
        return [kfs[i] for i in sorted(np.argsort(cost)[:n])]

    def _retrieve_refs(self, rgb: np.ndarray) -> list[Keyframe]:
        """Lost: find the keyframe that looks most like the current image, and
        its neighbours in the map."""
        q = thumbnail(rgb)
        kfs = [k for k in self.keyframes if k.has_data]
        sims = [float(k.thumb @ q) for k in kfs]
        best = kfs[int(np.argmax(sims))]
        return [best] + self.neighbors(best, self.n_recent + self.n_near - 1)

    def _overlap(self, kf: Keyframe, c2w: np.ndarray, depth: np.ndarray,
                 valid: np.ndarray, K: np.ndarray) -> float:
        """Share of the current view's 3D points that land on the same surface in
        keyframe `kf` (depth within 10%). Low when the pose is wrong, or when the
        two views do not show the same place."""
        P = backproject(depth, K, 6)[valid[::6, ::6]]
        if len(P) < 50:
            return 0.0
        W = P @ c2w[:3, :3].T + c2w[:3, 3]
        C = (W - kf.c2w[:3, 3]) @ kf.c2w[:3, :3]  # into the keyframe camera
        z = C[:, 2]
        front = z > 1e-3
        u = np.round(kf.K[0, 0] * C[front, 0] / z[front] + kf.K[0, 2]).astype(int)
        v = np.round(kf.K[1, 1] * C[front, 1] / z[front] + kf.K[1, 2]).astype(int)
        h, w = kf.depth.shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inside.sum() < 0.15 * len(P):  # hardly any common field of view
            return 0.0
        u, v, zf = u[inside], v[inside], z[front][inside]
        ok = kf.valid[v, u]
        if ok.sum() < 30:
            return 0.0
        dk = kf.depth[v[ok], u[ok]]
        return float(np.mean(np.abs(zf[ok] - dk) / dk < 0.10))

    # -- alignment ----------------------------------------------------------
    def _align(self, refs: list[Keyframe], depth: np.ndarray, L: np.ndarray):
        """Bring one network output into the world frame.

        `depth`/`L` hold the network's depths and camera->network poses for
        [*refs, new view]. The refs already have world poses and world-scale
        depths, which fixes the similarity network -> world. Returns the world
        pose of the last view, the scale, and a 0..1 agreement score.
        """
        m = len(refs)
        # Scale: the same pixels of a keyframe, stored depth vs fresh depth.
        ratios = []
        for i, k in enumerate(refs):
            kd, kv = k.depth, k.valid
            if kd.shape != depth[i].shape:  # keyframe stored at another resolution
                hw = (depth[i].shape[1], depth[i].shape[0])
                kd = cv2.resize(kd, hw, interpolation=cv2.INTER_NEAREST)
                kv = cv2.resize(kv.astype(np.uint8), hw, interpolation=cv2.INTER_NEAREST) > 0
            ok = kv & (depth[i] > 1e-6)
            if ok.sum() > 100:
                ratios.append(float(np.median(kd[ok] / depth[i][ok])))
        s = float(np.median(ratios)) if ratios else 1.0
        # Rotation: average of the per-keyframe estimates.
        R = project_to_so3(sum(k.c2w[:3, :3] @ L[i, :3, :3].T for i, k in enumerate(refs)))
        # Translation.
        ts = np.stack([k.c2w[:3, 3] - s * R @ L[i, :3, 3] for i, k in enumerate(refs)])
        t = np.median(ts, axis=0)

        c2w = np.eye(4)
        c2w[:3, :3] = R @ L[m, :3, :3]
        c2w[:3, 3] = s * R @ L[m, :3, 3] + t

        # How well do the reference keyframes agree once aligned?
        scene = float(np.median(depth[m])) * s
        pos_err = float(np.max(np.linalg.norm(ts - t, axis=1))) / max(scene, 1e-6)
        rot_err = max(rot_angle_deg(R @ L[i, :3, :3], k.c2w[:3, :3]) for i, k in enumerate(refs))
        quality = float(np.clip(1.0 - max(pos_err / 0.15, rot_err / 15.0), 0.0, 1.0))
        self.last_align = (pos_err, rot_err, s)  # for logging / diagnosis
        return c2w, s, quality

    # -- background refinement ----------------------------------------------
    def neighbors(self, kf: Keyframe, n: int = 4) -> list[Keyframe]:
        """The n keyframes seeing the most similar part of the scene as `kf`."""
        others = [k for k in self.keyframes if k is not kf and k.has_data]
        if not others:
            return []
        scene = float(np.median(kf.depth[kf.valid])) if kf.valid.any() else 1.0
        fwd = kf.c2w[:3, 2]
        cost = [np.linalg.norm(k.c2w[:3, 3] - kf.c2w[:3, 3]) / scene + (1.0 - k.c2w[:3, 2] @ fwd)
                for k in others]
        return [others[i] for i in np.argsort(cost)[:n]]

    def refine_keyframe(self, kf: Keyframe, n_neighbors: int = 4, res: int = 504,
                        rebuild: bool = True) -> float:
        """Re-estimate one keyframe's pose and depth with more views at a higher
        resolution. This is the slow job meant to run while the robot moves.
        Returns the agreement score of the neighbours (0..1)."""
        refs = self.neighbors(kf, n_neighbors)
        if not refs:
            return 0.0
        old_res, self.process_res = self.process_res, res
        try:
            depth, conf, L, K, imgs = self._infer([k.rgb for k in refs] + [kf.rgb])
        finally:
            self.process_res = old_res
        c2w, s, quality = self._align(refs, depth, L)
        if quality < 0.3:  # neighbours disagree: keep what we had
            return quality
        m = len(refs)
        kf.c2w, kf.K, kf.image = c2w, K[m], imgs[m]
        kf.depth = depth[m] * s
        kf.valid = self._valid_mask(kf.depth, conf[m])
        if rebuild:
            self._rebuild_points()
        return quality

    # -- main entry -------------------------------------------------------
    def step(self, rgb: np.ndarray, sensor_depth: np.ndarray | None = None,
             K_src: np.ndarray | None = None) -> TrackResult | None:
        """Process one camera frame. `sensor_depth`: optional depth image in
        metres, pixel-aligned with `rgb` (0 = no measurement). `K_src`: the
        camera's intrinsics at the size of `rgb`, when known; they replace the
        network's own estimate (30-40 % too long in the forest) for the current
        view. Measured in the forest it made tracking worse (ATE 1.42 -> 2.49 m:
        the network's depth and poses go with its own focal length).
        Returns None when the frame has too little detail to be used."""
        t0 = time.time()
        src_hw = rgb.shape[:2]
        if max(rgb.shape[:2]) > 720:
            f = 720 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

        # Blank, blown-out or covered frames carry no geometry: skip them.
        tex = texture(rgb)
        if tex < self.min_texture or (not self.keyframes and tex < self.min_kf_texture):
            return None

        # --- first frame: it defines the world -----------------------------
        if not self.keyframes:
            depth, conf, _, K, imgs = self._infer([rgb])
            if K_src is not None:
                K[0] = net_intrinsics(K_src, src_hw, depth[0].shape)
            valid = self._valid_mask(depth[0], conf[0], imgs[0]) & ~self._moving(rgb, depth[0].shape)
            # Until the floor is found, assume a level camera: map OpenCV axes
            # (x right, y down, z fwd) to world (x fwd, y left, z up).
            c2w = np.eye(4)
            c2w[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
            c2w[2, 3] = self.cam_height
            kf = self._add_keyframe(rgb, c2w, K[0], depth[0], valid, imgs[0])
            self._try_calibrate(kf)
            if sensor_depth is not None:
                self._sensor_scale(kf, sensor_depth)
            self.last_c2w = kf.c2w
            return TrackResult(kf.c2w, kf.K, kf.image, kf.depth, kf, 1.0, time.time() - t0,
                               self.map_version, valid=kf.valid)

        # --- track against reference keyframes ----------------------------
        # Normally: keyframes around the last known pose. When lost: the
        # keyframes that look most like the current image (relocalisation).
        # While lost, alternate: look for a lookalike place (the robot came back
        # somewhere known), or stay around the last good pose (it moved on into
        # new ground: it cannot have teleported).
        by_look = self.lost and self._lost_n % 2 == 1
        refs = self._retrieve_refs(rgb) if by_look else self._select_refs()
        self._step_n += 1
        for k in refs:
            k.last_used = self._step_n
        depth, conf, L, K, imgs = self._infer([k.rgb for k in refs] + [rgb])
        m = len(refs)
        if K_src is not None:
            K[m] = net_intrinsics(K_src, src_hw, depth[m].shape)
        dyn = self._moving(rgb, depth[m].shape)  # people etc.: not part of the map
        thr_m = np.percentile(conf[m], self.conf_percentile)
        # Use only the references the current view really overlaps. One bad or
        # unrelated keyframe among them must not drag the pose away (seen live:
        # a featureless first keyframe made every later frame fail).
        per_ov = []
        for j, k in enumerate(refs):
            c_j, s_j, _ = self._align([k], depth[[j, m]], L[[j, m]])
            d_j = depth[m] * s_j
            per_ov.append(self._overlap(k, c_j, d_j, self._valid_mask(d_j, conf[m], imgs[m], thr_m) & ~dyn, K[m]))
        use = [j for j in range(m) if per_ov[j] >= self.min_overlap] or [int(np.argmax(per_ov))]
        sub = [refs[j] for j in use]
        c2w, s, quality = self._align(sub, depth[use + [m]], L[use + [m]])
        d_cur = depth[m] * s
        valid = self._valid_mask(d_cur, conf[m], imgs[m], thr_m) & ~dyn
        scene = float(np.median(d_cur[valid])) if valid.any() else 1.0
        overlap = max(self._overlap(k, c2w, d_cur, valid, K[m]) for k in sub)

        self.last_candidate = c2w  # for diagnosis: the pose even when rejected
        # Where the camera can be now. Frozen guess: anywhere within one
        # frame's move per frame lost. Extrapolating: around the predicted
        # position, the doubt growing like sqrt(frames lost).
        if self.lost_extrapolate:
            pred = self.last_c2w[:3, 3] + self._vel * (self._lost_n + 1)
            allowed = self.max_jump * np.sqrt(self._lost_n + 1)
        else:
            pred, allowed = self.last_c2w[:3, 3], self.max_jump * (self._lost_n + 1)
        jump_ok = self.max_jump <= 0 or np.linalg.norm(c2w[:3, 3] - pred) <= allowed
        # Accept the pose only if the references agree with each other AND the
        # current view lands on the same surfaces as one of them.
        if quality < self.min_quality or overlap < self.min_overlap or not jump_ok:
            self._lost_n += 1
            self._prev_ok = False
            # Where the robot probably is: frozen, or moved on at its last speed.
            guess = self.last_c2w[:3, 3] + (self._vel * self._lost_n if self.lost_extrapolate else 0.0)
            # Lost for a while in a place that matches nothing known (the robot
            # drove on into new ground): rather than stay stuck forever, carry
            # on from the keyframes around the last good pose, if they agree.
            if not (self.lost_patience > 0 and self._lost_n >= self.lost_patience
                    and not by_look and quality >= self.min_quality):
                self.lost = True
                shown = self.last_c2w.copy()
                shown[:3, 3] = guess
                return TrackResult(shown, K[m], imgs[m], d_cur, None, min(quality, overlap),
                                   time.time() - t0, self.map_version, lost=True, overlap=overlap)
            if overlap < self.min_overlap or not jump_ok:
                # Restarting with no view in common with any keyframe: the
                # network's translation is unreliable (seen: 1.5 m jumps). The
                # camera cannot have gone far in ~1.5 s, so keep the last good
                # position (or where its last speed took it) and only take the
                # estimated orientation.
                c2w = c2w.copy()
                c2w[:3, 3] = guess
        restarted = self._lost_n > 0
        self.lost = False
        self._lost_n = 0
        if self.ground_robot:
            c2w = self._planar(c2w)

        # --- keyframe decision --------------------------------------------
        gray = cv2.cvtColor(imgs[m], cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        self._sharp_hist = (self._sharp_hist + [sharp])[-30:]
        blurry = sharp < 0.5 * float(np.median(self._sharp_hist))

        def novelty(k: Keyframe) -> float:
            dt_ = np.linalg.norm(k.c2w[:3, 3] - c2w[:3, 3]) / max(self.kf_translation * scene,
                                                                  self.kf_min_dist)
            dr_ = rot_angle_deg(k.c2w[:3, :3], c2w[:3, :3]) / self.kf_rotation
            return max(dt_, dr_)

        new_kf = None
        if min(novelty(k) for k in self.keyframes) > 1.0 and not blurry and tex >= self.min_kf_texture:
            if self.ground_robot and not self.sensor_ratios:
                d_cur = self._floor_anchor(d_cur, K[m], valid)
            if self.sensor_anchor and self.sensor_scaled and sensor_depth is not None:
                d_cur = self._sensor_anchor(d_cur, valid, sensor_depth)
            new_kf = self._add_keyframe(rgb, c2w, K[m], d_cur, valid, imgs[m])
            # The floor gives the up direction (and a first guess of the scale);
            # a depth sensor, when there is one, gives the scale.
            if not self.scale_calibrated and len(self.keyframes) <= 15:
                if self._try_calibrate(new_kf):
                    c2w, d_cur = new_kf.c2w, new_kf.depth
            elif (sensor_depth is None and not self.sensor_ratios and self.scale_calibrated
                  and not self.scale_final and len(self.keyframes) <= 40):
                if self._refine_scale(new_kf):
                    c2w, d_cur = new_kf.c2w, new_kf.depth
            if sensor_depth is not None:
                if self._sensor_scale(new_kf, sensor_depth):
                    c2w, d_cur = new_kf.c2w, new_kf.depth

        # Speed for lost_extrapolate, from two tracked frames in a row only.
        step_ = c2w[:3, 3] - self.last_c2w[:3, 3]
        if self._prev_ok and np.linalg.norm(step_) < 0.5:
            self._vel = 0.5 * self._vel + 0.5 * step_
        self._prev_ok = not restarted
        self.last_c2w = c2w
        return TrackResult(c2w, K[m], imgs[m], d_cur, new_kf, quality, time.time() - t0,
                           self.map_version, overlap=overlap, valid=valid)

    # -- export -----------------------------------------------------------
    def all_points(self):
        if not self.keyframes:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        return (np.concatenate([k.points for k in self.keyframes]),
                np.concatenate([k.colors for k in self.keyframes]))

    def save_ply(self, path: str) -> int:
        P, C = self.all_points()
        with open(path, "wb") as f:
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(P)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode())
            rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = P, C
            f.write(rec.tobytes())
        return len(P)
