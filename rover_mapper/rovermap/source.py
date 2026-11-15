"""Frame sources: robot MJPEG stream, webcam, or a video file, and the
simulator's snapshot endpoint (robot camera + simulated D405 + true poses).

Every source runs in its own thread and only keeps the most recent frame, so
the mapper always works on what the camera sees *now* (no latency build-up).
"""
from __future__ import annotations

import io
import threading
import time

import cv2
import numpy as np

# Sensor modes ("Capteurs" menu). What the mapper gets in each:
SENSOR_MODES = ("Caméra du robot (RGB)", "D405 seule", "Les deux")
#   RGB      -> mapper.step(robot rgb)                      scale from the floor
#   D405     -> mapper.step(d405 rgb, d405 depth)           scale from the sensor
#   Les deux -> mapper.step(robot rgb, d405 depth reprojected into the robot
#               camera with the exact extrinsics: sparse, pixel-aligned)
MODE_RGB, MODE_D405, MODE_BOTH = SENSOR_MODES


def reproject_depth(depth: np.ndarray, K_src: np.ndarray, T_src: np.ndarray,
                    K_dst: np.ndarray, T_dst: np.ndarray, size_dst: tuple[int, int]) -> np.ndarray:
    """Depth image of one camera -> sparse depth image of another camera.

    depth: metres along the source optical axis (0 = none); T_*: camera->world
    4x4, OpenCV axes; size_dst: (w, h). Returns float32 (h, w) depth along the
    destination optical axis, nearest surface kept per pixel, 0 elsewhere.
    """
    w, h = size_dst
    vs, us = np.nonzero(depth > 0)
    z = depth[vs, us].astype(np.float64)
    P = np.stack([(us - K_src[0, 2]) / K_src[0, 0] * z, (vs - K_src[1, 2]) / K_src[1, 1] * z, z], 1)
    T = np.linalg.inv(T_dst) @ T_src                       # source cam -> destination cam
    Q = P @ T[:3, :3].T + T[:3, 3]
    front = Q[:, 2] > 1e-3
    Q = Q[front]
    u = np.round(K_dst[0, 0] * Q[:, 0] / Q[:, 2] + K_dst[0, 2]).astype(np.int64)
    v = np.round(K_dst[1, 1] * Q[:, 1] / Q[:, 2] + K_dst[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zq = u[inside], v[inside], Q[inside, 2]
    out = np.full(h * w, np.inf)
    np.minimum.at(out, v * w + u, zq)                      # z-buffer: nearest wins
    out[~np.isfinite(out)] = 0.0
    return out.reshape(h, w).astype(np.float32)


def d405_in_rgb(snap: dict) -> np.ndarray:
    """D405 depth reprojected into the robot camera, computed once per snapshot
    (40-60 ms): the live preview and the mapper both need it."""
    sd = snap.get("_d405_in_rgb")
    if sd is None:
        h, w = snap["rgb"].shape[:2]
        sd = reproject_depth(snap["d405_depth"], snap["d405_K"], snap["d405_T"], snap["K"], snap["T"], (w, h))
        snap["_d405_in_rgb"] = sd
    return sd


def sensor_inputs(snap: dict, mode: str):
    """(image, sensor depth or None, K, camera->world truth, true depth) fed to
    the mapper in `mode`, all from one simulation snapshot."""
    if mode == MODE_D405:
        return snap["d405_rgb"], snap["d405_depth"], snap["d405_K"], snap["d405_T"], snap["d405_depth_true"]
    if mode == MODE_BOTH:
        return snap["rgb"], d405_in_rgb(snap), snap["K"], snap["T"], snap["depth"]
    return snap["rgb"], None, snap["K"], snap["T"], snap["depth"]


def mode_settings(mode: str, cam_height: float, d405_height: float,
                  default_range: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    """(mapper.cam_height, mapper.sensor_range) for a sensor mode.

    The fed camera's height above the floor: the robot camera's, or the D405
    mount height. In "Les deux" the D405 depth has already been cut to its
    valid range in its own frame; once moved into the robot camera it lies at
    0.45-1.1 m from that camera, so the mapper's window is widened."""
    if mode == MODE_D405:
        return d405_height, tuple(default_range)
    if mode == MODE_BOTH:
        return cam_height, (default_range[0], 2.0)
    return cam_height, tuple(default_range)


def umeyama(a: np.ndarray, b: np.ndarray, with_scale: bool = True):
    """Similarity (s, R, t) minimising |s R a + t - b| (s = 1 if not with_scale)."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((A ** 2).sum(1).mean(), 1e-12)) if with_scale else 1.0
    return s, R, mb - s * R @ ma


class TruthTracker:
    """Estimated vs true camera positions of one mapping run (simulation).

    Position error is measured after a similarity alignment (rotation,
    translation AND scale) of the estimated trajectory onto the true one, so
    it says how well the *shape* of the path is recovered. The scale of that
    alignment is reported on its own as the scale error of the map.
    """

    def __init__(self, min_frames: int = 10, min_path: float = 0.5):
        self.min_frames, self.min_path = min_frames, min_path
        self.reset()

    def reset(self) -> None:
        self.est, self.true, self.lost, self.dts = [], [], [], []
        self.n_none = 0                 # frames the mapper refused (no detail)
        self.depth_err: list[float] = []  # per frame: median |d - d_true| / d_true
        self._seen = -1

    def add(self, mapper, r, T_true: np.ndarray, depth_true: np.ndarray | None = None) -> None:
        # The map may have been re-expressed (metric calibration): move the
        # positions recorded so far with it, as the viewer does.
        for v, s, R, t in mapper.transforms:
            if v > self._seen:
                self.est = [s * R @ p + t for p in self.est]
        self._seen = max([self._seen] + [x[0] for x in mapper.transforms])
        self.est.append(r.c2w[:3, 3].copy())
        self.true.append(np.asarray(T_true)[:3, 3].copy())
        self.lost.append(bool(r.lost))
        self.dts.append(float(r.dt))
        if depth_true is not None and r.valid is not None and not r.lost:
            h, w = r.depth.shape
            dt_ = cv2.resize(depth_true.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            ok = r.valid & (dt_ > 0.05) & (dt_ < 8.0)
            if ok.sum() > 100:
                self.depth_err.append(float(np.median(np.abs(r.depth[ok] - dt_[ok]) / dt_[ok])))

    def metrics(self) -> dict:
        n = len(self.est)
        lost = np.array(self.lost, bool)
        out = {"frames": n, "lost": int(lost.sum()), "none": self.n_none,
               "lost_rate": float(lost.mean()) if n else 0.0,
               "dt": float(np.median(self.dts)) if self.dts else 0.0,
               "depth_err": float(np.median(self.depth_err[-30:])) if self.depth_err else None,
               "ready": False}
        if n == 0:
            return out
        E, G = np.array(self.est)[~lost], np.array(self.true)[~lost]
        path = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum()) if len(G) > 1 else 0.0
        out["path"] = path
        if len(E) < self.min_frames or path < self.min_path:
            return out
        s, R, t = umeyama(E, G)
        if s < 1e-9:  # estimated positions all equal (never really tracked): no alignment
            return out
        err = np.linalg.norm(s * E @ R.T + t - G, axis=1)
        _, R1, t1 = umeyama(E, G, with_scale=False)
        err1 = np.linalg.norm(E @ R1.T + t1 - G, axis=1)
        out.update(ready=True, ate=float(np.sqrt((err ** 2).mean())), ate_max=float(err.max()),
                   ate_noscale=float(np.sqrt((err1 ** 2).mean())),
                   # map metres per true metre, minus 1: +0.12 = map 12 % too big
                   scale_err=float(1.0 / s - 1.0))
        # The same over the last ~2 m only: the scale drifts along the route,
        # and over the whole route the alignment mixes drift into the scale.
        seg = np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1)[::-1])
        k = int(np.searchsorted(seg, 2.0)) + 2
        if seg[-1] >= 2.0 and k >= 8:
            s2, R2, t2 = umeyama(E[-k:], G[-k:])
            if s2 < 1e-9:
                return out
            err2 = np.linalg.norm(s2 * E[-k:] @ R2.T + t2 - G[-k:], axis=1)
            out.update(scale_recent=float(1.0 / s2 - 1.0), ate_recent=float(np.sqrt((err2 ** 2).mean())))
        return out


def depth_preview(depth: np.ndarray, max_m: float = 1.0) -> np.ndarray:
    """Colour picture of a depth image (near = warm, none = black), RGB uint8."""
    d = np.clip(depth / max_m, 0, 1)
    img = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth <= 0] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SimSource:
    """The simulator's GET /snapshot (sim/run_world.py), fetched back to back
    in a thread: every stream of one simulation step plus the true poses.

    latest() -> (seq, picture of what the current `mode` feeds the mapper),
    for the live panel; snapshot() -> (seq, dict of arrays) for the mapper.
    """

    is_file = False

    def __init__(self, base_url: str, mode: str = MODE_RGB, max_hz: float = 6.0):
        self.max_hz = max_hz
        self.base = base_url.rstrip("/")
        for suffix in ("/video_feed", "/snapshot"):
            if self.base.endswith(suffix):
                self.base = self.base[: -len(suffix)]
        self.uri = self.base + "/snapshot"
        self.mode = mode
        self._lock = threading.Lock()
        self._snap: dict | None = None
        self._seq = 0
        self._preview: tuple[int, str, np.ndarray] | None = None
        self.ended = False
        self.error: str | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> tuple[int, dict | None]:
        with self._lock:
            return self._seq, self._snap

    def latest(self) -> tuple[int, np.ndarray | None]:
        seq, snap = self.snapshot()
        if snap is None:
            return seq, None
        mode = self.mode
        cached = self._preview
        if cached is not None and cached[0] == seq and cached[1] == mode:
            return seq, cached[2]
        img, sd = snap["rgb"], None
        if mode == MODE_D405:
            img, sd = snap["d405_rgb"], snap["d405_depth"]
        elif mode == MODE_BOTH:
            sd = d405_in_rgb(snap)
        if sd is not None:   # picture on top, the depth the mapper gets below it
            img = np.concatenate([img, depth_preview(sd, 1.2)], 0)
        self._preview = (seq, mode, img)
        return seq, img

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import requests

        sess = requests.Session()
        next_t = 0.0
        while not self._stop:
            # The mapper uses 3-6 frames/s: asking more often would only slow
            # the simulator down (each snapshot renders the D405 too).
            time.sleep(max(0.0, next_t - time.time()))
            next_t = time.time() + 1.0 / self.max_hz
            try:
                r = sess.get(self.uri, timeout=10)
                r.raise_for_status()
                z = np.load(io.BytesIO(r.content))
                snap = {k: z[k] for k in z.files}
                with self._lock:
                    self._snap = snap
                    self._seq += 1
                self.error = None
            except Exception as e:  # sim not up yet / restarted: retry
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(1.0)


class FrameSource:
    """Background reader exposing the latest RGB frame."""

    def __init__(self, uri: str, realtime: bool = True, rotate: int = 0, record: str | None = None):
        self.uri = uri
        self.rotate = rotate
        # Optional recording of everything the camera sends (960 px wide), to
        # replay a live session offline with `--source <file>`.
        self.record = record
        self._writer = None
        self.is_file = not (uri.isdigit() or "://" in uri)
        # A file is played at its native FPS when realtime, to mimic a live camera.
        self.realtime = realtime
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self.ended = False
        self.error: str | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public -----------------------------------------------------------
    def latest(self) -> tuple[int, np.ndarray | None]:
        """Return (sequence number, RGB frame). The number changes with each new frame."""
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)  # let the reader finish its last frame
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # -- internals --------------------------------------------------------
    def _publish(self, bgr: np.ndarray) -> None:
        if self.rotate:
            code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate]
            bgr = cv2.rotate(bgr, code)
        if self.record:
            f = min(1.0, 960 / bgr.shape[1])
            small = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else bgr
            if self._writer is None:
                # MJPEG/AVI stays readable even if the program is killed mid-way.
                self._writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                                               (small.shape[1], small.shape[0]))
            self._writer.write(small)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._frame = rgb
            self._seq += 1

    def _run(self) -> None:
        if "://" in self.uri and self.uri.startswith("http"):
            self._run_mjpeg()
        else:
            self._run_capture()

    def _run_capture(self) -> None:
        src = int(self.uri) if self.uri.isdigit() else self.uri
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error = f"cannot open {self.uri}"
            self.ended = True
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / fps if (self.is_file and self.realtime) else 0.0
        t_next = time.time()
        while not self._stop:
            ok, bgr = cap.read()
            if not ok:
                break
            self._publish(bgr)
            if period:
                t_next += period
                time.sleep(max(0.0, t_next - time.time()))
        cap.release()
        self.ended = True

    def _run_mjpeg(self) -> None:
        """Parse a multipart MJPEG HTTP stream by hand (lowest latency, auto-reconnect)."""
        import requests

        while not self._stop:
            try:
                with requests.get(self.uri, stream=True, timeout=5) as r:
                    r.raise_for_status()
                    self.error = None
                    buf = b""
                    for chunk in r.iter_content(chunk_size=16384):
                        if self._stop:
                            return
                        buf += chunk
                        # Keep only the last complete JPEG in the buffer.
                        end = buf.rfind(b"\xff\xd9")
                        if end < 0:
                            continue
                        start = buf.rfind(b"\xff\xd8", 0, end)
                        if start < 0:
                            buf = buf[end + 2:]
                            continue
                        jpg, buf = buf[start:end + 2], buf[end + 2:]
                        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            self._publish(bgr)
            except Exception as e:  # network hiccup: retry
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(1.0)
