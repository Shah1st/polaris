"""Serveur de capture de la D405, côté robot : il capture et envoie, rien de plus.

Tout le calcul se fait sur la VM (alignement profondeur -> couleur, filtres, conversion en
mètres, nuages de points). Ici : un fil possède le pipeline RealSense, couleur + profondeur
dans le même frameset, SANS alignement ni filtre ; seule la dernière capture est gardée (copie
des tableaux bruts) avec son heure time.monotonic() ; JPEG et PNG ne sont encodés que si la VM
les demande (et une seule fois par image).

    GET /info        nom, série, firmware, USB, fps, K profondeur et couleur, depth_scale_m
                     (mètres par unité brute : 0,1 mm sur la D405, lu sur l'appareil),
                     depth_to_color {R (3x3 par lignes), t (m)} : X_couleur = R X_prof + t
    GET /frame       .npz (np.savez) : color_jpg, depth_png (PNG 16 bits des unités BRUTES, dans
                     le repère de la caméra de PROFONDEUR, non aligné), depth_scale, K_color,
                     K_depth, R_dc, t_dc, age_s, seq. En-tête X-Age-Ms.
                     ?fresh=1 : attend (2 s au plus) une image capturée après la demande.
    GET /color.jpg   même image couleur, pour un coup d'oeil dans un navigateur
    GET /depth.png   même profondeur brute 16 bits
    GET /health      {"ok", "device", "age_s", "frames", "errors", "last_error", "mode", ...}
    GET /grid        the obstacles in front of the rover, from the depth alone (no model), for
                     the rover's local guard (rover/local_map.py): {"res": 0.05, "x0", "y0",
                     "cells": rows of -1 not seen / 0 floor seen free / 1 obstacle, "floor",
                     "plane": "calibrated"|"fitted", "seq", "age_s", "stats"}. Cell [i][j] starts
                     at (x0 + j*res, y0 + i*res), x forward, y left, origin on the floor under
                     the lens. Computed (--grid-hz, 10 a second) only while someone asks.
    POST /calibrate_floor   rover on a flat, empty floor: fit the floor plane in the current
                     picture and keep it (--floor-file) for every later grid; the mount is
                     rigid. 422 when no floor is found or its height is not --cam-height
                     (within --cam-height-tol); ?force=1 keeps it anyway. GET /calibration.

503 {"error": ...} sans caméra ou quand la dernière image a plus de --max-age s. Les âges sont
mesurés avec time.monotonic() : l'horloge du robot a environ 8 h d'écart, l'heure murale ne
sert à rien.

Sur le rover (pyrealsense2 installé avec pip --target dans ~/polaris-d405/lib) :
    PYTHONPATH=~/polaris-d405/lib python3 d405_server.py            # 127.0.0.1:8091
Depuis la VM, par un tunnel SSH :
    ssh -N -L 8091:127.0.0.1:8091 <rover>
    curl -s http://127.0.0.1:8091/health ; curl -s http://127.0.0.1:8091/info

N'ouvre jamais /dev/video*, le port série ni rien de rover/rover_server.py. Python 3.10.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Essayés dans cet ordre après le mode demandé : en USB 2 la D405 n'offre que certains modes.
FALLBACK_MODES = ((640, 480, 15), (848, 480, 15), (480, 270, 15), (424, 240, 15))
FRESH_WAIT_S = 2.0


def mode_list(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
    """Le mode demandé puis les modes de repli, sans doublon."""
    out: list[tuple[int, int, int]] = []
    for m in ((width, height, fps),) + FALLBACK_MODES:
        if m not in out:
            out.append(m)
    return out


def k_matrix(intr) -> list[list[float]]:
    return [[float(intr.fx), 0.0, float(intr.ppx)], [0.0, float(intr.fy), float(intr.ppy)], [0.0, 0.0, 1.0]]


def extrinsics_rt(ext) -> tuple[np.ndarray, np.ndarray]:
    """(R par lignes, t) de rs2_extrinsics, tels que X_dst = R X_src + t.

    librealsense range `rotation` par COLONNES (rs2_transform_point_to_point utilise
    rotation[0], [3], [6] pour la première coordonnée) : il faut transposer."""
    R = np.asarray(ext.rotation, np.float64).reshape(3, 3).T
    t = np.asarray(ext.translation, np.float64).reshape(3)
    return R, t


class Capture:
    """Le fil qui possède le pipeline, et la dernière capture.

    En cas d'erreur (caméra débranchée, plus d'image) : arrêt du pipeline, erreur notée,
    nouvel essai après `retry_s` s, puis une attente doublée à chaque nouvel échec sans image
    (2, 4, 8, 15 s au plus) : chaque essai énumère l'USB, 23 % d'un cœur du robot à un essai
    toutes les 2 s sans caméra branchée. Retour à `retry_s` dès qu'une image arrive. Le côté
    HTTP ne fait que lire l'état sous verrou."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15, serial: str | None = None,
                 retry_s: float = 2.0, timeout_ms: int = 2000, log=print, retry_max_s: float = 15.0):
        self.modes = mode_list(width, height, fps)
        self.serial = serial
        self.retry_s, self.timeout_ms = retry_s, timeout_ms
        self.retry_max_s = max(retry_max_s, retry_s)
        self.next_retry_s = retry_s        # attente avant le prochain essai, doublée à chaque échec
        self.log = log
        self._cond = threading.Condition()
        self._latest: dict | None = None   # {'seq', 't', 'color', 'depth', 'info'} (t : monotonic)
        self.info: dict | None = None      # /info du pipeline en cours, None sans caméra
        self.device = False
        self.seq = 0                       # ne repart jamais à 0 : la VM s'en sert pour dédoublonner
        self.errors = 0
        self.last_error: str | None = None
        self.cpu_capture_s = 0.0           # temps CPU du fil de capture (copies comprises)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='d405-capture')

    # -- état lu par le serveur HTTP -----------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def latest(self) -> dict | None:
        with self._cond:
            return self._latest

    def wait_newer(self, t_after: float, timeout: float) -> dict | None:
        """Première capture faite après `t_after` (monotonic), ou None après `timeout` s."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest['t'] <= t_after:
                left = deadline - time.monotonic()
                if left <= 0 or self._stop.is_set():
                    return None
                self._cond.wait(left)
            return self._latest

    # -- fil de capture --------------------------------------------------------------------
    def _fail(self, msg: str) -> None:
        with self._cond:
            self.device = False
            self.info = None
            self.errors += 1
            if msg != self.last_error:  # pas une ligne toutes les 2 s pour la même erreur
                self.log(f'[d405] {msg}', flush=True)
            self.last_error = msg
            self._cond.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            pipe = None
            seq0 = self.seq
            try:
                import pyrealsense2 as rs  # importé ici : --help et les tests marchent sans lui

                pipe, info = self._open(rs)
                with self._cond:
                    self.info, self.device, self.last_error = info, True, None
                self._print_info(info)
                self._loop(pipe, info)
            except ImportError as e:
                self._fail(f'pyrealsense2 introuvable ({e}) : PYTHONPATH=~/polaris-d405/lib ?')
            except Exception as e:  # débranchée, plus d'image, mode refusé partout...
                self._fail(f'{type(e).__name__}: {e}')
            finally:
                if pipe is not None:
                    try:
                        pipe.stop()
                    except Exception:
                        pass
            # Des images sont arrivées pendant cet essai : on la recherche vite ; sinon on espace.
            wait = self.retry_s if self.seq != seq0 else self.next_retry_s
            self.next_retry_s = min(2 * wait, self.retry_max_s)
            self._stop.wait(wait)

    def _open(self, rs):
        """Démarre le pipeline sur le premier mode accepté ; (pipeline, dict de /info)."""
        ctx = rs.context()
        devs = [d for d in ctx.query_devices()
                if self.serial is None or d.get_info(rs.camera_info.serial_number) == self.serial]
        if not devs:
            raise RuntimeError('aucune caméra RealSense détectée' + (f' (série {self.serial})' if self.serial else ''))
        refused = []
        for w, h, fps in self.modes:
            pipe = rs.pipeline(ctx)
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                profile = pipe.start(cfg)
            except RuntimeError as e:  # "Couldn't resolve requests" : mode non offert sur ce lien USB
                refused.append(f'{w}x{h}@{fps} ({e})')
                continue
            try:
                return pipe, self._describe(rs, profile, (w, h, fps), refused)
            except Exception:
                pipe.stop()
                raise
        raise RuntimeError('aucun mode accepté : ' + ', '.join(refused))

    @staticmethod
    def _describe(rs, profile, mode, refused) -> dict:
        dev = profile.get_device()

        def dev_info(key):
            try:
                return str(dev.get_info(key))
            except Exception:
                return None

        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        di, ci = dprof.get_intrinsics(), cprof.get_intrinsics()
        R, t = extrinsics_rt(dprof.get_extrinsics_to(cprof))
        w, h, fps = mode
        return {
            'name': dev_info(rs.camera_info.name),
            'serial': dev_info(rs.camera_info.serial_number),
            'firmware': dev_info(rs.camera_info.firmware_version),
            'usb': dev_info(rs.camera_info.usb_type_descriptor),
            'fps': int(fps),
            'mode': f'{w}x{h}@{fps}',
            'refused_modes': list(refused),
            # Distorsion donnée pour information : la VM l'ignore pour l'instant (sténopé).
            'depth': {'width': int(di.width), 'height': int(di.height), 'K': k_matrix(di),
                      'model': str(di.model), 'coeffs': [float(c) for c in di.coeffs]},
            'color': {'width': int(ci.width), 'height': int(ci.height), 'K': k_matrix(ci),
                      'model': str(ci.model), 'coeffs': [float(c) for c in ci.coeffs]},
            'depth_scale_m': float(dev.first_depth_sensor().get_depth_scale()),
            'depth_to_color': {'R': R.tolist(), 't': t.tolist()},
        }

    def _print_info(self, info: dict) -> None:
        R = info['depth_to_color']['R']
        self.log(f"[d405] {info['name']} série {info['serial']} firmware {info['firmware']} USB {info['usb']}\n"
                 f"[d405] mode {info['mode']}"
                 + (f" (refusés : {', '.join(info['refused_modes'])})" if info['refused_modes'] else '')
                 + f"\n[d405] depth_scale {info['depth_scale_m']:g} m/unité\n"
                 f"[d405] profondeur -> couleur : t = {info['depth_to_color']['t']} m, "
                 f"R = {[[round(v, 5) for v in row] for row in R]}", flush=True)

    def _loop(self, pipe, info: dict) -> None:
        while not self._stop.is_set():
            frames = pipe.wait_for_frames(self.timeout_ms)  # RuntimeError si rien : on relance tout
            t = time.monotonic()
            c0 = time.thread_time()
            d, c = frames.get_depth_frame(), frames.get_color_frame()
            if not d or not c:
                continue
            # Copies tout de suite : librealsense n'a qu'un petit stock de trames, en garder une
            # bloquerait la capture.
            depth = np.asanyarray(d.get_data()).copy()
            color = np.asanyarray(c.get_data()).copy()
            del frames, d, c
            with self._cond:
                self.seq += 1
                # Les infos du pipeline qui a pris l'image : après une reconnexion le mode peut changer.
                self._latest = {'seq': self.seq, 't': t, 'color': color, 'depth': depth, 'info': info}
                self.next_retry_s = self.retry_s
                self.cpu_capture_s += time.thread_time() - c0
                self._cond.notify_all()


class Encoder:
    """JPEG / PNG / npz d'une capture, encodés à la demande et une seule fois par image."""

    def __init__(self, jpeg_quality: int = 85):
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._seq = -1
        self._cache: dict[str, bytes] = {}

    def get(self, snap: dict, kind: str) -> bytes:
        with self._lock:  # deux requêtes sur la même image : un seul encodage
            if self._seq != snap['seq']:
                self._seq, self._cache = snap['seq'], {}
            out = self._cache.get(kind)
            if out is None:
                if kind == 'jpg':
                    out = cv2.imencode('.jpg', snap['color'], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1].tobytes()
                else:  # PNG 16 bits, compression rapide : le CPU du robot compte plus que le débit
                    out = cv2.imencode('.png', snap['depth'], [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
                self._cache[kind] = out
            return out

    def npz(self, snap: dict, age_s: float) -> bytes:
        info = snap['info']
        dc = info['depth_to_color']
        buf = io.BytesIO()
        np.savez(buf,
                 color_jpg=np.frombuffer(self.get(snap, 'jpg'), np.uint8),
                 depth_png=np.frombuffer(self.get(snap, 'png'), np.uint8),
                 depth_scale=np.float64(info['depth_scale_m']),
                 K_color=np.asarray(info['color']['K'], np.float64),
                 K_depth=np.asarray(info['depth']['K'], np.float64),
                 R_dc=np.asarray(dc['R'], np.float64), t_dc=np.asarray(dc['t'], np.float64),
                 age_s=np.float64(age_s), seq=np.int64(snap['seq']))
        return buf.getvalue()


class GridMaker:
    """The latest depth picture as a grid of 5 cm cells in front of the rover (GET /grid).

    Same geometry as the perception service's D405 mode (rovermap/perception.py, numpy and
    OpenCV only): points 4 to 40 cm above the floor are obstacles, the floor seen is free,
    what an obstacle hides is not seen. The floor plane comes from POST /calibrate_floor when
    there is one (the mount is rigid, and a wall filling the picture defeats a fit); otherwise
    it is fitted in each picture.

    A thread of its own, at most `hz` grids a second, and only while GET /grid is asked.
    """

    IDLE = 3.0  # s without a request before the thread rests

    def __init__(self, cap: Capture, hz: float = 10.0, cam_height: float = 0.15,
                 height_tol: float = 0.03, floor_file: str | None = None, stride: int = 2,
                 max_range: float = 1.25, band: tuple[float, float] = (0.04, 0.40),
                 depth_min: float = 0.07, depth_max: float = 3.0, log=print):
        self.cap, self.hz = cap, hz
        self.cam_height, self.height_tol = cam_height, height_tol
        self.floor_file = floor_file
        self.stride, self.max_range, self.band = max(1, int(stride)), max_range, band
        self.depth_min, self.depth_max = depth_min, depth_max
        self.log = log
        self.plane = None                  # (normal, distance in m), camera frame
        self.calibration: dict | None = None
        self.error: str | None = None
        self.grids = 0
        self.cpu_s = 0.0
        self._cond = threading.Condition()
        self._latest: dict | None = None   # {'seq', 't' (capture, monotonic), 'grid'}
        self._wanted = float('-inf')
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='d405-grid')
        self._load()

    # -- floor plane ---------------------------------------------------------------------
    def _load(self) -> None:
        if not self.floor_file or not os.path.exists(self.floor_file):
            return
        try:
            with open(self.floor_file, encoding='utf-8') as f:
                c = json.load(f)
            self.plane = (np.asarray(c['normal'], np.float64), float(c['d']))
            self.calibration = c
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log(f'[d405] floor file {self.floor_file} unreadable ({e}): fitting each picture',
                     flush=True)

    def depth_m(self, snap: dict, stride: int = 1):
        """(depth in metres with out-of-range pixels at 0, K for that stride)."""
        info = snap['info']
        d = snap['depth'][::stride, ::stride].astype(np.float32) * np.float32(info['depth_scale_m'])
        d[(d < self.depth_min) | (d > self.depth_max)] = 0
        K = np.asarray(info['depth']['K'], np.float64).copy()
        K[:2] /= stride
        return d, K

    def calibrate(self, snap: dict, force: bool = False) -> dict:
        """Fit the floor in `snap` and keep it. Raises ValueError when there is no floor or
        its height is not the mount's."""
        from rovermap.perception import fit_floor, valid_depth

        depth, K = self.depth_m(snap)
        plane = fit_floor(depth, K, valid_depth(depth))
        if plane is None:
            raise ValueError('no floor found in the picture (is the rover on a flat, empty floor?)')
        n, d = np.asarray(plane[0], np.float64), float(plane[1])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(n[1])))))
        if abs(d - self.cam_height) > self.height_tol and not force:
            raise ValueError(f'the floor is {d:.3f} m under the lens, expected {self.cam_height:.3f} m '
                             f'(+-{self.height_tol:g}): wrong surface, or pass --cam-height / ?force=1')
        c = {'normal': [float(v) for v in n], 'd': d, 'height_m': round(d, 4),
             'tilt_deg': round(tilt, 2), 'seq': int(snap['seq'])}
        if self.floor_file:
            tmp = self.floor_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(c, f, indent=2)
            os.replace(tmp, self.floor_file)
        self.plane, self.calibration = (n, d), c
        return c

    # -- grid ----------------------------------------------------------------------------
    def compute(self, snap: dict) -> dict:
        from rovermap.perception import perceive

        depth, K = self.depth_m(snap, self.stride)
        per = perceive(depth, K, cam_height=self.cam_height, plane=self.plane, scale=1.0,
                       max_range=self.max_range, band=self.band)
        out = per.to_json()
        out['plane'] = 'calibrated' if self.plane is not None else ('fitted' if per.floor else None)
        out['seq'] = int(snap['seq'])
        return out

    def start(self) -> None:
        if self.hz > 0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        last_t = float('-inf')
        while not self._stop.is_set():
            if time.monotonic() - self._wanted > self.IDLE:
                self._stop.wait(0.05)
                continue
            snap = self.cap.wait_newer(last_t, 0.5)
            if snap is None:
                continue
            last_t = snap['t']
            c0 = time.thread_time()
            try:
                grid = self.compute(snap)
                self.error = None
            except Exception as e:  # a bad picture or a missing module: /grid says why
                self.error, grid = f'{type(e).__name__}: {e}', None
            with self._cond:
                if grid is not None:
                    self._latest = {'seq': snap['seq'], 't': snap['t'], 'grid': grid}
                    self.grids += 1
                    self.cpu_s += time.thread_time() - c0
                self._cond.notify_all()
            self._stop.wait(max(0.0, 1.0 / self.hz - (time.monotonic() - last_t)))

    def get(self, max_age: float, wait: float = 1.0) -> dict | None:
        """The newest grid, its picture at most `max_age` s old (waits `wait` s for the thread
        to wake up); None when there is none."""
        self._wanted = time.monotonic()
        deadline = time.monotonic() + wait
        with self._cond:
            while self._latest is None or time.monotonic() - self._latest['t'] > max_age:
                left = deadline - time.monotonic()
                if left <= 0 or self.hz <= 0:
                    return None
                self._cond.wait(left)
            return self._latest


def make_handler(cap: Capture, enc: Encoder, max_age: float, grid: GridMaker | None = None):
    stats_lock = threading.Lock()
    cpu: dict[str, list[float]] = {}  # route -> [requêtes, secondes CPU]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # une ligne par requête à 3-6 Hz : du bruit
            pass

        def _send(self, code: int, body: bytes, ctype: str, age: float | None = None) -> None:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if age is not None:
                self.send_header('X-Age-Ms', str(int(round(age * 1000))))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), 'application/json')

        def _snap(self, fresh: bool):
            """Capture à servir, ou None après avoir répondu 503."""
            if not cap.device:  # tout de suite, sans attendre 2 s une image qui ne viendra pas
                self._json({'error': 'pas de D405', 'last_error': cap.last_error}, 503)
                return None
            if fresh:
                snap = cap.wait_newer(time.monotonic(), FRESH_WAIT_S)
                if snap is None:
                    self._json({'error': f'aucune image capturée dans les {FRESH_WAIT_S:g} s après la demande',
                                'last_error': cap.last_error}, 503)
                    return None
            else:
                snap = cap.latest()
            if snap is None:
                self._json({'error': 'aucune image encore', 'last_error': cap.last_error}, 503)
                return None
            age = time.monotonic() - snap['t']
            if age > max_age:
                self._json({'error': f'dernière image trop ancienne ({age:.2f} s > {max_age:g} s)',
                            'last_error': cap.last_error}, 503)
                return None
            return snap

        def _health(self) -> dict:
            snap, info = cap.latest(), cap.info
            age = None if snap is None else round(time.monotonic() - snap['t'], 3)
            with stats_lock:
                cpu_ms = {k: round(1000 * v[1] / v[0], 2) for k, v in cpu.items() if v[0]}
            return {
                'ok': bool(cap.device and age is not None and age <= max_age),
                'device': bool(cap.device), 'age_s': age, 'frames': cap.seq, 'errors': cap.errors,
                'last_error': cap.last_error, 'mode': None if info is None else info['mode'],
                'usb': None if info is None else info['usb'],
                'retry_s': cap.next_retry_s,  # attente du prochain essai si la caméra manque
                # Temps CPU moyen par requête (ms) et par image capturée : à surveiller sur le robot.
                'cpu_ms_per_request': cpu_ms,
                'cpu_ms_per_capture': round(1000 * cap.cpu_capture_s / cap.seq, 3) if cap.seq else None,
                # GET /grid: its CPU time per grid is the number to watch next to the rover server.
                'grid': None if grid is None else {
                    'hz': grid.hz, 'grids': grid.grids, 'error': grid.error,
                    'floor': 'calibrated' if grid.plane is not None else 'fitted in each picture',
                    'cpu_ms_per_grid': round(1000 * grid.cpu_s / grid.grids, 2) if grid.grids else None,
                },
            }

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            fresh = parse_qs(url.query).get('fresh', ['0'])[0] not in ('0', '', 'false')
            c0 = time.thread_time()  # chaque requête a son fil : temps CPU de CETTE requête
            try:
                if url.path == '/frame':
                    snap = self._snap(fresh)
                    if snap:
                        enc.get(snap, 'jpg'), enc.get(snap, 'png')  # encodés avant de mesurer l'âge
                        age = time.monotonic() - snap['t']
                        self._send(200, enc.npz(snap, age), 'application/x-npz', age)
                elif url.path in ('/color.jpg', '/depth.png'):
                    snap = self._snap(fresh)
                    if snap:
                        jpg = url.path == '/color.jpg'
                        body = enc.get(snap, 'jpg' if jpg else 'png')
                        self._send(200, body, 'image/jpeg' if jpg else 'image/png', time.monotonic() - snap['t'])
                elif url.path == '/info':
                    info = cap.info
                    if info is None:
                        self._json({'error': 'pas de D405', 'last_error': cap.last_error}, 503)
                    else:
                        self._json(info)
                elif url.path == '/health':
                    self._json(self._health())
                elif url.path == '/grid':
                    self._grid()
                elif url.path == '/calibration':
                    self._json({'calibration': None if grid is None else grid.calibration})
                else:
                    self._json({'error': 'introuvable'}, 404)
                    return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # le serveur continue ; la VM traite l'erreur comme « pas d'image »
                try:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 500)
                except Exception:
                    pass
                return
            with stats_lock:
                s = cpu.setdefault(url.path, [0, 0.0])
                s[0] += 1
                s[1] += time.thread_time() - c0

        def _grid(self) -> None:
            if grid is None or grid.hz <= 0:
                self._json({'error': 'no grid (--grid-hz 0)'}, 503)
                return
            if not cap.device:
                self._json({'error': 'pas de D405', 'last_error': cap.last_error}, 503)
                return
            latest = grid.get(max_age)
            if latest is None:
                self._json({'error': grid.error or 'no grid from a recent picture'}, 503)
                return
            age = time.monotonic() - latest['t']
            body = json.dumps({**latest['grid'], 'age_s': round(age, 3)}).encode()
            self._send(200, body, 'application/json', age)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            n = int(self.headers.get('Content-Length') or 0)
            if n:
                self.rfile.read(n)
            try:
                if url.path != '/calibrate_floor' or grid is None:
                    self._json({'error': 'introuvable'}, 404)
                    return
                snap = self._snap(True)
                if snap is None:
                    return
                force = parse_qs(url.query).get('force', ['0'])[0] not in ('0', '', 'false')
                try:
                    self._json({'ok': True, 'calibration': grid.calibrate(snap, force)})
                except ValueError as e:
                    self._json({'error': str(e)}, 422)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                try:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 500)
                except Exception:
                    pass

    return Handler


def make_server(host: str, port: int, cap: Capture, max_age: float = 2.0, jpeg_quality: int = 85,
                grid: GridMaker | None = None):
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((host, port), make_handler(cap, Encoder(jpeg_quality), max_age, grid))
    srv.daemon_threads = True
    return srv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1', help='adresse d\'écoute (127.0.0.1 : tunnel SSH depuis la VM)')
    ap.add_argument('--port', type=int, default=8091)
    ap.add_argument('--width', type=int, default=640, help='largeur demandée (couleur et profondeur)')
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--max-age', type=float, default=2.0, help='503 si la dernière image a plus de N s')
    ap.add_argument('--jpeg-quality', type=int, default=85)
    ap.add_argument('--serial', default=None, help='numéro de série si plusieurs RealSense sont branchées')
    ap.add_argument('--grid-hz', type=float, default=10.0,
                    help='GET /grid: obstacle grids a second at most, computed only while asked (0: off)')
    ap.add_argument('--cam-height', type=float, default=0.15, help='lens centre above the floor, m')
    ap.add_argument('--cam-height-tol', type=float, default=0.03,
                    help='POST /calibrate_floor refuses a floor farther than this from --cam-height')
    ap.add_argument('--floor-file',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'd405_floor.json'),
                    help='where POST /calibrate_floor keeps the floor plane')
    ap.add_argument('--grid-stride', type=int, default=2, help='GET /grid: one depth pixel in N')
    ap.add_argument('--grid-range', type=float, default=1.25, help='GET /grid: metres ahead')
    args = ap.parse_args(argv)

    cap = Capture(args.width, args.height, args.fps, serial=args.serial)
    grid = GridMaker(cap, args.grid_hz, args.cam_height, args.cam_height_tol, args.floor_file,
                     args.grid_stride, args.grid_range)
    srv = make_server(args.host, args.port, cap, args.max_age, args.jpeg_quality, grid)

    def on_term(*_):  # SIGTERM (systemd, kill) : même sortie propre que Ctrl+C
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    cap.start()
    grid.start()
    print(f'D405 : http://{args.host}:{args.port}/health  (modes essayés : '
          + ', '.join(f'{w}x{h}@{f}' for w, h, f in cap.modes) + ')', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        grid.stop()
        cap.stop()


if __name__ == '__main__':
    main()
