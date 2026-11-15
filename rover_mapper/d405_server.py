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
    nouvel essai toutes les `retry_s` s. Le côté HTTP ne fait que lire l'état sous verrou."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15, serial: str | None = None,
                 retry_s: float = 2.0, timeout_ms: int = 2000, log=print):
        self.modes = mode_list(width, height, fps)
        self.serial = serial
        self.retry_s, self.timeout_ms = retry_s, timeout_ms
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
            self._stop.wait(self.retry_s)

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


def make_handler(cap: Capture, enc: Encoder, max_age: float):
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
                # Temps CPU moyen par requête (ms) et par image capturée : à surveiller sur le robot.
                'cpu_ms_per_request': cpu_ms,
                'cpu_ms_per_capture': round(1000 * cap.cpu_capture_s / cap.seq, 3) if cap.seq else None,
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

    return Handler


def make_server(host: str, port: int, cap: Capture, max_age: float = 2.0, jpeg_quality: int = 85):
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((host, port), make_handler(cap, Encoder(jpeg_quality), max_age))
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
    args = ap.parse_args(argv)

    cap = Capture(args.width, args.height, args.fps, serial=args.serial)
    srv = make_server(args.host, args.port, cap, args.max_age, args.jpeg_quality)

    def on_term(*_):  # SIGTERM (systemd, kill) : même sortie propre que Ctrl+C
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    cap.start()
    print(f'D405 : http://{args.host}:{args.port}/health  (modes essayés : '
          + ', '.join(f'{w}x{h}@{f}' for w, h, f in cap.modes) + ')', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        cap.stop()


if __name__ == '__main__':
    main()
