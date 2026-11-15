"""d405_server.py sans la caméra : un faux module pyrealsense2 (mis dans sys.modules) imite ce que
le serveur utilise (context, pipeline, config, frames, profils, intrinsèques, extrinsèques,
depth_scale). Serveurs de test sur 127.0.0.1:8240-8244.

Run: CUDA_VISIBLE_DEVICES= python -m pytest rover_mapper/test_d405_server.py -q -p no:cacheprovider \\
         --rootdir=rover_mapper
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d405_server as ds  # noqa: E402

PORTS = range(8240, 8245)


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


# Non symétrique : un oubli de transposition des extrinsèques ferait échouer /info.
R_TRUE = rot(2, 0.10) @ rot(0, 0.05)
T_TRUE = np.array([0.0151, -0.0023, 0.0007])
SCALE = 1e-4  # D405 : 0,1 mm par unité brute


def hashed_depth(n, w, h):
    """Toute la plage 16 bits : un passage par 8 bits ou une perte se verrait."""
    x = np.arange(h * w, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(n * 7919)
    return (x % np.uint64(65536)).astype(np.uint16).reshape(h, w)


def smooth_depth(n, w, h):
    """Profondeur d'allure réelle (sol incliné + bruit de quelques unités) pour la mesure de CPU."""
    rng = np.random.default_rng(n)
    v = np.linspace(2000, 9000, h)[:, None] + np.linspace(0, 500, w)[None, :]
    d = v + rng.normal(0, 15, (h, w))
    d[rng.random((h, w)) < 0.05] = 0  # trous
    return d.astype(np.uint16)


def gradient_color(n, w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, w)[None, :]
    img[..., 1] = np.linspace(0, 255, h)[:, None]
    img[..., 2] = (n * 10) % 256
    return img


class World:
    """L'état de la fausse caméra, modifié par les tests pendant que le serveur tourne."""

    def __init__(self):
        self.plugged = True
        self.paused = False         # bloque sans erreur : la caméra est là mais n'envoie plus rien
        self.offered = {(640, 480, 15), (848, 480, 15), (480, 270, 15), (424, 240, 15)}
        self.rate = 60.0            # images/s de la fausse caméra (plus vite que la vraie : tests courts)
        self.depth_fn = hashed_depth
        self.n = 0
        self.history = {}
        self.starts, self.refused, self.stops = [], [], 0
        self.lock = threading.Lock()


def intrinsics(kind, w, h):
    # Profondeur et couleur volontairement différentes, pour vérifier qu'elles ne sont pas mélangées.
    f = 0.60 * w if kind == 'depth' else 0.62 * w
    return SimpleNamespace(width=w, height=h, fx=f, fy=f * 1.001, ppx=w / 2 - 1.5 if kind == 'depth' else w / 2 + 2.0,
                           ppy=h / 2 + 0.5, model='distortion.brown_conrady', coeffs=[0.0] * 5)


def make_fake_rs(world):
    rs = types.ModuleType('pyrealsense2')
    rs.stream = SimpleNamespace(depth='depth', color='color')
    rs.format = SimpleNamespace(z16='z16', bgr8='bgr8')
    rs.camera_info = SimpleNamespace(name='name', serial_number='serial_number',
                                     firmware_version='firmware_version', usb_type_descriptor='usb_type_descriptor')
    infos = {'name': 'Intel RealSense D405', 'serial_number': '123456789', 'firmware_version': '5.16.0.1',
             'usb_type_descriptor': '3.2'}

    class Device:
        def get_info(self, key):
            return infos[key]

        def first_depth_sensor(self):
            return SimpleNamespace(get_depth_scale=lambda: SCALE)

    class StreamProfile:
        def __init__(self, kind, w, h, fps):
            self.kind, self.w, self.h, self._fps = kind, w, h, fps

        def as_video_stream_profile(self):
            return self

        def get_intrinsics(self):
            return intrinsics(self.kind, self.w, self.h)

        def fps(self):
            return self._fps

        def get_extrinsics_to(self, other):
            assert (self.kind, other.kind) == ('depth', 'color')
            # librealsense : rotation rangée par COLONNES.
            return SimpleNamespace(rotation=list(R_TRUE.T.reshape(-1)), translation=list(T_TRUE))

    class Frame:
        def __init__(self, data):
            self.data = data

        def get_data(self):
            return self.data

    class Frameset:
        def __init__(self, depth, color):
            self.d, self.c = Frame(depth), Frame(color)

        def get_depth_frame(self):
            return self.d

        def get_color_frame(self):
            return self.c

    class context:  # noqa: N801
        def query_devices(self):
            return [Device()] if world.plugged else []

    class config:  # noqa: N801
        def __init__(self):
            self.streams = {}

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, stream, w, h, fmt, fps):
            self.streams[stream] = (w, h, fmt, fps)

    class pipeline:  # noqa: N801
        def __init__(self, ctx=None):
            self.mode = None

        def start(self, cfg):
            if not world.plugged:
                raise RuntimeError('No device connected')
            (w, h, fz, fps), (w2, h2, fc, fps2) = cfg.streams['depth'], cfg.streams['color']
            assert (fz, fc) == ('z16', 'bgr8') and (w, h, fps) == (w2, h2, fps2)
            if (w, h, fps) not in world.offered:
                world.refused.append((w, h, fps))
                raise RuntimeError("Couldn't resolve requests")
            world.starts.append((w, h, fps))
            self.mode = (w, h, fps)
            return SimpleNamespace(get_device=lambda: Device(),
                                   get_stream=lambda s: StreamProfile(s, w, h, fps))

        def wait_for_frames(self, timeout_ms=5000):
            deadline = time.monotonic() + timeout_ms / 1000
            while world.paused or not world.plugged:
                if time.monotonic() > deadline or not world.plugged:
                    time.sleep(0.02)
                    raise RuntimeError(f"Frame didn't arrive within {timeout_ms}")
                time.sleep(0.01)
            time.sleep(1.0 / world.rate)
            w, h, _ = self.mode
            with world.lock:
                world.n += 1
                n = world.n
                depth, color = world.depth_fn(n, w, h), gradient_color(n, w, h)
                world.history[n] = (depth, color)
                world.history.pop(n - 200, None)
            return Frameset(depth, color)

        def stop(self):
            world.stops += 1

    rs.context, rs.config, rs.pipeline = context, config, pipeline
    return rs


@pytest.fixture
def world(monkeypatch):
    w = World()
    monkeypatch.setitem(sys.modules, 'pyrealsense2', make_fake_rs(w))
    return w


@pytest.fixture
def serve():
    """serve(max_age=..., timeout_ms=..., ...) -> base URL ; tout est arrêté à la fin du test."""
    started = []

    def start(max_age=1.0, timeout_ms=300, retry_s=0.1, **kw):
        logs = []
        cap = ds.Capture(retry_s=retry_s, timeout_ms=timeout_ms, log=lambda *a, **k: logs.append(' '.join(a)), **kw)
        srv = None
        for port in PORTS:
            try:
                srv = ds.make_server('127.0.0.1', port, cap, max_age=max_age)
                break
            except OSError:
                continue
        assert srv is not None, 'aucun port libre dans 8240-8244'
        cap.start()
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append((srv, cap))
        base = 'http://127.0.0.1:%d' % srv.server_address[1]
        base_ns = SimpleNamespace(url=base, cap=cap, logs=logs)
        return base_ns

    yield start
    for srv, cap in started:
        srv.shutdown()
        srv.server_close()
        cap.stop()


def get(url, timeout=5):
    """(code, en-têtes, corps) sans lever d'exception sur 503."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def get_json(url):
    code, _, body = get(url)
    return code, json.loads(body)


def wait_for(pred, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def health(s):
    return get_json(s.url + '/health')[1]


def ready(s):
    assert wait_for(lambda: health(s)['ok']), health(s)


def load_frame(s, query=''):
    code, headers, body = get(s.url + '/frame' + query)
    assert code == 200, body
    assert headers['Content-Type'] == 'application/x-npz'
    z = np.load(io.BytesIO(body))
    return headers, {k: z[k] for k in z.files}


# -- tests ------------------------------------------------------------------------------------


def test_mode_list_and_extrinsics_layout():
    assert ds.mode_list(640, 480, 15) == [(640, 480, 15), (848, 480, 15), (480, 270, 15), (424, 240, 15)]
    assert ds.mode_list(1280, 720, 30)[:2] == [(1280, 720, 30), (640, 480, 15)]
    R, t = ds.extrinsics_rt(SimpleNamespace(rotation=list(R_TRUE.T.reshape(-1)), translation=list(T_TRUE)))
    assert np.allclose(R, R_TRUE) and np.allclose(t, T_TRUE)


def test_info(world, serve):
    s = serve()
    ready(s)
    code, info = get_json(s.url + '/info')
    assert code == 200
    assert info['name'] == 'Intel RealSense D405' and info['serial'] == '123456789'
    assert info['firmware'] == '5.16.0.1' and info['usb'] == '3.2' and info['fps'] == 15
    assert info['depth_scale_m'] == SCALE
    for kind in ('depth', 'color'):
        exp = intrinsics(kind, 640, 480)
        assert (info[kind]['width'], info[kind]['height']) == (640, 480)
        assert np.allclose(info[kind]['K'], [[exp.fx, 0, exp.ppx], [0, exp.fy, exp.ppy], [0, 0, 1]])
    assert np.allclose(info['depth_to_color']['R'], R_TRUE)  # par lignes : X_c = R X_d + t
    assert np.allclose(info['depth_to_color']['t'], T_TRUE)
    log = '\n'.join(s.logs)
    assert '640x480@15' in log and 'USB 3.2' in log and 'depth_scale 0.0001' in log and '5.16.0.1' in log


def test_frame_roundtrip(world, serve):
    s = serve()
    ready(s)
    headers, z = load_frame(s)
    assert set(z) == {'color_jpg', 'depth_png', 'depth_scale', 'K_color', 'K_depth', 'R_dc', 't_dc', 'age_s', 'seq'}
    assert z['color_jpg'].dtype == np.uint8 and z['color_jpg'].ndim == 1
    assert z['depth_png'].dtype == np.uint8 and z['depth_png'].ndim == 1
    assert z['depth_scale'].dtype == np.float64 and float(z['depth_scale']) == SCALE
    assert z['seq'].dtype == np.int64 and z['age_s'].dtype == np.float64
    for k in ('K_color', 'K_depth', 'R_dc'):
        assert z[k].dtype == np.float64 and z[k].shape == (3, 3)
    assert z['t_dc'].shape == (3,)
    assert np.allclose(z['R_dc'], R_TRUE) and np.allclose(z['t_dc'], T_TRUE)
    exp_d, exp_c = intrinsics('depth', 640, 480), intrinsics('color', 640, 480)
    assert np.isclose(z['K_depth'][0, 2], exp_d.ppx) and np.isclose(z['K_color'][0, 2], exp_c.ppx)
    assert 0 <= float(z['age_s']) < 1.0
    assert abs(int(headers['X-Age-Ms']) - 1000 * float(z['age_s'])) < 50

    depth_exp, color_exp = world.history[int(z['seq'])]  # même frameset pour les deux
    depth = cv2.imdecode(z['depth_png'], cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and np.array_equal(depth, depth_exp)  # exact, unités brutes
    assert depth_exp.max() > 60000  # la plage 16 bits complète a bien été testée
    color = cv2.imdecode(z['color_jpg'], cv2.IMREAD_COLOR)
    assert color.shape == (480, 640, 3)
    assert np.abs(color.astype(int) - color_exp.astype(int)).mean() < 3  # BGR, JPEG

    # Coup d'oeil navigateur : même contenu.
    code, h, body = get(s.url + '/depth.png')
    assert code == 200 and h['Content-Type'] == 'image/png' and 'X-Age-Ms' in h
    d2 = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    assert d2.dtype == np.uint16 and d2.shape == (480, 640)
    code, h, body = get(s.url + '/color.jpg')
    assert code == 200 and h['Content-Type'] == 'image/jpeg' and 'X-Age-Ms' in h
    assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)


def test_stale_frame_is_503(world, serve):
    # Attente de trame plus longue que --max-age : la caméra reste « là » mais l'image vieillit.
    s = serve(max_age=0.3, timeout_ms=5000)
    ready(s)
    world.paused = True
    time.sleep(0.6)
    for path in ('/frame', '/color.jpg', '/depth.png'):
        code, _, body = get(s.url + path)
        assert code == 503 and 'trop ancienne' in json.loads(body)['error']
    h = health(s)
    assert h['device'] is True and h['ok'] is False and h['age_s'] > 0.3
    world.paused = False
    ready(s)
    assert load_frame(s)[1]['age_s'] < 0.3


def test_fresh_waits_for_a_new_capture(world, serve, monkeypatch):
    world.rate = 8.0  # une image toutes les 125 ms : la dernière a en général déjà de l'âge
    s = serve()
    ready(s)
    for _ in range(3):
        before = health(s)['frames']
        _, z = load_frame(s, '?fresh=1')
        assert int(z['seq']) > before
    # Plus rien n'arrive : ?fresh=1 abandonne après le délai (raccourci ici) avec un 503.
    monkeypatch.setattr(ds, 'FRESH_WAIT_S', 0.3)
    s2 = serve(max_age=5.0, timeout_ms=5000)
    ready(s2)
    world.paused = True
    time.sleep(0.2)
    t0 = time.monotonic()
    code, _, body = get(s2.url + '/frame?fresh=1')
    assert code == 503 and 'après la demande' in json.loads(body)['error']
    assert 0.25 < time.monotonic() - t0 < 2.0
    assert get(s2.url + '/frame')[0] == 200  # sans fresh : l'image de 0,2 s reste servie
    world.paused = False


def test_mode_fallback(world, serve):
    world.offered = {(480, 270, 15)}  # lien USB 2 : seuls certains modes passent
    s = serve()
    ready(s)
    assert world.refused[:2] == [(640, 480, 15), (848, 480, 15)] and world.starts == [(480, 270, 15)]
    code, info = get_json(s.url + '/info')
    assert info['fps'] == 15 and (info['depth']['width'], info['depth']['height']) == (480, 270)
    assert health(s)['mode'] == '480x270@15'
    _, z = load_frame(s)
    assert cv2.imdecode(z['depth_png'], cv2.IMREAD_UNCHANGED).shape == (270, 480)
    assert np.isclose(z['K_depth'][0, 2], 480 / 2 - 1.5)

    world.offered = set()
    s2 = serve(width=1280, height=720, fps=30)
    assert wait_for(lambda: health(s2)['errors'] > 0)
    h = health(s2)
    assert h['device'] is False and 'aucun mode accepté' in h['last_error'] and '1280x720@30' in h['last_error']


def test_recovery_after_unplug(world, serve):
    s = serve()
    ready(s)
    seq0 = int(load_frame(s)[1]['seq'])
    world.plugged = False
    assert wait_for(lambda: health(s)['device'] is False)
    h = health(s)
    assert h['ok'] is False and h['errors'] >= 1 and h['last_error'] and h['mode'] is None
    for path in ('/frame', '/frame?fresh=1', '/info'):
        code, _, body = get(s.url + path)
        assert code == 503 and 'D405' in json.loads(body)['error']
    assert world.stops >= 1  # le pipeline a été arrêté proprement
    time.sleep(0.3)  # plusieurs essais pendant l'absence : le serveur HTTP tient
    assert health(s)['device'] is False
    world.plugged = True
    ready(s)
    _, z = load_frame(s)
    assert int(z['seq']) > seq0  # la numérotation ne repart pas de zéro après reconnexion
    assert len(world.starts) >= 2


def test_without_pyrealsense2(monkeypatch, serve):
    monkeypatch.setitem(sys.modules, 'pyrealsense2', None)  # import -> ImportError
    s = serve()
    assert wait_for(lambda: health(s)['errors'] > 0)
    h = health(s)
    assert h['device'] is False and h['ok'] is False and 'pyrealsense2' in h['last_error']
    assert h['age_s'] is None and h['frames'] == 0 and h['mode'] is None
    assert get(s.url + '/frame')[0] == 503 and get(s.url + '/info')[0] == 503
    assert get(s.url + '/nope')[0] == 404


def test_help_without_pyrealsense2():
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, 'd405_server.py'), '--help'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and '--max-age' in r.stdout and 'D405' in r.stdout
    r = subprocess.run([sys.executable, '-c', 'import sys, d405_server; print("pyrealsense2" in sys.modules)'],
                       cwd=here, capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == 'False', r.stderr  # importé seulement dans le fil de capture


def test_cpu_per_request(world, serve):
    """Temps CPU par requête mesuré par le serveur lui-même (640x480, profondeur d'allure réelle).
    Le chiffre de cette machine est affiché ; celui du robot se lit sur /health."""
    world.depth_fn = smooth_depth
    world.rate = 30.0
    s = serve()
    ready(s)
    sizes = []
    for _ in range(15):
        code, _, body = get(s.url + '/frame?fresh=1')  # une nouvelle image à chaque fois : encodage réel
        assert code == 200
        sizes.append(len(body))
    for _ in range(5):
        get(s.url + '/frame')
    h = health(s)
    cpu = h['cpu_ms_per_request']
    print(f"\nCPU par requête (ms) : {cpu} ; par image capturée : {h['cpu_ms_per_capture']} ms ; "
          f"taille /frame : {np.median(sizes) / 1024:.0f} Ko", flush=True)
    assert 0 < cpu['/frame'] < 250 and h['cpu_ms_per_capture'] is not None
