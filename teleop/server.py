"""Teleop dashboard for the real rover, served on the VM (loopback only).

    python server.py [--robot http://127.0.0.1:18080] [--port 8082]
                     [--drone http://<phone>:<port>/video]

  GET  /             the page: ZQSD driving and the rover camera on the left; on the right the
                     drone camera (--drone) and the rover, drone and fused maps of the live
                     fusion (/fusion/, below), or the live 3D map of the mapper
  GET  /drone_feed   the drone camera (--drone: a phone camera app's MJPEG or snapshot URL) as
                     MJPEG, re-served from ONE connection to the phone, for local readers
                     (merging_map/lancer_overhead.py --camera)
  GET  /video_feed   the rover camera as MJPEG, re-served from ONE upstream
                     connection (every extra reader of the robot's own
                     /video_feed loads the link). For local readers (the mapper).
  GET  /video_rect   the same, with the fisheye lens removed (calib/plumbline.json):
                     straight lines are straight, as the mapper's network expects
  GET  /frame.jpg    ?c=<viewer id>&p=hd|sd|ld&f=0|1  the next frame this viewer has
                     not been given yet, as one JPEG. The page keeps a few of
                     these requests open: a slow link then lowers the frame
                     rate instead of piling seconds of video up in the tunnel,
                     which is what a pushed MJPEG stream does.
                     &s=overhead: the drone camera instead (404 without --drone)
  GET  /api/ping     empty, to time the browser <-> VM round trip
  GET  /sim          the MuJoCo simulation live: rover camera and drone view
                     (/frame.jpg?s=rover|drone, same picture-by-picture delivery),
                     plus the live map fusion (polaris repo, sim/fusion_live.py)
  GET  /fusion/<rover.png|drone.png|fused.png|depth_d405.png|depth_camera.png|status.json>
                     one picture or the status of the live fusion, relayed from
                     --fusion (127.0.0.1:8094); nothing else is relayed
  POST /fusion/source {"source": "camera"|"d405"}  which rover sensor feeds the fusion
  POST /fusion/reset   empty the maps of the live fusion (when the fusion offers it)
                     (simulation only: never reaches the robot)
  POST /api/client   what the browser measures (frames shown, delays), logged
  GET  /mission      the live page of the mission for the jury: map, rover, victim, path,
                     decision and its reason, journal, rover camera, perception picture
  GET  /api/mission  the mission document written by lancer_overhead.py --suivi FICHIER
                     (read from that file) with its age; ?v=<version> -> {"inchange": true}
                     while the file has not changed, so a slow link carries the map once
  GET  /perception/debug.jpg  one picture relayed from --perception-debug (default: none)

Two ways to drive. "Direct": when the browser is on the robot's own network it
talks to the robot itself (camera and commands), which avoids two trips across
the Internet; this server then only fetches the few frames per second the
mapper needs, instead of the whole stream. "Via the VM": everything goes
through this server (the only way from elsewhere).
  GET  /api/state    robot telemetry + link figures (camera fps, command latency)
  POST /api/cmd      {"v": m/s, "omega": rad/s}  wanted speed, repeat while held
  POST /api/stop     stop now

Safety: nothing is sent to the robot until a key is pressed. Speeds are clamped
here, and anything that is not a finite number is refused (400). A wanted speed
expires after HOLD seconds without a new browser message (closed tab, lost
focus, broken tunnel), on top of the robot's own 0.5 s timeout and the ESP32
heartbeat. A stop latches: a command that left the browser before the stop and
arrives after it is refused; driving resumes with a new key press (the page
numbers its presses: "press").
"""

import argparse
import json
import math
import os
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests

V_MAX, W_MAX = 0.20, 1.5  # hard limits (m/s, rad/s), below the robot's own v_max 0.3
STALE = 0.6  # s a command may lag behind the fastest one seen before it is dropped
HOLD = 0.45  # s a wanted speed stays valid without a new browser message
# s pause between repeats: the link delay (~0.25 s) already spaces them; robot timeout 0.5 s
RESEND = 0.05
# Picture sizes for the browser: (width or None = as received, JPEG quality).
PRESETS = {'hd': (None, 0), 'sd': (480, 55), 'ld': (320, 50)}
HERE = os.path.dirname(os.path.abspath(__file__))
# Changes at every start of this process: the page restarts its frame numbering when it changes.
BOOT = secrets.token_hex(6)
SESSION_IDLE = 5.0  # s without a command after which a browser's delay reference is relearnt
KEEPALIVE = 2.0  # s: an MJPEG reader gets a byte this often, so a gone reader is noticed
MISSION_MAX = 8_000_000  # bytes: larger mission documents are refused
# What /fusion/ relays from the live fusion process (simulation only).
FUSION_FILES = {
    'rover.png',
    'drone.png',
    'fused.png',
    'depth_d405.png',
    'depth_camera.png',
    'status.json',
}
SNAPSHOT_PERIOD = 0.1  # s: a camera that answers one picture per request is asked this often
SNAPSHOT_MAX = 10_000_000  # bytes: a larger "picture" is refused


def camera_label(url: str | None) -> str | None:
    """'host:port' of a camera URL, without the user name and password it may carry."""
    if not url:
        return None
    parts = urlsplit(url)
    return f'{parts.hostname}:{parts.port}' if parts.port else parts.hostname


def error_text(e: Exception) -> str:
    """An error for the page, without the user name and password of a camera URL (requests
    writes the whole URL in some of its messages)."""
    return re.sub(r'//[^/@\s]+@', '//', f'{type(e).__name__}: {e}')[:160]


class CameraHub:
    """One reader of the robot's MJPEG stream, any number of local viewers."""

    def __init__(
        self, url: str, calib: str | None = None, rect_fov: float = 100.0, snapshots: bool = True
    ):
        self.url = url
        # Lens model (fisheye, estimated from straight edges) -> remap tables
        # to a pinhole image of `rect_fov` degrees horizontally.
        self._maps, self.rect_K, self._rect = None, None, None
        if calib and os.path.exists(calib):
            with open(calib) as f:
                c = json.load(f)
            w, h = c['size']
            K = np.array([[c['f'], 0, c['cx']], [0, c['f'], c['cy']], [0, 0, 1.0]])
            fo = (w / 2) / np.tan(np.radians(rect_fov / 2))
            self.rect_K = np.array([[fo, 0, w / 2], [0, fo, h / 2], [0, 0, 1.0]])
            self._maps = cv2.fisheye.initUndistortRectifyMap(
                K, np.array([c['k1'], 0, 0, 0.0]), np.eye(3), self.rect_K, (w, h), cv2.CV_16SC2
            )
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.fps, self.kbps, self.error = None, 0, 0.0, 0.0, None
        self.t_frame = 0.0  # arrival time of the latest frame
        self._given: dict[str, int] = {}  # viewer id -> last frame handed out
        self._small: tuple[int, str, bytes] | None = None  # last transcoded frame
        self.t_viewer = 0.0  # last time a browser asked for a frame
        self.mode = 'stream'
        self._snap = (0, 0, time.time())  # snapshot URL: pictures, bytes, since when
        for _ in range(
            2 if snapshots else 0
        ):  # a simulator's feed is simply left alone when unwatched
            threading.Thread(target=self._pull, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    def watched(self) -> bool:
        return time.time() - self.t_viewer < 8.0

    def _publish(self, jpeg: bytes) -> None:
        with self.cond:
            self.jpeg, self.seq = jpeg, self.seq + 1
            self.t_frame = time.time()
            self.cond.notify_all()

    def _pull(self) -> None:
        """Nobody watches through this server: single pictures are enough for
        the mapper (two readers, ~4 per second, a quarter of the stream's load
        on the robot's network)."""
        sess, snap = requests.Session(), self.url.rsplit('/', 1)[0] + '/snapshot.jpg'
        n, nbytes, t0 = 0, 0, time.time()
        while True:
            if self.watched():
                time.sleep(0.2)
                n, nbytes, t0 = 0, 0, time.time()
                continue
            self.mode = 'snapshots'
            try:
                jpeg = sess.get(snap, timeout=6).content
                if jpeg[:2] == b'\xff\xd8':
                    self._publish(jpeg)
                    self.error = None
                    n, nbytes = n + 1, nbytes + len(jpeg)
            except Exception as e:
                self.error = error_text(e)
                time.sleep(1.0)
            if time.time() - t0 >= 2.0:
                dt = time.time() - t0
                self.fps, self.kbps = 2 * n / dt, 2 * nbytes / 1024 / dt  # two readers
                n, nbytes, t0 = 0, 0, time.time()

    def _run(self) -> None:
        while True:
            if not self.watched():
                time.sleep(0.2)
                continue
            try:
                asked = time.time()
                with requests.get(self.url, stream=True, timeout=8) as r:
                    r.raise_for_status()
                    self.error = None
                    kind = r.headers.get('Content-Type', '').lower()
                    if kind.startswith('image/'):
                        # One picture per request (a phone app's snapshot URL): ask again.
                        self._one_picture(self._read_picture(r), asked)
                        continue
                    if not kind.startswith('multipart'):  # e.g. an mp4 or a web page
                        raise ValueError(
                            f'not an MJPEG stream nor a JPEG picture: {kind or "no Content-Type"}'
                        )
                    self.mode = 'stream'
                    buf, n, nbytes, t0 = b'', 0, 0, time.time()
                    for chunk in r.iter_content(chunk_size=16384):
                        if not self.watched():
                            break
                        buf += chunk
                        nbytes += len(chunk)
                        while True:
                            end = buf.find(b'\xff\xd9')
                            if end < 0:
                                break
                            start = buf.rfind(b'\xff\xd8', 0, end)
                            if start >= 0:
                                self._publish(buf[start : end + 2])
                                n += 1
                            buf = buf[end + 2 :]
                        if time.time() - t0 >= 2.0:
                            dt = time.time() - t0
                            self.fps, self.kbps = n / dt, nbytes / 1024 / dt
                            n, nbytes, t0 = 0, 0, time.time()
            except Exception as e:  # tunnel hiccup: retry
                self.error = error_text(e)
                self.fps = 0.0
                time.sleep(1.0)

    @staticmethod
    def _read_picture(r) -> bytes:
        body = b''
        for chunk in r.iter_content(chunk_size=65536):
            body += chunk
            if len(body) > SNAPSHOT_MAX:
                raise ValueError(f'picture larger than {SNAPSHOT_MAX} bytes')
        return body

    def _one_picture(self, jpeg: bytes, asked: float) -> None:
        """A snapshot URL answered: keep the picture, count it, wait before asking again."""
        if jpeg[:2] != b'\xff\xd8':
            raise ValueError('the camera answered something that is not a JPEG picture')
        self._publish(jpeg)
        self.mode = 'snapshot URL'
        n, nbytes, t0 = self._snap
        n, nbytes = n + 1, nbytes + len(jpeg)
        now = time.time()
        if now - t0 >= 2.0:
            self.fps, self.kbps = n / (now - t0), nbytes / 1024 / (now - t0)
            n, nbytes, t0 = 0, 0, now
        self._snap = (n, nbytes, t0)
        time.sleep(max(0.0, SNAPSHOT_PERIOD - (now - asked)))

    def wait(self, last_seq: int, timeout: float = 5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.seq, self.jpeg

    def rectified(self, seq: int, jpeg: bytes) -> bytes:
        """`jpeg` without lens distortion (one conversion per frame, shared)."""
        cached = self._rect
        if cached is not None and cached[0] == seq:
            return cached[1]
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None or self._maps is None or img.shape[:2] != self._maps[0].shape[:2]:
            return jpeg
        img = cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        out = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
        self._rect = (seq, out)
        return out

    def next_for(self, viewer: str, preset: str, fluid: bool = False, timeout: float = 2.0):
        """(seq, jpeg, age in s) of the newest frame not yet given to `viewer`,
        or None. Parallel requests of one viewer each get a different frame."""
        if fluid:  # only then is the whole stream worth its load on the robot's network
            self.t_viewer = time.time()
        with self.cond:
            if len(self._given) > 200:
                self._given.clear()
            if not self.cond.wait_for(lambda: self.seq > self._given.get(viewer, 0), timeout):
                return None
            seq, jpeg, t = self.seq, self.jpeg, self.t_frame
            self._given[viewer] = seq
        width, quality = PRESETS.get(preset, PRESETS['sd'])
        if width:
            small = self._small
            if small is None or small[:2] != (seq, preset):
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[1] > width:
                    img = cv2.resize(
                        img,
                        (width, round(img.shape[0] * width / img.shape[1])),
                        interpolation=cv2.INTER_AREA,
                    )
                if img is not None:
                    jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[
                        1
                    ].tobytes()
                self._small = small = (seq, preset, jpeg)
            jpeg = small[2]
        return seq, jpeg, time.time() - t


class Robot:
    def __init__(self, base: str):
        self.base = base.rstrip('/')
        self.lock = threading.Lock()
        self.want, self.want_t = (0.0, 0.0), 0.0
        self.sent, self.sent_t = (0.0, 0.0), 0.0
        self.stop_req = False
        self.state, self.state_t, self.state_ms = {}, 0.0, None
        self.cmd_ms, self.cmd_err, self.n_cmd = None, None, 0
        # Per browser session: delay reference, last command time, presses seen, first press
        # allowed after a stop. Global, one browser's clock blocked every other one.
        self._sessions: dict[str, dict] = {}
        self._latch_legacy = False  # stop latch for clients that do not number their presses
        self.n_stale, self.n_latched, self.n_stop = 0, 0, 0
        self.refused: tuple[float, str] | None = None  # (time, reason) of the last refusal
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()

    @staticmethod
    def _number(x, name: str) -> float:
        """float(x), or ValueError: NaN would pass the clamp as the full speed."""
        if isinstance(x, bool):
            raise ValueError(f'{name}: not a number')
        try:
            f = float(x)
        except OverflowError:
            f = math.inf
        if not math.isfinite(f):
            raise ValueError(f'{name}: not a finite number')
        return f

    def _session(self, key: str) -> dict:
        s = self._sessions.get(key)
        if s is None:
            if len(self._sessions) >= 64:  # forget the browser silent for the longest
                del self._sessions[min(self._sessions, key=lambda k: self._sessions[k]['seen'])]
            s = self._sessions[key] = {'lead': None, 'seen': -math.inf, 'press': None, 'min': 0}
        return s

    def set(
        self,
        v: float,
        w: float,
        t_client: float | None = None,
        session: str | None = None,
        press: int | None = None,
    ) -> bool:
        """Wanted speed from one browser; False when refused (late, or before a stop)."""
        v = max(-V_MAX, min(V_MAX, self._number(v, 'v')))
        w = max(-W_MAX, min(W_MAX, self._number(w, 'omega')))
        t = None if t_client is None else self._number(t_client, 't')
        press = None if press is None else int(self._number(press, 'press'))
        now = time.monotonic()
        with self.lock:
            reason = None
            if v or w:
                s = self._session('' if session is None else str(session)[:64])
                # A "go" that spent seconds in a congested tunnel must not move the robot
                # when it finally arrives (seen: 12 s round trips). The browser stamps each
                # command with its own clock; the smallest difference with ours is the
                # clock offset plus the best transit time, and a command much later than
                # that is old. Per browser, relearnt after a pause: another computer's
                # clock, or one wrong stamp, must not block driving for good.
                if t is not None:
                    lead = time.time() - t / 1000.0
                    if s['lead'] is None or now - s['seen'] > SESSION_IDLE:
                        s['lead'] = lead
                    else:
                        s['lead'] = min(s['lead'] + 0.0005, lead)
                    if lead - s['lead'] > STALE:
                        reason = 'stale'
                s['seen'] = now
                if press is not None:
                    s['press'] = press if s['press'] is None else max(s['press'], press)
                    if press < s['min']:
                        reason = reason or 'latched'
                elif self._latch_legacy:
                    reason = reason or 'latched'
            if reason:
                if reason == 'stale':
                    self.n_stale += 1
                else:
                    self.n_latched += 1
                self.refused = (time.time(), reason)
                return False
            self.want, self.want_t = (v, w), now
            if not (v or w) and press is None:
                self._latch_legacy = False  # such a client released its keys
            return True

    def stop(self, session: str | None = None, press=None) -> None:
        """Stop now, and refuse every key press made before this stop (all browsers)."""
        try:
            press = None if press is None else int(self._number(press, 'press'))
        except (ValueError, TypeError):
            press = None  # a stop is obeyed whatever comes with it
        with self.lock:
            self.want, self.stop_req = (0.0, 0.0), True
            self.n_stop += 1
            self._latch_legacy = True
            for s in self._sessions.values():
                if s['press'] is not None:
                    s['min'] = max(s['min'], s['press'] + 1)
            if session is not None and press is not None:
                s = self._session(str(session)[:64])
                s['min'] = max(s['min'], press + 1)

    def _post(self, sess, path: str, body: dict) -> bool:
        t = time.monotonic()
        try:
            sess.post(self.base + path, json=body, timeout=3).raise_for_status()
            self.cmd_ms, self.cmd_err = 1000 * (time.monotonic() - t), None
            self.n_cmd += 1
            return True
        except Exception as e:
            self.cmd_err = f'{type(e).__name__}: {e}'[:160]
            return False

    def _sender(self) -> None:
        sess = requests.Session()
        while True:
            time.sleep(0.03)
            now = time.monotonic()
            with self.lock:
                if now - self.want_t > HOLD:
                    self.want = (0.0, 0.0)
                want, stop_req = self.want, self.stop_req
                self.stop_req = False
            if stop_req:
                if self._post(sess, '/stop', {}):
                    self.sent = (0.0, 0.0)
                else:
                    self.stop_req = True
                continue
            moving = want != (0.0, 0.0)
            if moving and (want != self.sent or now - self.sent_t >= RESEND):
                if self._post(sess, '/cmd', {'v': want[0], 'omega': want[1]}):
                    self.sent, self.sent_t = want, time.monotonic()
            elif not moving and self.sent != (0.0, 0.0):
                # Released: say so at once instead of waiting for the robot's timeout.
                if self._post(sess, '/cmd', {'v': 0.0, 'omega': 0.0}):
                    self.sent = (0.0, 0.0)

    def _poller(self) -> None:
        sess = requests.Session()
        while True:
            t = time.monotonic()
            try:
                s = sess.get(self.base + '/state', timeout=4).json()
                self.state, self.state_t = s, time.time()
                self.state_ms = 1000 * (time.monotonic() - t)
            except Exception:
                self.state_ms = None
            time.sleep(0.5)


def _finite_only(x):
    """The document with every non-finite float replaced by None: the page's JSON.parse
    refuses NaN, and one NaN must not blank the whole mission page."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite_only(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite_only(v) for v in x]
    return x


def read_mission(path: str | None, version: str | None = None) -> tuple[int, dict]:
    """(HTTP status, answer) for GET /api/mission: the document of lancer_overhead --suivi."""
    if not path:
        return 404, {'erreur': 'pas de fichier de suivi : lancer server.py avec --suivi FICHIER'}
    name = os.path.basename(path)
    try:
        st = os.stat(path)
        if st.st_size > MISSION_MAX:
            return 503, {'erreur': f'document trop gros ({st.st_size} octets)', 'fichier': name}
        age = time.time() - st.st_mtime
        if version is not None and version == str(st.st_mtime_ns):
            return 200, {'inchange': True, 'age_s': age, 'version': version, 'fichier': name}
        with open(path, 'rb') as f:
            doc = json.loads(f.read())
    except FileNotFoundError:
        return 503, {'erreur': 'en attente du premier document de la mission', 'fichier': name}
    except (OSError, ValueError) as e:  # written by hand, or not atomically
        return 503, {'erreur': f'document illisible : {type(e).__name__}', 'fichier': name}
    if not isinstance(doc, dict):
        return 503, {'erreur': "le document n'est pas un objet JSON", 'fichier': name}
    t = doc.get('t')
    age_doc = time.time() - t if isinstance(t, int | float) and math.isfinite(t) else None
    return 200, {
        'age_s': age,
        'age_doc_s': age_doc,
        'version': str(st.st_mtime_ns),
        'fichier': name,
        'doc': _finite_only(doc),
    }


def make_handler(
    robot: Robot,
    hub: CameraHub,
    map_port: int,
    robot_lan: str,
    sims: dict,
    fusion: str = 'http://127.0.0.1:8094',
    suivi: str | None = None,
    perception_debug: str | None = None,
    overhead: CameraHub | None = None,
):
    """`overhead`: the drone camera (--drone), None without one."""

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), 'application/json')

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split('?')[0]
            if path == '/':
                with open(os.path.join(HERE, 'index.html'), 'rb') as f:
                    page = (
                        f.read()
                        .replace(b'__MAP_PORT__', str(map_port).encode())
                        .replace(b'__ROBOT_LAN__', robot_lan.encode())
                    )
                self._send(200, page, 'text/html; charset=utf-8')
            elif path == '/api/state':
                s = robot.state
                age = time.time() - robot.state_t if robot.state_t else None
                self._json(
                    {
                        'robot': {
                            k: s.get(k)
                            for k in ('battery_v', 'link', 'odom', 'camera', 'sim', 'cpu_temp')
                        },
                        'ctrl': {
                            k: (s.get('ctrl') or {}).get(k) for k in ('mode', 'v_cmd', 'w_cmd')
                        },
                        'state_age': age,
                        'state_ms': robot.state_ms,
                        'cmd_ms': robot.cmd_ms,
                        'cmd_err': robot.cmd_err,
                        'n_cmd': robot.n_cmd,
                        'n_stale': robot.n_stale,
                        'n_latched': robot.n_latched,
                        # the reason of a refusal in the last 2 s, for the page to show
                        'refused': (
                            robot.refused[1]
                            if robot.refused and time.time() - robot.refused[0] < 2.0
                            else None
                        ),
                        'boot': BOOT,
                        'want': robot.want,
                        'limits': {'v': V_MAX, 'w': W_MAX},
                        'video': {
                            'fps': round(hub.fps, 1),
                            'kbps': round(hub.kbps),
                            'error': hub.error,
                            'mode': hub.mode,
                        },
                        'drone': (
                            {'configured': False}
                            if overhead is None
                            else {
                                'configured': True,
                                'camera': camera_label(overhead.url),
                                'fps': round(overhead.fps, 1),
                                'kbps': round(overhead.kbps),
                                'error': overhead.error,
                                'mode': overhead.mode,
                                'age_s': (
                                    round(time.time() - overhead.t_frame, 1)
                                    if overhead.t_frame
                                    else None
                                ),
                            }
                        ),
                    }
                )
            elif path == '/sim':
                with open(os.path.join(HERE, 'sim.html'), 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/api/ping':
                self._send(200, b'', 'text/plain')
            elif path == '/mission':
                with open(os.path.join(HERE, 'mission.html'), 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/api/mission':
                q = dict(
                    x.split('=', 1) for x in self.path.partition('?')[2].split('&') if '=' in x
                )
                code, answer = read_mission(suivi, q.get('v'))
                answer['perception'] = bool(perception_debug)
                self._json(answer, code)
            elif path == '/perception/debug.jpg':
                if not perception_debug:
                    return self._json({'erreur': 'pas de --perception-debug'}, 404)
                try:
                    r = requests.get(perception_debug, timeout=3)
                except requests.RequestException as e:
                    err = f'perception injoignable : {type(e).__name__}'
                    return self._json({'erreur': err}, 503)
                if r.status_code != 200:
                    return self._json({'erreur': f'perception : HTTP {r.status_code}'}, 503)
                self.send_response(200)
                self.send_header('Content-Type', r.headers.get('Content-Type', 'image/jpeg'))
                self.send_header('Content-Length', str(len(r.content)))
                self.send_header('Cache-Control', 'no-store')
                if 'X-Age-Ms' in r.headers:
                    self.send_header('X-Age-Ms', r.headers['X-Age-Ms'])
                self.end_headers()
                self.wfile.write(r.content)
            elif path.startswith('/fusion/'):
                name = path[len('/fusion/') :]
                if name not in FUSION_FILES:
                    return self._json({'error': 'not found'}, 404)
                try:
                    r = requests.get(f'{fusion}/{name}', timeout=4)
                except requests.RequestException as e:
                    return self._json({'error': f'fusion not running: {type(e).__name__}'}, 503)
                self.send_response(r.status_code)
                self.send_header(
                    'Content-Type', r.headers.get('Content-Type', 'application/octet-stream')
                )
                self.send_header('Content-Length', str(len(r.content)))
                self.send_header('Cache-Control', 'no-store')
                for h in ('X-Age-Ms', 'X-Seq'):
                    if h in r.headers:
                        self.send_header(h, r.headers[h])
                self.end_headers()
                self.wfile.write(r.content)
            elif path == '/frame.jpg':
                q = dict(
                    x.split('=', 1) for x in self.path.partition('?')[2].split('&') if '=' in x
                )
                s = q.get('s', '')
                # the robot's camera, the drone camera, or a simulator view
                src = overhead if s == 'overhead' else sims.get(s)
                if s and src is None:  # never the rover camera in place of another one
                    return self._json({'error': f'no camera {s!r}'}, 404)
                got = (
                    src.next_for(q.get('c', 'x'), q.get('p', 'sd'), True)
                    if src is not None
                    else hub.next_for(
                        q.get('c', self.client_address[0]), q.get('p', 'sd'), q.get('f') == '1'
                    )
                )
                if got is None:
                    return self._send(204, b'', 'image/jpeg')
                seq, jpeg, age = got
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Seq', str(seq))
                self.send_header('X-Age-Ms', str(round(1000 * age)))
                self.send_header('X-Boot', BOOT)  # frame numbers restart with the process
                # The page fetches frames from the other loopback name
                # (localhost <-> 127.0.0.1): a second pool of browser
                # connections, so frames never queue in front of commands.
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Expose-Headers', 'X-Seq, X-Age-Ms, X-Boot')
                self.end_headers()
                self.wfile.write(jpeg)
            elif path in ('/video_feed', '/video_rect', '/drone_feed'):
                cam = overhead if path == '/drone_feed' else hub
                if cam is None:
                    return self._json({'error': 'no drone camera: start with --drone URL'}, 404)
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                seq = -1
                try:
                    while True:
                        if cam is overhead:
                            # The drone camera has no snapshot reader of its own: this
                            # reader is what keeps its stream open.
                            cam.t_viewer = time.time()
                        got, jpeg = cam.wait(seq, KEEPALIVE)
                        if jpeg is None or got == seq:
                            # No new frame: a byte anyway, or a reader that left while the
                            # robot was offline would hold this thread and its socket for
                            # ever. Between parts, MJPEG readers skip it.
                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                            seq = got
                            continue
                        seq = got
                        if path == '/video_rect':
                            jpeg = hub.rectified(seq, jpeg)
                        else:
                            cam.t_viewer = time.time()
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                            + str(len(jpeg)).encode()
                            + b'\r\n\r\n'
                            + jpeg
                            + b'\r\n'
                        )
                except OSError:  # the reader left
                    pass
                self.close_connection = True
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == '/api/stop':
                # Obeyed whatever the body: a malformed stop is still a stop.
                body = None
                try:
                    n = int(self.headers.get('Content-Length') or 0)
                    if 0 <= n < 65536:
                        body = json.loads(self.rfile.read(n) or b'{}')
                    else:
                        self.close_connection = True  # body left unread
                except (ValueError, OSError):
                    self.close_connection = True
                body = body if isinstance(body, dict) else {}
                robot.stop(body.get('c'), body.get('press'))
                return self._json({'ok': True})
            try:
                n = int(self.headers.get('Content-Length') or 0)
                if not 0 <= n < 1_000_000:
                    raise ValueError('bad Content-Length')
                body = json.loads(self.rfile.read(n) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('the body must be a JSON object')
                if self.path == '/api/cmd':
                    ok = robot.set(
                        body.get('v', 0),
                        body.get('omega', 0),
                        body.get('t'),
                        body.get('c'),
                        body.get('press'),
                    )
                    return self._json({'ok': True, 'accepted': ok})
                elif self.path in ('/fusion/source', '/fusion/reset'):  # relayed to the fusion
                    name = self.path[len('/fusion/') :]
                    data = {'source': str(body.get('source'))} if name == 'source' else {}
                    try:
                        r = requests.post(f'{fusion}/{name}', json=data, timeout=4)
                    except requests.RequestException as e:
                        return self._json(
                            {'error': f'fusion not running: {type(e).__name__}'}, 503
                        )
                    return self._send(r.status_code, r.content, 'application/json')
                elif self.path == '/api/client':
                    print(time.strftime('%H:%M:%S'), 'client', json.dumps(body)[:400], flush=True)
                else:
                    return self._json({'error': 'not found'}, 404)
                self._json({'ok': True})
            except (ValueError, TypeError, OverflowError) as e:
                self.close_connection = True  # the body may be left unread
                self._json({'error': str(e)[:200]}, 400)

    return H


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='http://127.0.0.1:18080')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8082)
    ap.add_argument('--map-port', type=int, default=8081, help='port of the 3D viewer (run.py)')
    ap.add_argument(
        '--calib',
        default=os.path.join(HERE, 'calib', 'plumbline.json'),
        help='lens model for /video_rect',
    )
    ap.add_argument(
        '--rect-fov', type=float, default=100.0, help='horizontal field of /video_rect (deg)'
    )
    ap.add_argument(
        '--robot-lan',
        default='',
        help="the robot's address on its own network (e.g. http://192.168.1.42:8080), tried "
        "by the browser for the direct link ('' = never)",
    )
    ap.add_argument('--sim', default='http://127.0.0.1:5050', help='simulator shown on /sim')
    ap.add_argument(
        '--fusion', default='http://127.0.0.1:8094', help='live map fusion shown on /sim'
    )
    ap.add_argument(
        '--suivi',
        default=None,
        help='mission document of merging_map/lancer_overhead.py --suivi, shown on /mission',
    )
    ap.add_argument(
        '--perception-debug',
        default=None,
        help='URL of one debug picture of the rover perception, shown on /mission (default: none)',
    )
    ap.add_argument(
        '--drone',
        default=None,
        help='the drone camera: MJPEG stream or JPEG snapshot URL of a phone camera app, e.g. '
        'http://192.168.1.50:8081/video (default: none)',
    )
    args = ap.parse_args()
    sims = {
        'rover': CameraHub(args.sim.rstrip('/') + '/video_feed', snapshots=False),
        'drone': CameraHub(args.sim.rstrip('/') + '/drone_feed', snapshots=False),
    }
    hub = CameraHub(args.robot.rstrip('/') + '/video_feed', args.calib, args.rect_fov)
    # The phone is read only while someone watches it (the page, /drone_feed readers).
    overhead = CameraHub(args.drone, snapshots=False) if args.drone else None
    srv = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            Robot(args.robot),
            hub,
            args.map_port,
            args.robot_lan,
            sims,
            args.fusion.rstrip('/'),
            args.suivi,
            args.perception_debug,
            overhead,
        ),
    )
    srv.daemon_threads = True
    print(f'teleop: http://{args.host}:{args.port}  robot: {args.robot}', flush=True)
    if overhead is not None:
        print(f'drone camera: {camera_label(args.drone)}', flush=True)
    srv.serve_forever()
