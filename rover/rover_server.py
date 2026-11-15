"""RaspRover test server: drive the rover over Wi-Fi from a browser, with the camera feed.

Runs on the rover's onboard computer (RDK X5). Standard library + pyserial + OpenCV.

    python3 rover_server.py                 # real robot, ESP32 port found automatically
    python3 rover_server.py --port-serial /dev/ttyUSB0
    python3 rover_server.py --sim           # simulated ESP32 and camera, for a laptop

Then open http://<rover IP>:8080.

HTTP API (JSON bodies). The first two match the shared simulation in sim/run_world.py:
  GET  /video_feed     camera, MJPEG
  POST /cmd            {"v": m/s, "omega": rad/s}  or a raw RaspRover command:
                       {"T":13,"X":..,"Z":..}, {"T":1,"L":..,"R":..}, {"T":0} (stop),
                       anything else is forwarded to the ESP32 as is.
                       Speed commands last `manual_timeout` (0.5 s): repeat them at >= 5 Hz.
  GET  /state          telemetry: link, battery, pose, wheel speeds, gyro, current step;
                       boot_id (changes when the server restarts) and uptime_s;
                       stop_count: stops received since start (/stop, /cmd {"T":0})
  GET  /snapshot.jpg   one camera picture, its age in the X-Age-Ms header; 503 when the
                       camera gives no picture newer than SNAPSHOT_MAX_AGE
  POST /move           {"distance": m}            (+ forward, - backward)
  POST /turn           {"angle": deg}             (+ left, - right)
  POST /sequence       {"steps": [{"move": 0.5}, {"turn": 90}, {"path": [[x, y], ...]}, ...]}
  POST /path           follow a trajectory: {"points": [[x, y], ...], "frame": "local"|"odom",
                       "speed": m/s}. local (default) = metres relative to the rover now, x
                       forward, y left; odom = the odometry frame (x, y in /state). Sent again,
                       it replaces the running one. Progress and errors in /state ctrl.path.
  POST /stop           stop now and drop the queued steps
  POST /reset_odom     pose back to (0, 0, 0 deg)
  POST /calibrate_gyro measure the gyro bias again (rover must stand still 2 s)
  POST /sim            simulation only: {"drift": rad/s} the rover turns by itself while moving
  POST /gimbal         camera pan-tilt: {"pan_rate": -1|0|1, "tilt_rate": -1|0|1} while a key
                       is held (repeat it like /cmd), {"pan": deg, "tilt": deg} (pan + = right,
                       tilt + = up), or {"center": true}. A raw {"T":133,...} on /cmd works too.
  GET  /config         settings;  POST /config {"name": value, ...} changes and saves them

Numbers must be finite and within the limits of rover_core (MAX_*, CONFIG_LIMITS): otherwise
HTTP 400 and nothing moves or changes. Steps before "Gyro OK": HTTP 409.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rover_core import (
    MAX_MANUAL_V,
    Config,
    NotReadyError,
    Rover,
    Step,
    find_esp32_port,
    finite,
    open_serial,
)

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, 'web', 'index.html')
CPU_TEMP = '/sys/class/thermal/thermal_zone1/temp'  # RDK X5 CPU (zone0 is the DDR)
# Kernel send buffer for each video client. Small on purpose: on a slow Wi-Fi, frames must be
# dropped here, not queued in the socket (a 1.5 MB queue is 1.5 s of delayed video).
STREAM_SNDBUF = 48 * 1024
# /snapshot.jpg: a picture older than this is not served as the rover's view (the camera
# was released after IDLE_RELEASE, or it failed); wait up to SNAPSHOT_WAIT for a new one.
SNAPSHOT_MAX_AGE = 2.0  # s
SNAPSHOT_WAIT = 3.0  # s, opening the camera again takes about 1 s
# Identity of this server process: a client sees a restart (odometry reset) by a new boot_id.
BOOT_ID = secrets.token_hex(8)
STARTED = time.monotonic()

try:
    import cv2
except ImportError:  # the server still drives the rover without a camera
    cv2 = None


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------


class Camera:
    """Grabs and JPEG-encodes frames only while someone is watching (the RDK has no fan)."""

    IDLE_RELEASE = 5.0  # s without viewers before the camera is closed

    def __init__(self, source, width: int, height: int, fps: float, quality: int) -> None:
        self.source = source
        self.width, self.height, self.fps, self.quality = width, height, fps, quality
        self.frame: bytes | None = None
        self.frame_id = 0
        self.frame_t: float | None = None  # time.monotonic() when the frame was published
        self.error: str | None = None if cv2 else 'OpenCV (python3-opencv) is not installed'
        self.measured_fps = 0.0
        self._cond = threading.Condition()
        self._last_request = 0.0
        self._thread: threading.Thread | None = None

    def request(self) -> None:
        """Mark that a viewer wants frames; starts the capture thread if needed."""
        self._last_request = time.monotonic()
        if cv2 is None:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name='camera', daemon=True)
            self._thread.start()

    def wait_frame(self, last_id: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
        self.request()
        with self._cond:
            self._cond.wait_for(
                lambda: self.frame is not None and self.frame_id != last_id, timeout=timeout
            )
            return self.frame, self.frame_id

    def latest(self) -> tuple[bytes | None, int, float | None]:
        """(newest frame, its id, its age in s) without waiting; age None without frame."""
        with self._cond:
            age = None if self.frame_t is None else time.monotonic() - self.frame_t
            return self.frame, self.frame_id, age

    def _publish(self, jpeg: bytes) -> None:
        with self._cond:
            self.frame = jpeg
            self.frame_id += 1
            self.frame_t = time.monotonic()
            self._cond.notify_all()

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if os.name == 'posix' else cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _frames(self):
        """Yields BGR images until no one watches. Subclasses replace it (simulation)."""
        cap = self._open()
        if not cap.isOpened():
            self.error = f'cannot open camera {self.source!r}'
            time.sleep(2.0)
            return
        self.error = None
        try:
            while time.monotonic() - self._last_request < self.IDLE_RELEASE:
                ok, img = cap.read()
                if not ok:
                    self.error = 'camera read failed'
                    time.sleep(0.5)
                    return
                yield img
        finally:
            cap.release()

    def _run(self) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        period = 1.0 / self.fps
        count, t_count = 0, time.monotonic()
        nxt = time.monotonic()
        for img in self._frames():
            now = time.monotonic()
            if now < nxt:  # the camera may deliver more fps than asked: drop extra frames
                continue
            nxt = max(nxt + period, now)
            ok, jpeg = cv2.imencode('.jpg', img, params)
            if ok:
                self._publish(jpeg.tobytes())
            count += 1
            if now - t_count >= 2.0:
                self.measured_fps = count / (now - t_count)
                count, t_count = 0, now
        self.measured_fps = 0.0


class SimCamera(Camera):
    """Top view of the simulated rover, drawn instead of a camera image."""

    def __init__(self, board, **kw) -> None:
        super().__init__(source='sim', **kw)
        self.board = board

    def _frames(self):
        import numpy as np

        self.error = None
        w, h = self.width, self.height
        scale = h / 3.0  # pixels per metre: 3 m tall view
        while time.monotonic() - self._last_request < self.IDLE_RELEASE:
            time.sleep(1.0 / self.fps)
            b = self.board
            img = np.full((h, w, 3), 32, np.uint8)
            cx, cy = b.x, b.y  # the view follows the rover

            def px(x, y, cx=cx, cy=cy):
                return int(w / 2 - (y - cy) * scale), int(h / 2 - (x - cx) * scale)

            step = 0.1
            for i in range(-30, 31):  # 10 cm grid, x up, y left
                gx = math.floor(cx / step) * step + i * step
                gy = math.floor(cy / step) * step + i * step
                col = (70, 70, 70) if round(gx / step) % 10 else (110, 110, 110)
                cv2.line(img, px(gx, cy - 5), px(gx, cy + 5), col, 1)
                col = (70, 70, 70) if round(gy / step) % 10 else (110, 110, 110)
                cv2.line(img, px(cx - 5, gy), px(cx + 5, gy), col, 1)
            cv2.circle(img, px(0, 0), 5, (80, 200, 255), 2)
            trail = b.trail[-800:]
            for (x1, y1), (x2, y2) in zip(trail, trail[1:], strict=False):
                cv2.line(img, px(x1, y1), px(x2, y2), (60, 170, 60), 2)
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            body = [(0.12, 0.09), (0.12, -0.09), (-0.12, -0.09), (-0.12, 0.09)]
            pts = [px(b.x + c * x - s * y, b.y + s * x + c * y) for x, y in body]
            cv2.polylines(img, [np.array(pts, np.int32)], True, (230, 230, 230), 2)
            cv2.arrowedLine(img, px(b.x, b.y), px(b.x + 0.2 * c, b.y + 0.2 * s), (0, 200, 255), 2)
            cam = b.yaw - math.radians(b.pan)  # pan + = right, i.e. clockwise from above
            for half in (-0.45, 0.45):  # camera field of view, about 90 deg
                end = (b.x + 0.5 * math.cos(cam + half), b.y + 0.5 * math.sin(cam + half))
                cv2.line(img, px(b.x, b.y), px(*end), (255, 200, 80), 1)
            yaw = math.degrees(b.yaw)
            text = f'SIMULATION  true pose x={b.x:+.3f} y={b.y:+.3f} yaw={yaw:+.1f}'
            cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            text = f'camera pan {b.pan:+.0f} tilt {b.tilt:+.0f}  drift {b.yaw_drift:+.2f} rad/s'
            cv2.putText(img, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1)
            cv2.putText(img, 'grid 10 cm', (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (150, 150, 150), 1)  # fmt: skip
            yield img


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def read_cpu_temp() -> float | None:
    try:
        with open(CPU_TEMP, encoding='ascii') as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class App:
    def __init__(self, rover: Rover, camera: Camera, cfg_path: str, sim_board=None) -> None:
        self.rover = rover
        self.camera = camera
        self.cfg_path = cfg_path
        self.sim_board = sim_board

    def state(self) -> dict:
        s = self.rover.snapshot()
        s['cpu_temp'] = read_cpu_temp()
        s['camera'] = {'error': self.camera.error, 'fps': round(self.camera.measured_fps, 1)}
        s['sim'] = self.sim_board is not None
        s['boot_id'] = BOOT_ID
        s['uptime_s'] = round(time.monotonic() - STARTED, 1)
        if self.sim_board is not None:
            b = self.sim_board
            s['sim_truth'] = {
                'x': b.x,
                'y': b.y,
                'heading_deg': math.degrees(b.yaw),
                'drift': b.yaw_drift,
            }
        return s


def json_safe(obj):
    """`obj` with NaN and infinities replaced by None: JSON.parse refuses them."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


def _refuse_constant(name: str):
    raise ValueError(f'{name} is not a valid number')


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'RoverTest/1.0'

        def log_message(self, fmt, *args):  # only log errors, not every poll
            pass

        # -- helpers ----------------------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(json_safe(obj), allow_nan=False).encode()
            self._send(code, body, 'application/json')

        def _body(self) -> dict:
            n = int(self.headers.get('Content-Length') or 0)
            if not n:
                return {}
            data = json.loads(self.rfile.read(n), parse_constant=_refuse_constant)
            if not isinstance(data, dict):
                raise ValueError('expected a JSON object')
            return data

        # -- routes -----------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                with open(INDEX_HTML, 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/state':
                self._json(app.state())
            elif path == '/config':
                self._json(json.loads(json.dumps(app.rover.cfg.__dict__)))
            elif path == '/video_feed':
                self._stream()
            elif path == '/snapshot.jpg':
                self._snapshot()
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split('?')[0]
            rover = app.rover
            try:
                body = self._body()
                if path == '/cmd':
                    self._cmd(body)
                elif path == '/move':
                    rover.run([Step('move', body['distance'])])
                elif path == '/turn':
                    rover.run([Step('turn', body['angle'])])
                elif path == '/sequence':
                    if not isinstance(body['steps'], list):
                        raise ValueError('steps must be a list')
                    rover.run([Step.parse(s) for s in body['steps']])
                elif path == '/path':
                    rover.run(
                        [Step.path(body['points'], body.get('frame', 'local'), body.get('speed'))]
                    )
                elif path == '/stop':
                    rover.stop()
                elif path == '/reset_odom':
                    rover.reset_odometry()
                    if app.sim_board is not None:  # keep the true pose aligned with the estimate
                        b = app.sim_board
                        b.x = b.y = b.yaw = 0.0
                        b.trail.clear()
                elif path == '/calibrate_gyro':
                    rover.recalibrate_gyro()
                elif path == '/sim' and app.sim_board is not None:
                    app.sim_board.yaw_drift = finite(body['drift'], 'drift', 5.0)
                elif path == '/gimbal':
                    if body.get('center'):
                        rover.gimbal_set(0.0, 0.0)
                    elif 'pan' in body or 'tilt' in body:
                        g = rover.gimbal
                        rover.gimbal_set(
                            finite(body.get('pan', g.pan), 'pan', 360.0),
                            finite(body.get('tilt', g.tilt), 'tilt', 360.0),
                        )
                    else:
                        rover.gimbal_rate(
                            finite(body.get('pan_rate', 0), 'pan_rate', 1.0),
                            finite(body.get('tilt_rate', 0), 'tilt_rate', 1.0),
                        )
                elif path == '/config':
                    # All or nothing (Config.update), saved under the lock: two pages saving at
                    # once cannot interleave their writes.
                    with rover.lock:
                        changed = rover.cfg.update(body)
                        rover.cfg.save(app.cfg_path)
                    self._json({'ok': True, 'changed': changed})
                    return
                else:
                    self._json({'error': 'not found'}, 404)
                    return
            except NotReadyError as e:
                self._json({'error': str(e)}, 409)
                return
            except (KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
                self._json({'error': f'bad request: {e}'}, 400)
                return
            self._json({'ok': True})

        def _cmd(self, body: dict) -> None:
            rover = app.rover
            kind = body.get('T')
            if 'v' in body or 'omega' in body:  # checked (finite, bounded) by Controller.manual
                rover.manual(body.get('v', 0), body.get('omega', 0))
            elif kind == 13:
                rover.manual(body.get('X', 0), body.get('Z', 0))
            elif kind == 1:  # wheel speeds -> (v, w), the ESP32 does the inverse for T:13
                left = finite(body.get('L', 0), 'L', MAX_MANUAL_V)
                right = finite(body.get('R', 0), 'R', MAX_MANUAL_V)
                rover.manual((left + right) / 2, (right - left) / rover.cfg.track)
            elif kind == 133:  # pan-tilt: keep the commanded angles in sync
                rover.gimbal_set(
                    finite(body.get('X', 0), 'pan', 360.0), finite(body.get('Y', 0), 'tilt', 360.0)
                )
            elif kind == 0:
                rover.stop()
            elif kind is not None:
                rover.raw(body)
            else:
                raise ValueError('expected {"v","omega"} or a RaspRover {"T": ...} command')

        def _snapshot(self) -> None:
            cam = app.camera
            cam.request()  # keeps the camera open while a client polls pictures
            frame, fid, age = cam.latest()
            if frame is None or age > SNAPSHOT_MAX_AGE:
                # Idle camera (released after IDLE_RELEASE) or a stalled one: a picture taken
                # before the rover last moved must not pass for its view now.
                cam.wait_frame(fid, timeout=SNAPSHOT_WAIT)
                frame, fid, age = cam.latest()
            age_ms = None if age is None else str(int(age * 1000))
            if frame is None or age > SNAPSHOT_MAX_AGE:
                body = json.dumps({'error': cam.error or 'no recent camera frame'}).encode()
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                if age_ms is not None:
                    self.send_header('X-Age-Ms', age_ms)
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(frame)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Age-Ms', age_ms)
            self.end_headers()
            self.wfile.write(frame)

        def _stream(self) -> None:
            boundary = b'frame'
            query = self.path.partition('?')[2]
            params = dict(kv.partition('=')[::2] for kv in query.split('&') if kv)
            try:  # optional per-client frame rate limit: /video_feed?fps=10
                min_period = 1.0 / float(params['fps']) if 'fps' in params else 0.0
            except (ValueError, ZeroDivisionError):
                min_period = 0.0
            try:
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, STREAM_SNDBUF)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            last = -1
            sent_at = 0.0
            try:
                while True:
                    # Always the newest frame: when the client is slow, the frames produced
                    # while the previous write was blocked are skipped.
                    frame, fid = app.camera.wait_frame(last)
                    if frame is None or fid == last:
                        continue
                    if min_period and time.monotonic() - sent_at < min_period:
                        last = fid
                        continue
                    last = fid
                    sent_at = time.monotonic()
                    self.wfile.write(
                        b'--' + boundary + b'\r\nContent-Type: image/jpeg\r\n'
                        + f'Content-Length: {len(frame)}\r\n\r\n'.encode() + frame + b'\r\n'
                    )  # fmt: skip
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            self.close_connection = True

    return Handler


def local_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # no packet is sent: only picks the outgoing interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith('127.'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080, help='HTTP port (default 8080)')
    ap.add_argument('--port-serial', help='ESP32 serial port (default: found automatically)')
    ap.add_argument('--sim', action='store_true', help='simulated ESP32 and camera')
    ap.add_argument(
        '--sim-drift',
        type=float,
        default=0.0,
        help='simulation: rad/s the rover turns by itself while moving (slope, grip)',
    )
    ap.add_argument('--camera', default='0', help='camera index or device (default 0)')
    ap.add_argument('--no-camera', action='store_true')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--quality', type=int, default=70, help='JPEG quality (default 70)')
    ap.add_argument(
        '--config', default=os.path.join(HERE, 'rover_config.json'), help='settings file'
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cam_kw = {'width': args.width, 'height': args.height, 'fps': args.fps, 'quality': args.quality}
    sim_board = None
    if args.sim:
        from fake_esp32 import FakeEsp32, FakeSerial

        sim_board = FakeEsp32(seed=int(time.time()), yaw_drift=args.sim_drift)
        rover = Rover(cfg, lambda: FakeSerial(sim_board))
        camera = SimCamera(sim_board, **cam_kw)
        print('Simulated ESP32 and camera.')
    else:
        port = args.port_serial or find_esp32_port()
        if port is None:
            raise SystemExit('No ESP32 serial port found (/dev/ttyUSB*, /dev/ttyACM*). '
                             'Is the USB cable plugged in? Or give --port-serial.')  # fmt: skip
        print(f'ESP32 on {port}. Opening the port reboots it: ~7 s, KEEP THE ROVER STILL.')
        rover = Rover(cfg, lambda: open_serial(port))
        camera = Camera(args.camera, **cam_kw)
    if args.no_camera:
        camera.error = 'camera disabled (--no-camera)'
        camera.request = lambda: None

    rover.start()
    app = App(rover, camera, args.config, sim_board)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print('Keep the rover still until the page shows "Gyro OK" (ESP32 boot + gyro bias).')

    def _terminate(signum, frame):  # systemctl stop/restart, kill: stop the rover as Ctrl-C does
        raise KeyboardInterrupt

    try:
        # Inside the try: a SIGTERM that comes before serve_forever still stops the rover.
        signal.signal(signal.SIGTERM, _terminate)
        for ip in local_ips() or ['<rover IP>']:
            print(f'Open http://{ip}:{args.port}')
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not cut the stop
        print('Stopping the rover.')
        rover.close()
        server.server_close()


if __name__ == '__main__':
    main()
