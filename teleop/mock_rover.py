"""A stand-in for rover/rover_server.py, for the tests and the demo of the mission page.

Records every POST (time, path, body), answers GET /state, and serves a generated picture on
/snapshot.jpg (or 503, like a robot without a camera frame). It never moves anything.
MockPhone stands in for the phone camera app of the drone camera (server.py --drone).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


class MockRover:
    def __init__(self, port: int, snapshots: bool = True):
        self.log: list[tuple[float, str, dict]] = []
        self.gets: list[str] = []  # every GET path, with its query
        self.snapshots = snapshots
        self.lock = threading.Lock()
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _reply(self, code, body=b'{"ok": true}', ctype='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with mock.lock:
                    mock.gets.append(self.path)
                if self.path.startswith('/d405/status'):
                    return self._reply(200, b'{"ok": true, "camera": {"name": "RealSense D405"}}')
                if self.path.startswith(('/d405/color.jpg', '/d405/depth.jpg')):
                    return self._reply(200, picture('D405 (simulee)'), 'image/jpeg')
                if self.path.startswith('/state'):
                    return self._reply(
                        200,
                        b'{"link": {"connected": true}, "battery_v": 11.4, '
                        b'"gimbal": {"pan": 0.0, "tilt": 0.0, "moving": false}}',
                    )
                if self.path.startswith('/snapshot.jpg') and mock.snapshots:
                    time.sleep(0.2)  # about the rate of the real camera's snapshots
                    return self._reply(200, picture('camera du rover (simulee)'), 'image/jpeg')
                self._reply(503, b'{"error": "no frame"}')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
                try:
                    body = json.loads(raw or b'{}')
                except ValueError:
                    body = {'_raw': raw.decode(errors='replace')}
                with mock.lock:
                    mock.log.append((time.monotonic(), self.path, body))
                self._reply(200)

        self.srv = ThreadingHTTPServer(('127.0.0.1', port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{port}'

    def moving(self, after: float = 0.0) -> list:
        """The /cmd with a non-zero speed received after `after` (time.monotonic)."""
        with self.lock:
            return [
                e
                for e in self.log
                if e[0] >= after and e[1] == '/cmd' and (e[2].get('v') or e[2].get('omega'))
            ]

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


class MockPhone:
    """A stand-in for a phone camera app: an MJPEG stream on /video (about 10 pictures per
    second) and one JPEG per request on /shot.jpg. Wrong URLs for the tests: /locked (401,
    password needed) and /clip.mp4 (an endless video file)."""

    def __init__(self, port: int, text: str = 'camera du drone (iPhone simule)'):
        self.readers = 0  # stream readers so far
        phone = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith('/locked'):  # an app with a password, not given
                    self.send_response(401)
                    self.send_header('WWW-Authenticate', 'Basic realm="camera"')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if self.path.startswith('/clip.mp4'):  # a video file, not pictures
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    try:
                        while True:
                            self.wfile.write(b'\x00' * 65536)
                            time.sleep(0.01)
                    except OSError:  # the reader left
                        pass
                    self.close_connection = True
                    return
                if self.path.startswith('/shot.jpg'):
                    body = picture(text, 480, 360)
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not self.path.startswith('/video'):
                    self.send_error(404)
                    return
                phone.readers += 1
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=ipcam')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    while True:
                        jpeg = picture(text, 480, 360)
                        self.wfile.write(
                            b'--ipcam\r\nContent-Type: image/jpeg\r\nContent-Length: '
                            + str(len(jpeg)).encode()
                            + b'\r\n\r\n'
                            + jpeg
                            + b'\r\n'
                        )
                        time.sleep(0.1)
                except OSError:  # the reader left
                    pass
                self.close_connection = True

        self.srv = ThreadingHTTPServer(('127.0.0.1', port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{port}'

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def picture(text: str, w: int = 640, h: int = 480) -> bytes:
    """A JPEG with `text` and the time on it, so a frozen picture shows."""
    img = np.full((h, w, 3), (60, 50, 40), np.uint8)
    cv2.rectangle(img, (40, 300), (600, 470), (70, 110, 70), -1)
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(
        img, time.strftime('%H:%M:%S'), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 220, 255), 3
    )
    return cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
