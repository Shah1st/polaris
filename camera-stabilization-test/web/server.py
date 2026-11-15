from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
from stabilization.pipeline import StabilizationPipeline

BOUNDARY = b'--frame'
INDEX = Path(__file__).with_name('index.html')
logger = logging.getLogger(__name__)


class WebServer:
    """Serve the comparison page and two independent MJPEG streams."""

    def __init__(self, pipeline: StabilizationPipeline, host: str = '127.0.0.1', port: int = 8085):
        self.pipeline = pipeline
        self._server = ThreadingHTTPServer((host, port), self._handler())

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address

    def serve_forever(self) -> None:
        try:
            logger.info('web server listening on http://%s:%d', *self.address)
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def shutdown(self) -> None:
        self._server.shutdown()

    def stop(self) -> None:
        logger.info('stop requested through HTTP')
        self.pipeline.stop()
        threading.Thread(target=self._server.shutdown, name='web-shutdown', daemon=True).start()

    def _handler(self):
        pipeline = self.pipeline
        owner_stop = self.stop

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == '/':
                    self._send_bytes(HTTPStatus.OK, INDEX.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/status':
                    payload = json.dumps(pipeline.status().__dict__).encode()
                    self._send_bytes(HTTPStatus.OK, payload, 'application/json')
                elif path in ('/video/raw', '/video/stabilized'):
                    self._stream(path.rsplit('/', 1)[-1])
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                if urlparse(self.path).path != '/api/stop':
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(HTTPStatus.ACCEPTED, b' stopping\n', 'text/plain; charset=utf-8')
                threading.Thread(target=owner_stop, name='web-stop', daemon=True).start()

            def _stream(self, view: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                sequence = 0
                try:
                    while pipeline.status().running:
                        packet = pipeline.wait_for_new(sequence, timeout=2.0)
                        if packet is None:
                            continue
                        sequence = packet.sequence
                        image = packet.raw if view == 'raw' else packet.stabilized
                        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok:
                            continue
                        self.wfile.write(
                            BOUNDARY
                            + b'\r\nContent-Type: image/jpeg\r\n\r\n'
                            + encoded.tobytes()
                            + b'\r\n'
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler
