"""Pretend to be the rover: serve a video file as MJPEG on /video_feed (for tests without the robot).

    python scripts/fake_rover.py video.mp4 [port]
    python run.py --source http://127.0.0.1:5000/video_feed
"""
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

video = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/video_feed":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    jpg = cv2.imencode(".jpg", bgr)[1].tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / fps)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
