"""A fake phone for the tests and for the demo without hardware: it films the SIMULATED rover
of rover/rover_server.py from a camera that keeps moving, and serves the pictures the way
CamStream does in ARKit mode (MJPEG, one X-Pose header per picture)."""

from __future__ import annotations

import math
import time
from http.server import BaseHTTPRequestHandler

import cv2
import synthetic as syn
from pose_camera import Pose

BOUNDARY = 'camstreamframe'
# The floor frame is NOT the simulation's: turned by 40 degrees and shifted. The guide must
# find that out by itself, from the pictures.
TURN, SHIFT = math.radians(40), (0.2, -0.3)
SIDE, HEIGHT = 0.10, 0.12


def floor_pose(board) -> tuple[float, float, float]:
    c, s = math.cos(TURN), math.sin(TURN)
    return (
        c * board.x - s * board.y + SHIFT[0],
        s * board.x + c * board.y + SHIFT[1],
        board.yaw + TURN,
    )


def make_phone(app, floor):
    printed = syn.marker_image(0)
    t0 = time.monotonic()

    class Phone(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.end_headers()
            try:
                while True:
                    t = time.monotonic() - t0
                    x, y, heading = floor_pose(app.sim_board)
                    # a hand-held phone: it circles above the area and never looks straight down
                    eye = (0.3 + 0.25 * math.cos(0.7 * t), 0.25 * math.sin(0.7 * t), 1.9)
                    d = syn.arkit_pose(eye, (0.3, 0.0, 0.0), roll_deg=20 * math.sin(0.4 * t), t=t)
                    d['sid'] = 7
                    marker = (printed, x, y, heading - math.pi / 2, SIDE * 1.5, HEIGHT)
                    image = syn.render(Pose.from_arkit(d), floor, marker)
                    ok, jpg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    body = jpg.tobytes()
                    head = (
                        f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                        f'Content-Length: {len(body)}\r\nX-Pose: {syn.header(d)}\r\n\r\n'
                    )
                    self.wfile.write(head.encode() + body + b'\r\n')
                    time.sleep(0.05)
            except OSError:
                pass

    return Phone
