"""Reading the phone's stream with the pose of each picture, against a fake CamStream."""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import synthetic as syn  # noqa: E402
from camstream import PoseStream, parse_pose, read_parts, snapshot  # noqa: E402

BOUNDARY = 'camstreamframe'


def jpeg(value: int) -> bytes:
    ok, data = cv2.imencode('.jpg', np.full((72, 128, 3), value, np.uint8))
    return data.tobytes()


def part(body: bytes, pose: str | None) -> bytes:
    head = f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n'
    if pose is not None:
        head += f'X-Pose: {pose}\r\n'
    return head.encode() + b'\r\n' + body + b'\r\n'


class FakeCamStream(BaseHTTPRequestHandler):
    arkit = True

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        if self.path == '/snapshot.jpg':
            body = jpeg(200)
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(body)))
            if self.arkit:
                self.send_header('X-Pose', syn.header(syn.arkit_pose((0, 0, 1.8), (0, 0, 0))))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
        self.end_headers()
        try:
            for i in range(200):
                d = syn.arkit_pose((0.01 * i, 0, 1.8), (0.01 * i, 0, 0), t=float(i))
                self.wfile.write(part(jpeg(i), syn.header(d) if self.arkit else None))
                time.sleep(0.02)
        except OSError:
            pass


@pytest.fixture
def phone():
    FakeCamStream.arkit = True
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeCamStream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()
    srv.server_close()


def test_parts_with_and_without_pose() -> None:
    d = syn.arkit_pose((0.2, 0.1, 1.8), (0.2, 0.1, 0))
    raw = b'junk\r\n' + part(b'AAAA', syn.header(d)) + part(b'BB', None) + part(b'CCC', '{bad')
    parts = list(read_parts(io.BytesIO(raw), BOUNDARY.encode()))
    assert [body for _, body in parts] == [b'AAAA', b'BB', b'CCC']
    pose, error = parse_pose(parts[0][0].get('x-pose'))
    assert error is None and pose.c == pytest.approx([0.2, 0.1, 1.8])
    assert parse_pose(parts[1][0].get('x-pose')) == (None, parse_pose(None)[1])
    assert 'ARKit' in parse_pose(None)[1]
    assert 'unusable' in parse_pose(parts[2][0].get('x-pose'))[1]
    d['track'] = 'none'
    assert 'not tracking' in parse_pose(syn.header(d))[1]
    # a stream cut in the middle of a picture gives nothing more, and no exception
    assert list(read_parts(io.BytesIO(part(b'AAAA', None)[:-8]), BOUNDARY.encode())) == []


def test_stream_keeps_the_newest_picture_with_its_own_pose(phone) -> None:
    stream = PoseStream(phone)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (stream.latest() is None or stream.latest().seq < 5):
            time.sleep(0.02)
        frame = stream.latest(max_age=1.0)
        assert frame is not None and stream.error is None and frame.pose_error is None
        # the picture's grey level is its number, and so is its pose's time: they match
        level = int(frame.image[36, 64, 0])
        assert abs(level - frame.pose.t) <= 3
        assert frame.pose.c[0] == pytest.approx(0.01 * frame.pose.t)
        assert stream.fps > 5
    finally:
        stream.close()
    time.sleep(0.1)
    assert stream.latest(max_age=0.0) is None  # too old: nothing fresh


def test_snapshot_and_a_phone_without_arkit(phone) -> None:
    frame = snapshot(phone)
    assert frame.pose.c[2] == pytest.approx(1.8) and frame.image.shape == (72, 128, 3)
    FakeCamStream.arkit = False
    frame = snapshot(phone)
    assert frame.pose is None and 'ARKit' in frame.pose_error
    dead = PoseStream('http://127.0.0.1:9', timeout=0.5)
    time.sleep(0.8)
    assert dead.latest() is None and dead.error and dead.error != 'not connected yet'
    dead.close()
