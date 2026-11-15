"""Read CamStream (the iPhone app, iphone-camera/) with the pose of each picture.

In ARKit mode the app adds an `X-Pose` header to every picture of `/stream` (and to
`/snapshot.jpg`): one line of JSON, the camera's pose when that very picture was taken (see
README.md). Picture and pose travel together, so a late picture is still placed correctly.

    stream = PoseStream('http://172.20.10.6:8080')   # a thread; keeps the newest picture only
    frame = stream.latest(max_age=1.0)               # None when nothing fresh
    frame.image, frame.pose, frame.received

Standard library only for the network (urllib), OpenCV to decode the JPEG.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from pose_camera import Pose, PoseError


@dataclass
class Frame:
    jpeg: bytes
    pose: Pose | None  # None: the app is not in ARKit mode, or its tracking is not ready
    pose_error: str | None
    received: float  # time.monotonic() when the picture was complete here
    seq: int
    _image: np.ndarray | None = None

    @property
    def image(self) -> np.ndarray | None:
        """The decoded picture (BGR), decoded once, only if someone asks for it."""
        if self._image is None:
            self._image = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        return self._image


def parse_pose(header: str | None) -> tuple[Pose | None, str | None]:
    """(pose, None), or (None, why not) for a missing or unusable X-Pose header."""
    if not header:
        return None, 'no X-Pose header: switch on "Position (ARKit)" in the app\'s settings'
    try:
        d = json.loads(header)
        if d.get('track') == 'none':
            return None, 'ARKit is not tracking yet: move the phone a little'
        return Pose.from_arkit(d), None
    except (ValueError, PoseError) as e:
        return None, f'unusable X-Pose header: {e}'


def read_parts(stream, boundary: bytes):
    """Yield (headers, body) for each part of a multipart/x-mixed-replace stream that gives a
    Content-Length (CamStream does). `stream` has readline() and read(n)."""
    marker = b'--' + boundary
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.startswith(marker):
            continue
        headers = {}
        while True:
            line = stream.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                break
            name, _, value = line.decode('latin-1').partition(':')
            headers[name.strip().lower()] = value.strip()
        try:
            length = int(headers['content-length'])
        except (KeyError, ValueError):
            continue  # a part without a length: skip to the next boundary
        body = stream.read(length)
        if len(body) < length:
            return
        yield headers, body


def snapshot(base_url: str, timeout: float = 3.0) -> Frame:
    """One picture and its pose from /snapshot.jpg."""
    with urllib.request.urlopen(base_url.rstrip('/') + '/snapshot.jpg', timeout=timeout) as r:
        jpeg = r.read()
        pose, error = parse_pose(r.headers.get('X-Pose'))
    return Frame(jpeg, pose, error, time.monotonic(), 0)


class PoseStream:
    """Reads /stream in a thread and keeps the newest picture; reconnects by itself."""

    def __init__(self, base_url: str, timeout: float = 5.0, start: bool = True) -> None:
        self.url = base_url.rstrip('/') + '/stream'
        self.timeout = timeout
        self.error: str | None = 'not connected yet'
        self.fps = 0.0
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self._thread.start()

    def latest(self, max_age: float | None = None) -> Frame | None:
        with self._lock:
            frame = self._frame
        if frame is None or (max_age is not None and time.monotonic() - frame.received > max_age):
            return None
        return frame

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                    ctype = r.headers.get('Content-Type', '')
                    if 'boundary=' not in ctype:
                        raise ValueError(f'not an MJPEG stream ({ctype or "no content type"})')
                    boundary = ctype.split('boundary=')[1].strip().strip('"').encode()
                    last = time.monotonic()
                    for headers, body in read_parts(r, boundary):
                        if self._stop.is_set():
                            return
                        now = time.monotonic()
                        pose, error = parse_pose(headers.get('x-pose'))
                        seq += 1
                        with self._lock:
                            self._frame = Frame(body, pose, error, now, seq)
                        self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
                        last = now
                        self.error = None
                self.error = 'the phone closed the stream'
            except (OSError, ValueError, urllib.error.URLError) as e:
                self.error = f'{type(e).__name__}: {e}'
            self._stop.wait(1.0)
