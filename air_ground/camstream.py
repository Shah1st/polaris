"""Read CamStream (the iPhone app, iphone-camera/) with the pose of each picture, and its
LiDAR depth with the pose of each depth map (Pro iPhones).

In ARKit mode the app adds an `X-Pose` header to every picture of `/stream` (and to
`/snapshot.jpg`): one line of JSON, the camera's pose when that very picture was taken (see
README.md). Picture and pose travel together, so a late picture is still placed correctly.

    stream = PoseStream('http://172.20.10.6:8080')   # a thread; keeps the newest picture only
    frame = stream.latest(max_age=1.0)               # None when nothing fresh
    frame.image, frame.pose, frame.received

    depth = read_depth('http://172.20.10.6:8080')    # one LiDAR depth map and its pose
    depth.metres, depth.k, depth.pose

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


class DepthUnavailableError(RuntimeError):
    """The phone gives no LiDAR depth (no LiDAR, depth switched off, or not started yet)."""


@dataclass
class DepthMap:
    metres: np.ndarray  # float32 (h, w): distance along the optical axis; 0 = no measure
    k: tuple[float, float, float, float]  # fx, fy, cx, cy in pixels of the depth map
    pose: Pose | None  # None outside the ARKit mode: the depth cannot be placed on the floor
    pose_error: str | None
    age_ms: int | None  # how old the depth was when the phone sent it
    received: float  # time.monotonic() here


def read_depth(base_url: str, timeout: float = 2.0) -> DepthMap:
    """One LiDAR depth map from /depth, with its pose (X-Pose) in the ARKit mode."""
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/depth', timeout=timeout) as r:
            body = r.read()
            h = r.headers
    except urllib.error.HTTPError as e:
        if e.code == 503:
            raise DepthUnavailableError(e.read().decode('utf-8', 'replace')[:200]) from e
        raise
    width, height = int(h['X-Depth-Width']), int(h['X-Depth-Height'])
    if len(body) != width * height * 4:
        raise ValueError(f'depth of {len(body)} bytes for {width} x {height}')
    k = h.get('X-Depth-Intrinsics')
    if not k:
        raise ValueError('depth without intrinsics (X-Depth-Intrinsics)')
    fx, fy, cx, cy = (float(v) for v in k.split(','))
    pose, error = parse_pose(h.get('X-Pose'))
    age = h.get('X-Age-Ms')
    return DepthMap(
        np.frombuffer(body, '<f4').reshape(height, width).copy(),
        (fx, fy, cx, cy),
        pose,
        error,
        int(age) if age else None,
        time.monotonic(),
    )


class DepthPoller:
    """Asks the phone for its newest depth map a few times a second, in a thread; keeps the
    newest. Without LiDAR depth it asks again every few seconds only."""

    def __init__(self, base_url: str, period: float = 0.2, start: bool = True) -> None:
        self.url = base_url.rstrip('/')
        self.period = period
        self.error: str | None = 'not asked yet'
        self.count = 0
        self._depth: DepthMap | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def latest(self, max_age: float | None = None) -> DepthMap | None:
        with self._lock:
            d = self._depth
        if d is None or (max_age is not None and time.monotonic() - d.received > max_age):
            return None
        return d

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last_t = None
        while not self._stop.is_set():
            wait = self.period
            try:
                d = read_depth(self.url)
                t = None if d.pose is None else d.pose.t
                if t is None or t != last_t:  # a new depth map, not the same one again
                    last_t = t
                    with self._lock:
                        self._depth = d
                    self.count += 1
                self.error = d.pose_error
            except DepthUnavailableError as e:
                self.error = f'no LiDAR depth: {e}'
                wait = 3.0
            except (OSError, ValueError, KeyError) as e:
                self.error = f'{type(e).__name__}: {e}'
                wait = 1.0
            self._stop.wait(wait)
