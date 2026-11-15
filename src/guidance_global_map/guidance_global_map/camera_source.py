"""Live overhead camera: keep only the newest frame of a camera, a stream or a file. No ROS.

    read = camera_reader('http://192.168.1.20:8080/video')   # MJPEG stream of a phone app
    image = read()                  # newest BGR frame, or None when it is older than max_age
    image = read.read_new()         # a frame taken after this call (waits, None on timeout)
    read.close()

`source` is one of:

- an int or a digit string: a local camera index for `cv2.VideoCapture`;
- an `http(s)://` URL: an MJPEG stream (`multipart/x-mixed-replace`) or a single-JPEG
  "snapshot" URL, told apart by the reply's Content-Type; a snapshot URL is polled;
- an `rtsp://` URL (or any other URL scheme OpenCV's FFmpeg backend opens);
- a file path: a still image, read again at each call, like `image_file_reader`.

A background thread reads continuously, so the camera's or the network's buffer never holds
old frames: the frame handed out is always the newest one received.
"""

import logging
import re
import threading
import time
import urllib.request
from collections.abc import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Delay between a frame being exposed and it being received, for streamed sources. A frame's
# time is its arrival time minus this, so that "a frame taken after t" does not pick a frame
# that was already on its way when the rover stopped. Phone apps over Wi-Fi: 0.1-0.5 s.
DEFAULT_LATENCY = 0.2
RECONNECT_DELAY = 0.5
SNAPSHOT_MIN_PERIOD = 0.05
_SOI, _EOI = b'\xff\xd8', b'\xff\xd9'


def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR image, or None when they are not a readable image."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _read_some(stream, size: int) -> bytes:
    # read1 returns what has arrived instead of blocking until `size` bytes are there.
    return stream.read1(size) if hasattr(stream, 'read1') else stream.read(size)


def iter_mjpeg(stream, boundary: bytes | None = None, chunk: int = 65536):
    """Yield the JPEG bytes of each part of a multipart MJPEG stream (a binary file object).

    Parts with a Content-Length are read exactly; otherwise a frame runs from the JPEG start
    marker to the next end marker.
    """
    buffer = b''
    while True:
        start = buffer.find(_SOI)
        if start >= 0:
            header = buffer[:start]
            length = re.search(rb'content-length:\s*(\d+)', header, re.IGNORECASE)
            if length is not None:
                size = int(length.group(1))
                while len(buffer) < start + size:
                    data = _read_some(stream, chunk)
                    if not data:
                        return
                    buffer += data
                yield buffer[start : start + size]
                buffer = buffer[start + size :]
                continue
            end = buffer.find(_EOI, start + 2)
            if end >= 0:
                yield buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                continue
        elif len(buffer) > 4096:
            # Keep only a tail that may hold the part's headers and the start marker.
            buffer = buffer[-4096:]
        data = _read_some(stream, chunk)
        if not data:
            return
        buffer += data


class CameraReader:
    """Callable that returns the newest frame of a live source; see `camera_reader`."""

    def __init__(
        self,
        source: int | str,
        max_age: float = 2.0,
        latency: float = DEFAULT_LATENCY,
        timeout: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.max_age = max_age
        self.latency = latency
        self._timeout = timeout
        self._clock = clock
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None  # newest MJPEG frame, decoded only when asked for
        self._frame_time: float | None = None
        self._closed = threading.Event()
        self.frames = 0  # frames received, for diagnostics and tests
        self._thread = threading.Thread(target=self._run, name='camera_reader', daemon=True)
        self._thread.start()

    # ---- what callers use ---------------------------------------------------------------

    def __call__(self) -> np.ndarray | None:
        """Newest frame, or None when there is none or it is older than `max_age`."""
        with self._condition:
            if self._frame_time is None or self._clock() - self._frame_time > self.max_age:
                return None
            return self._decoded()

    def read_new(self, timeout: float = 2.0) -> np.ndarray | None:
        """Wait for a frame taken after this call; None when none comes within `timeout`."""
        since = self._clock()
        deadline = since + timeout
        with self._condition:
            while self._frame_time is None or self._frame_time <= since:
                remaining = deadline - self._clock()
                if remaining <= 0 or self._closed.is_set():
                    return None
                self._condition.wait(min(remaining, 0.1))
            return self._decoded()

    @property
    def frame_time(self) -> float | None:
        """Estimated capture time of the newest frame, on `clock`."""
        with self._condition:
            return self._frame_time

    def close(self) -> None:
        self._closed.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    # ---- the reading thread -------------------------------------------------------------

    def _decoded(self) -> np.ndarray | None:
        if self._frame is None and self._jpeg is not None:
            self._frame = decode_jpeg(self._jpeg)
            self._jpeg = None
        return None if self._frame is None else self._frame.copy()

    def _store(self, frame: np.ndarray | None, jpeg: bytes | None, taken: float) -> None:
        with self._condition:
            self._frame, self._jpeg, self._frame_time = frame, jpeg, taken
            self.frames += 1
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                if isinstance(self.source, str) and self.source.lower().startswith(
                    ('http://', 'https://')
                ):
                    self._run_http(self.source)
                else:
                    self._run_capture(self.source)
            except Exception as error:  # a live camera must survive network and driver errors
                log.warning('camera %s: %s; reconnecting', self.source, error)
            self._closed.wait(RECONNECT_DELAY)

    def _run_capture(self, source: int | str) -> None:
        capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                raise OSError('cannot open the camera')
            # Not every backend honours it; reading continuously drains the buffer anyway.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self._closed.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError('the camera stopped giving frames')
                self._store(frame, None, self._clock() - self.latency)
        finally:
            capture.release()

    def _run_http(self, url: str) -> None:
        while not self._closed.is_set():
            asked = self._clock()
            with urllib.request.urlopen(url, timeout=self._timeout) as reply:
                kind = reply.headers.get('Content-Type', '')
                if kind.lower().startswith('multipart'):
                    for jpeg in iter_mjpeg(reply):
                        if self._closed.is_set():
                            return
                        self._store(None, jpeg, self._clock() - self.latency)
                    raise OSError('the MJPEG stream ended')
                image = decode_jpeg(reply.read())
            if image is None:
                raise OSError(f'not an image ({kind or "no Content-Type"})')
            # An app answers with its last captured frame: up to `latency` before the request.
            self._store(image, None, asked - self.latency)
            self._closed.wait(max(0.0, SNAPSHOT_MIN_PERIOD - (self._clock() - asked)))


class StillImageReader:
    """A still image file, read again at each call; same interface as `CameraReader`."""

    def __init__(self, path: str) -> None:
        self.source = path

    def __call__(self) -> np.ndarray | None:
        return cv2.imread(str(self.source))

    def read_new(self, timeout: float = 2.0) -> np.ndarray | None:
        return self()

    def close(self) -> None:
        pass


def camera_reader(
    source: int | str, max_age: float = 2.0, latency: float = DEFAULT_LATENCY
) -> CameraReader | StillImageReader:
    """Return `read() -> BGR image | None` for a live source, with `.close()`.

    See the module docstring for the sources. `read()` never returns a frame older than
    `max_age` seconds; `read.read_new()` waits for a frame taken after the call.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        return CameraReader(int(source), max_age, latency)
    if isinstance(source, str) and '://' in source:
        return CameraReader(source, max_age, latency)
    return StillImageReader(str(source))
