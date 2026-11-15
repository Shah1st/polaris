from __future__ import annotations

import threading
import time
from collections.abc import Callable
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource:
    """Read the newest frame from a webcam or video file in a background thread."""

    def __init__(
        self,
        source: int | str,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: cv2.VideoCapture | None = None
        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_frame: Callable[[np.ndarray], None] | None = None
        self._last_frame_time: float | None = None
        self._capture_fps = 0.0
        self._frame_count = 0
        self._fps_started = 0.0

    def start(self, on_frame: Callable[[np.ndarray], None] | None = None) -> None:
        if self._running:
            return
        self._capture = self._open_capture()
        if self._width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)
        self._on_frame = on_frame
        self._running = True
        self._thread = threading.Thread(target=self._run, name='camera-source', daemon=True)
        self._thread.start()
        logger.info('camera source started: %r', self._source)

    def _open_capture(self) -> cv2.VideoCapture:
        if self._source != 'auto':
            capture = cv2.VideoCapture(self._source)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f'cannot open video source: {self._source!r}')
            return capture

        for index in range(5):
            capture = cv2.VideoCapture(index)
            if not capture.isOpened():
                capture.release()
                continue
            ok, _ = capture.read()
            if ok:
                return capture
            capture.release()
        raise RuntimeError('no camera returned a frame; try --source with a camera index or file')

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        self._thread = None
        self._capture = None
        logger.info('camera source stopped')

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._condition:
            if self._latest is None:
                return self._sequence, None
            return self._sequence, self._latest.copy()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def fps(self) -> float:
        return self._capture_fps

    def _run(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                self._error = 'video source returned no frame'
                self._running = False
                break
            now = time.monotonic()
            self._frame_count += 1
            if self._fps_started == 0.0:
                self._fps_started = now
            elapsed = now - self._fps_started
            if elapsed >= 1.0:
                self._capture_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_started = now
                logger.info('capture: %.1f fps', self._capture_fps)
            with self._condition:
                self._latest = frame
                self._sequence += 1
                self._last_frame_time = now
                self._condition.notify_all()
            if self._on_frame is not None:
                self._on_frame(frame)
        if self._error is not None:
            logger.error('%s', self._error)
