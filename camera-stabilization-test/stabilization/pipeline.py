from __future__ import annotations

import threading
import time
import logging

import numpy as np

from .models import FramePacket, PipelineStatus
from .source import FrameSource
from .stabilizer import FrameStabilizer

logger = logging.getLogger(__name__)


class StabilizationPipeline:
    """Process the newest camera frames and publish raw/stabilized pairs."""

    def __init__(self, source: FrameSource, stabilizer: FrameStabilizer) -> None:
        self._source = source
        self._stabilizer = stabilizer
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._processing_fps = 0.0
        self._processing_ms = 0.0
        self._error: str | None = None
        self._processed_count = 0
        self._fps_started = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._source.start()
        self._stabilizer.reset()
        self._running = True
        self._thread = threading.Thread(target=self._run, name='stabilization-pipeline', daemon=True)
        self._thread.start()
        logger.info('stabilization pipeline started')

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._source.stop()
        self._thread = None
        logger.info('stabilization pipeline stopped')

    def latest(self) -> FramePacket | None:
        with self._condition:
            return self._latest

    def wait_for_new(self, sequence: int, timeout: float = 2.0) -> FramePacket | None:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._latest is not None and self._latest.sequence > sequence,
                timeout,
            )
            return self._latest if ready else None

    def status(self) -> PipelineStatus:
        return PipelineStatus(
            running=self._running,
            sequence=self._sequence,
            capture_fps=self._source.fps,
            processing_fps=self._processing_fps,
            processing_ms=self._processing_ms,
            error=self._error or self._source.error,
        )

    def _run(self) -> None:
        last_source_sequence = 0
        while self._running:
            source_sequence, frame = self._source.latest()
            if frame is None or source_sequence == last_source_sequence:
                time.sleep(0.001)
                continue
            last_source_sequence = source_sequence
            started = time.perf_counter()
            try:
                stabilized = self._stabilizer.process(frame)
            except Exception as exc:
                self._error = f'{type(exc).__name__}: {exc}'
                logger.exception('stabilization failed')
                self._running = False
                break
            elapsed = time.perf_counter() - started
            now = time.monotonic()
            self._processed_count += 1
            if self._fps_started == 0.0:
                self._fps_started = now
            window = now - self._fps_started
            if window >= 1.0:
                self._processing_fps = self._processed_count / window
                self._processed_count = 0
                self._fps_started = now
                logger.info(
                    'processing: %.1f fps, %.1f ms/frame',
                    self._processing_fps,
                    self._processing_ms,
                )
            self._processing_ms = elapsed * 1000.0
            self._sequence += 1
            packet = FramePacket(
                sequence=self._sequence,
                timestamp=now,
                raw=frame,
                stabilized=stabilized,
                processing_ms=self._processing_ms,
            )
            with self._condition:
                self._latest = packet
                self._condition.notify_all()
