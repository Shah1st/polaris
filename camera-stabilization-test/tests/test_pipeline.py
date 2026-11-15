from __future__ import annotations

import time

import numpy as np
import pytest

from stabilization.pipeline import StabilizationPipeline
from stabilization.source import FrameSource
from stabilization.stabilizer import VidStabAdapter


class FakeSource(FrameSource):
    def __init__(self) -> None:
        super().__init__('unused')

    def start(self, on_frame=None) -> None:
        self._running = True
        self._latest = np.zeros((4, 5, 3), dtype=np.uint8)
        self._sequence = 1

    def stop(self) -> None:
        self._running = False


class FakeStabilizer:
    def reset(self) -> None:
        return

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame + 1


def test_pipeline_publishes_raw_and_stabilized_frames() -> None:
    pipeline = StabilizationPipeline(FakeSource(), FakeStabilizer())
    pipeline.start()
    try:
        packet = pipeline.wait_for_new(0)
        assert packet is not None
        assert np.all(packet.raw == 0)
        assert np.all(packet.stabilized == 1)
    finally:
        pipeline.stop()


def test_pipeline_stops_cleanly() -> None:
    pipeline = StabilizationPipeline(FakeSource(), FakeStabilizer())
    pipeline.start()
    time.sleep(0.01)
    pipeline.stop()
    assert not pipeline.status().running


def test_stabilizer_rejects_small_zoom() -> None:
    pytest.importorskip('vidstab')
    with pytest.raises(ValueError, match='zoom'):
        VidStabAdapter(zoom=0.9)
