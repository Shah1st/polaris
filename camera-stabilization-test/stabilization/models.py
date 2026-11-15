from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """A raw/stabilized frame pair produced by the processing pipeline."""

    sequence: int
    timestamp: float
    raw: np.ndarray
    stabilized: np.ndarray
    processing_ms: float


@dataclass(frozen=True)
class PipelineStatus:
    """Current processing metrics exposed by the web UI."""

    running: bool
    sequence: int
    capture_fps: float
    processing_fps: float
    processing_ms: float
    error: str | None
