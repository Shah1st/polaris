from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class FrameStabilizer(Protocol):
    """Minimal interface required by the processing pipeline."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        ...

    def reset(self) -> None:
        ...


class VidStabAdapter:
    """Adapter around python_video_stab with a frame-based interface."""

    def __init__(
        self,
        smoothing_window: int = 24,
        border_type: str = 'black',
        zoom: float = 1.35,
    ) -> None:
        try:
            from vidstab import VidStab
        except ImportError as exc:
            raise RuntimeError(
                'python_video_stab is not installed; run pip install -r requirements.txt'
            ) from exc
        if smoothing_window < 1:
            raise ValueError('smoothing_window must be positive')
        if zoom < 1.0:
            raise ValueError('zoom must be at least 1.0')
        self._vidstab_type = VidStab
        self._smoothing_window = smoothing_window
        self._border_type = border_type
        self._zoom = zoom
        self._stabilizer = self._new_stabilizer()

    def process(self, frame: np.ndarray) -> np.ndarray:
        original_height, original_width = frame.shape[:2]
        working_frame = self._zoom_frame(frame)
        stabilized = self._stabilizer.stabilize_frame(
            input_frame=working_frame,
            smoothing_window=self._smoothing_window,
            border_type=self._border_type,
            border_size=0,
        )
        if stabilized is None:
            return frame.copy()
        return self._center_crop(stabilized, original_width, original_height)

    def reset(self) -> None:
        self._stabilizer = self._new_stabilizer()

    def _new_stabilizer(self):
        return self._vidstab_type()

    def _zoom_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._zoom == 1.0:
            return frame
        height, width = frame.shape[:2]
        return cv2.resize(
            frame,
            (round(width * self._zoom), round(height * self._zoom)),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def _center_crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        if frame_width < width or frame_height < height:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        x0 = (frame_width - width) // 2
        y0 = (frame_height - height) // 2
        return frame[y0 : y0 + height, x0 : x0 + width].copy()
