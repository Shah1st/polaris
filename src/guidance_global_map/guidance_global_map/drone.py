"""The overhead camera as the "drone" of the map fusion (`merging_map`). No ROS.

`merging_map` takes the drone's map as tiles and asks the drone side for three things. This
module provides them from the overhead pipeline:

    camera = OverheadDrone(camera_reader('http://phone:8080/video'), settings)
    for message in camera.tiles():          # first survey
        carte.recevoir(message)
    rover = RaspRoverReel(carte, ..., position_drone=camera.rover_pose, drone=camera)

(`merging_map/lancer_overhead.lancer` does exactly this.)

- `drone.disponible()` and `drone.zoomer(tile)`: methods of `OverheadDrone`;
- `position_drone()`: `OverheadDrone.rover_pose` (heading None without a marker; the rover
  adapter then measures the heading from the rover's motion).

Sources: `camera_reader` (live camera, stream, snapshot URL or still file; see
`camera_source.py`; `lancer_overhead.py --camera` uses it), `LiveCamera` (a stream read with
`cv2.VideoCapture`) or `image_file_reader` (a file read again at each call). Only
`camera_reader` has `read_new`, the frame taken after a call.

Tile format, from `merging_map/README.md`: N x N cells (N = 32), values 1 obstacle, 0 free and
seen, -1 not seen, row 0 = south, tile (z, x, y) covering x*T <= X < (x+1)*T with T = N * cell
size. The grid here has 5 cm cells, which is level z = 4 of the fusion's defaults.
"""

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from guidance_global_map.camera_source import camera_reader as camera_reader
from guidance_global_map.pipeline import (
    OverheadResult,
    OverheadSettings,
    brightness_spread,
    frame_problem,
    process_image,
)
from guidance_global_map.scale_check import Seen
from guidance_global_map.segmentation import grid_to_fusion

UNSEEN = -1
# Seconds after which a frame that only fails the comparison with the last good frame (the
# scene's contrast collapsed) becomes the new reference instead of being rejected for ever.
RELEARN_AFTER = 10.0
log = logging.getLogger(__name__)


def cut_tiles(
    matrix: np.ndarray,
    cell_size: float,
    level: int,
    x0: float = 0.0,
    y0: float = 0.0,
    tile_cells: int = 32,
    source: str = 'drone',
    t: float = 0.0,
) -> list[dict]:
    """Cut a dense -1/0/1 matrix into the fusion's tile messages.

    `matrix[i][j]` is the cell at X = x0 + j * cell_size, Y = y0 + i * cell_size. Tiles are
    aligned to the global grid; cells outside the matrix are -1, and tiles that are all -1 are
    left out. Same result as `decouper()` in `merging_map/fusion_carte.py`.
    """
    matrix = np.asarray(matrix, dtype=np.int8)
    n = tile_cells
    row0, col0 = round(y0 / cell_size), round(x0 / cell_size)
    rows, cols = matrix.shape
    messages = []
    for ty in range(math.floor(row0 / n), math.floor((row0 + rows - 1) / n) + 1):
        for tx in range(math.floor(col0 / n), math.floor((col0 + cols - 1) / n) + 1):
            tile = np.full((n, n), UNSEEN, dtype=np.int8)
            r0, r1 = max(ty * n, row0), min((ty + 1) * n, row0 + rows)
            c0, c1 = max(tx * n, col0), min((tx + 1) * n, col0 + cols)
            tile[r0 - ty * n : r1 - ty * n, c0 - tx * n : c1 - tx * n] = matrix[
                r0 - row0 : r1 - row0, c0 - col0 : c1 - col0
            ]
            if (tile != UNSEEN).any():
                messages.append(
                    {'source': source, 't': t, 'z': level, 'x': tx, 'y': ty, 'data': tile}
                )
    return messages


def unsee_disc(
    matrix: np.ndarray, cell_size: float, center: tuple[float, float], radius: float
) -> np.ndarray:
    """Copy of a fusion matrix (row 0 at y = 0) with the cells near `center` set to -1.

    A cell is cleared when its centre is within `radius` plus half a cell of `center`.
    """
    matrix = np.array(matrix, dtype=np.int8, copy=True)
    rows, cols = matrix.shape
    ys = (np.arange(rows) + 0.5) * cell_size
    xs = (np.arange(cols) + 0.5) * cell_size
    near = np.hypot(xs[None, :] - center[0], ys[:, None] - center[1]) <= radius + cell_size / 2
    matrix[near] = UNSEEN
    return matrix


def image_file_reader(path: str) -> Callable[[], np.ndarray | None]:
    """Return a reader that gives the image file's current content, or None if unreadable."""

    def read() -> np.ndarray | None:
        return cv2.imread(str(path))

    return read


class LiveCamera:
    """A camera stream that always gives its newest picture. Call it to get a BGR image, or
    None when no picture has arrived for `max_age` seconds.

    `source` is anything `cv2.VideoCapture` opens: the URL of a phone camera app
    (`http://<phone>:8080/video`, `rtsp://...`), a video file, or a device number. A thread
    keeps reading, since OpenCV queues pictures and a slow reader would get old ones. It
    reopens the stream when it breaks.

    A stream has its own scale: it is cropped and resized differently from a photo of the same
    camera, so set `OverheadSettings.metres_per_pixel` from a picture of this stream.
    """

    def __init__(
        self,
        source: str | int,
        max_age: float = 2.0,
        retry_delay: float = 1.0,
        open_capture: Callable = cv2.VideoCapture,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._max_age = max_age
        self._retry_delay = retry_delay
        self._open = open_capture
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._time: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        capture = None
        while not self._stop.is_set():
            if capture is None:
                capture = self._open(self._source)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            if ok and frame is not None:
                with self._lock:
                    self._frame, self._time = frame, self._clock()
            else:  # broken or not open yet: start again after a pause
                capture.release()
                capture = None
                self._stop.wait(self._retry_delay)
        if capture is not None:
            capture.release()

    def __call__(self) -> np.ndarray | None:
        with self._lock:
            if self._time is None or self._clock() - self._time > self._max_age:
                return None
            return self._frame.copy()

    def age(self) -> float | None:
        """Seconds since the newest picture arrived; None before the first one."""
        with self._lock:
            return None if self._time is None else self._clock() - self._time

    def wait_for_image(self, timeout: float = 10.0) -> bool:
        """Wait until a picture has arrived. False if none came within `timeout` seconds."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self() is not None:
                return True
            time.sleep(0.05)
        return self() is not None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class OverheadDrone:
    """The overhead camera, seen by the map fusion as its drone.

    `read_image()` returns the current BGR image, or None when the camera is not available.
    The image is processed again at most every `period` seconds, since the fusion asks for the
    rover position many times a second and one image takes a noticeable time. When the reader
    has `read_new(timeout)` (`camera_reader`), `refresh(force=True)` waits for a frame taken
    after the call, so "the picture after the rover stopped" really is one.

    A bad frame (black, white, uniform, scene lost: `pipeline.frame_problem`) is not mapped:
    the previous result is kept for `bad_frame_hold` seconds, without the rover pose, which
    may be out of date; after that the camera counts as unavailable.

    When the rover is not found but was seen before, the tiles leave the cells within
    `lost_rover_radius` of its last position unseen (-1): the unrecognised rover (touching a
    box, in its shadow) must not become an obstacle under itself.
    """

    def __init__(
        self,
        read_image: Callable[[], np.ndarray | None],
        settings: OverheadSettings | None = None,
        period: float = 0.5,
        level: int = 4,
        tile_cells: int = 32,
        clock: Callable[[], float] = time.monotonic,
        fresh_timeout: float = 2.0,
        bad_frame_hold: float = 2.0,
        lost_rover_radius: float = 0.30,
    ) -> None:
        self.settings = settings or OverheadSettings()
        self._read_image = read_image
        self._period = period
        self._level = level
        self._tile_cells = tile_cells
        self._clock = clock
        self._fresh_timeout = fresh_timeout
        self._bad_frame_hold = bad_frame_hold
        self._lost_rover_radius = lost_rover_radius
        self._result: OverheadResult | None = None
        self._updated: float | None = None
        self._good: OverheadResult | None = None  # last result of a good frame
        self._good_time: float | None = None
        self._reference_std: float | None = None
        self._held = False  # the result is an older frame's, kept over a bad one
        self._last_rover: tuple[float, float] | None = None
        self._bad_since: float | None = None
        self._image_size: tuple[int, int] | None = None  # (width, height), last good frame
        self.problem: str | None = None  # why the last frame was rejected, for logs

    def refresh(self, force: bool = False) -> OverheadResult | None:
        """Process a new image if the last one is older than `period`. None: no image."""
        now = self._clock()
        if force or self._updated is None or now - self._updated >= self._period:
            fresh = getattr(self._read_image, 'read_new', None)
            image = fresh(self._fresh_timeout) if force and fresh else self._read_image()
            self._updated = now
            self._held = False
            self._result = self._process(image, now)
        return self._result

    def _process(self, image: np.ndarray | None, now: float) -> OverheadResult | None:
        problem = 'no image' if image is None else frame_problem(image, self._reference_std)
        if problem is None or self._bad_since is None:
            self._bad_since = None if problem is None else now
        elif now - self._bad_since > RELEARN_AFTER and frame_problem(image) is None:
            # Only the comparison with the old scene fails, and for long: the scene itself
            # changed (lights, obstacles moved away). Take it as the new reference.
            log.warning('overhead camera: scene changed for good, new reference frame')
            problem, self._bad_since = None, None
        if problem is None:
            self.problem = None
            self._image_size = (image.shape[1], image.shape[0])
            self._good = process_image(image, self.settings)
            self._good_time = now
            self._reference_std = brightness_spread(image)
            if self._good.rover_position is not None:
                self._last_rover = self._good.rover_position
            return self._good
        if problem != self.problem:
            log.warning('overhead camera: frame rejected: %s', problem)
        self.problem = problem
        if image is None or self._good is None or now - self._good_time > self._bad_frame_hold:
            return None
        self._held = True
        return self._good

    # ---- what merging_map asks of the drone ------------------------------------------------

    def disponible(self) -> bool:
        """True while the camera gives images."""
        return self.refresh() is not None

    def zoomer(self, tuile: tuple[int, int, int]) -> list[dict]:
        """Look again and return the tiles that overlap the requested tile (z, x, y).

        The camera always sends its finest cells, so a zoom adds no detail: it only brings a
        fresh look at that area.
        """
        z, tx, ty = tuile
        side = self._tile_side(z)
        x0, y0 = tx * side, ty * side
        fine = self._tile_side(self._level)
        return [
            message
            for message in self.tiles(force=True)
            if message['x'] * fine < x0 + side
            and (message['x'] + 1) * fine > x0
            and message['y'] * fine < y0 + side
            and (message['y'] + 1) * fine > y0
        ]

    def position_drone(self) -> tuple[float, float, float] | None:
        """Rover (x, y, heading) for the fusion's `position_drone` slot, or None.

        None when the rover is not seen, and also when its heading is not known (rover found
        by size, no marker): the fusion assigns the heading without checking it.
        """
        pose = self.rover_pose()
        if pose is None or pose[2] is None:
            return None
        return pose

    # ---- the rest ---------------------------------------------------------------------------

    def rover_pose(self) -> tuple[float, float, float | None] | None:
        """Rover (x, y, heading); heading is None when unknown. None: rover not seen."""
        result = self.refresh()
        if result is None or result.rover_position is None or self._held:
            return None
        return (*result.rover_position, result.rover_heading)

    def tiles(self, force: bool = False) -> list[dict]:
        """The whole view as tile messages; empty when there is no image."""
        result = self.refresh(force)
        if result is None:
            return []
        matrix = grid_to_fusion(result.grid)
        if result.rover_position is None and self._last_rover is not None:
            matrix = unsee_disc(
                matrix, self.settings.cell_size, self._last_rover, self._lost_rover_radius
            )
        return cut_tiles(
            matrix,
            self.settings.cell_size,
            self._level,
            tile_cells=self._tile_cells,
            t=self._good_time if self._held else self._updated,
        )

    # ---- checking the scale against a move of the rover (scale_check.py) --------------------

    def current_scale(self) -> float | None:
        """Metres per pixel in use for the camera's images; None before the first image."""
        if self.refresh() is None or self._image_size is None:
            return None
        return self.settings.scale(self._image_size)

    def objects(self, force: bool = False) -> list[Seen] | None:
        """Everything that stands out from the floor, the rover included: centres and sizes in
        the camera's metres. None when there is no usable image."""
        result = self.refresh(force)
        if result is None or self._held:
            return None
        resolution = self.settings.resolution
        rover = []
        if self.settings.rover_marker_id is not None and result.rover_position is not None:
            # found by its marker: it is not among the blobs, which leave the rover out
            short, long = sorted(self.settings.rover_size)
            rover = [Seen(*result.rover_position, short, long)]
        return rover + [
            Seen(
                blob.center[0] * resolution,
                result.bounds.y_max - blob.center[1] * resolution,
                blob.size[0],
                blob.size[1],
            )
            for blob in result.blobs
        ]

    def rescale(self, correction: float) -> float:
        """Multiply the scale by `correction` (see scale_check) and look again. Everything
        learned at the old scale is dropped. Returns the new metres per pixel."""
        scale = self.current_scale()
        if scale is None:
            raise RuntimeError('no image from the camera: the scale cannot be changed')
        self.settings = dataclasses.replace(self.settings, metres_per_pixel=scale * correction)
        self._last_rover = None
        self._good = None
        self.refresh(force=True)
        return self.settings.metres_per_pixel

    def zone(self) -> tuple[float, float, float, float] | None:
        """Area (x0, y0, x1, y1) the camera sees, to pass as `zone` to the fusion's planner."""
        result = self.refresh()
        if result is None:
            return None
        bounds = result.bounds
        return (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)

    def _tile_side(self, z: int) -> float:
        return self._tile_cells * self.settings.cell_size * 2 ** (self._level - z)
