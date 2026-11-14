"""The overhead camera as the "drone" of the map fusion (`merging_map`). No ROS.

`merging_map` takes the drone's map as tiles and asks the drone side for three things. This
module provides them from the overhead pipeline:

    camera = OverheadDrone(image_file_reader('photo.jpg'))
    for message in camera.tiles():          # first survey
        carte.recevoir(message)
    rover = RaspRoverReel(carte, ..., position_drone=camera.position_drone, drone=camera)

- `drone.disponible()` and `drone.zoomer(tile)`: methods of `OverheadDrone`;
- `position_drone()`: `OverheadDrone.position_drone`.

Tile format, from `merging_map/README.md`: N x N cells (N = 32), values 1 obstacle, 0 free and
seen, -1 not seen, row 0 = south, tile (z, x, y) covering x*T <= X < (x+1)*T with T = N * cell
size. The grid here has 5 cm cells, which is level z = 4 of the fusion's defaults.
"""

import math
import time
from collections.abc import Callable

import cv2
import numpy as np

from guidance_global_map.pipeline import OverheadResult, OverheadSettings, process_image
from guidance_global_map.segmentation import grid_to_fusion

UNSEEN = -1


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


def image_file_reader(path: str) -> Callable[[], np.ndarray | None]:
    """Return a reader that gives the image file's current content, or None if unreadable."""

    def read() -> np.ndarray | None:
        return cv2.imread(str(path))

    return read


class OverheadDrone:
    """The overhead camera, seen by the map fusion as its drone.

    `read_image()` returns the current BGR image, or None when the camera is not available.
    The image is processed again at most every `period` seconds, since the fusion asks for the
    rover position many times a second and one image takes a noticeable time.
    """

    def __init__(
        self,
        read_image: Callable[[], np.ndarray | None],
        settings: OverheadSettings | None = None,
        period: float = 0.5,
        level: int = 4,
        tile_cells: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or OverheadSettings()
        self._read_image = read_image
        self._period = period
        self._level = level
        self._tile_cells = tile_cells
        self._clock = clock
        self._result: OverheadResult | None = None
        self._updated: float | None = None

    def refresh(self, force: bool = False) -> OverheadResult | None:
        """Process a new image if the last one is older than `period`. None: no image."""
        now = self._clock()
        if force or self._updated is None or now - self._updated >= self._period:
            image = self._read_image()
            self._result = None if image is None else process_image(image, self.settings)
            self._updated = now
        return self._result

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
        if result is None or result.rover_position is None:
            return None
        return (*result.rover_position, result.rover_heading)

    def tiles(self, force: bool = False) -> list[dict]:
        """The whole view as tile messages; empty when there is no image."""
        result = self.refresh(force)
        if result is None:
            return []
        return cut_tiles(
            grid_to_fusion(result.grid),
            self.settings.cell_size,
            self._level,
            tile_cells=self._tile_cells,
            t=self._updated,
        )

    def zone(self) -> tuple[float, float, float, float] | None:
        """Area (x0, y0, x1, y1) the camera sees, to pass as `zone` to the fusion's planner."""
        result = self.refresh()
        if result is None:
            return None
        bounds = result.bounds
        return (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)

    def _tile_side(self, z: int) -> float:
        return self._tile_cells * self.settings.cell_size * 2 ** (self._level - z)
