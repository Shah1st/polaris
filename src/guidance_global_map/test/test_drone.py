import math

import numpy as np
import pytest

from guidance_calib.synthetic import Box, Scene, render_top_down
from guidance_global_map.drone import OverheadDrone, cut_tiles
from guidance_global_map.pipeline import OverheadSettings, process_image
from guidance_global_map.segmentation import OCCUPIED, ROVER

# The synthetic top-down view is 300 px per metre and has a 0.30 m margin around the scene, so
# a scene point (x, y) is at (x + 0.30, y + 0.30) in the map of the image.
SCALE = 1.0 / 300.0
SHIFT = 0.30


def settings(**overrides) -> OverheadSettings:
    return OverheadSettings(metres_per_pixel=SCALE, floor_window=0.6, **overrides)


def scene_with_marked_rover() -> Scene:
    return Scene(
        marker_positions={},
        boxes=[Box(1.20, 0.90, 0.25, 0.35, color=(60, 140, 40))],
        rover_pose=(0.50, 0.60, math.radians(30.0)),
    )


def cell(grid: np.ndarray, x: float, y: float) -> int:
    return int(grid[int((y + SHIFT) / 0.05), int((x + SHIFT) / 0.05)])


def test_pipeline_with_a_marker_gives_position_and_heading() -> None:
    image = render_top_down(scene_with_marked_rover())

    result = process_image(image, settings(rover_marker_id=10))

    x, y = result.rover_position
    assert x == pytest.approx(0.50 + SHIFT, abs=0.01)
    assert y == pytest.approx(0.60 + SHIFT, abs=0.01)
    assert result.rover_heading == pytest.approx(math.radians(30.0), abs=math.radians(2.0))
    assert cell(result.grid, 1.32, 1.07) == OCCUPIED
    assert cell(result.grid, 0.50, 0.60) == ROVER
    assert not (result.grid[:10] == OCCUPIED).any()  # nothing but floor along the south edge


def test_pipeline_by_size_gives_position_without_heading() -> None:
    scene = Scene(
        marker_positions={},
        boxes=[
            Box(1.20, 0.90, 0.25, 0.35, color=(60, 140, 40)),
            Box(0.40, 0.50, 0.20, 0.23, color=(30, 30, 30)),  # the rover: a dark 20 x 23 cm box
        ],
    )

    result = process_image(render_top_down(scene), settings())

    x, y = result.rover_position
    assert x == pytest.approx(0.50 + SHIFT, abs=0.015)
    assert y == pytest.approx(0.615 + SHIFT, abs=0.015)
    assert result.rover_heading is None
    assert cell(result.grid, 0.50, 0.60) == ROVER
    assert cell(result.grid, 1.32, 1.07) == OCCUPIED


def test_no_rover_in_view() -> None:
    scene = Scene(marker_positions={}, boxes=[Box(1.20, 0.90, 0.25, 0.35)])

    result = process_image(render_top_down(scene), settings(rover_marker_id=10))

    assert result.rover_position is None
    assert result.rover_mask is None
    assert not (result.grid == ROVER).any()


def test_tiles_are_aligned_to_the_global_grid() -> None:
    matrix = np.zeros((40, 50), dtype=np.int8)
    matrix[35, 45] = 1  # x = 2.25, y = 1.75: tile (1, 1), cell [3][13]

    messages = cut_tiles(matrix, cell_size=0.05, level=4, t=7.0)

    assert [(m['z'], m['x'], m['y']) for m in messages] == [
        (4, 0, 0),
        (4, 1, 0),
        (4, 0, 1),
        (4, 1, 1),
    ]
    by_key = {(m['x'], m['y']): m for m in messages}
    assert all(m['data'].shape == (32, 32) and m['data'].dtype == np.int8 for m in messages)
    assert all(m['source'] == 'drone' and m['t'] == 7.0 for m in messages)
    assert by_key[(1, 1)]['data'][3, 13] == 1
    assert (by_key[(0, 0)]['data'] == 0).all()
    assert (by_key[(1, 1)]['data'][8:, :] == -1).all()  # beyond row 40 of the matrix
    assert (by_key[(1, 1)]['data'][:, 18:] == -1).all()  # beyond column 50


def test_tiles_handle_an_offset_origin_and_drop_unseen_tiles() -> None:
    matrix = np.full((32, 64), -1, dtype=np.int8)
    matrix[0, 0] = 0

    messages = cut_tiles(matrix, cell_size=0.05, level=4, x0=-1.6, y0=1.6)

    assert [(m['x'], m['y']) for m in messages] == [(-1, 1)]
    assert messages[0]['data'][0, 0] == 0


class FakeCamera:
    """Counts reads; gives an image, or None when switched off."""

    def __init__(self, image: np.ndarray) -> None:
        self.image, self.on, self.reads = image, True, 0

    def __call__(self) -> np.ndarray | None:
        self.reads += 1
        return self.image if self.on else None


def test_drone_reports_the_rover_and_the_tiles() -> None:
    camera = FakeCamera(render_top_down(scene_with_marked_rover()))
    drone = OverheadDrone(camera, settings(rover_marker_id=10), clock=lambda: 0.0)

    assert drone.disponible()
    x, y, heading = drone.position_drone()
    assert (x, y) == pytest.approx((0.80, 0.90), abs=0.01)
    assert heading == pytest.approx(math.radians(30.0), abs=math.radians(2.0))
    zone = drone.zone()
    assert zone == pytest.approx((0.0, 0.0, 2.6, 2.1), abs=0.01)
    tiles = drone.tiles()
    assert {(m['x'], m['y']) for m in tiles} == {(0, 0), (1, 0), (0, 1), (1, 1)}
    values = np.concatenate([m['data'].ravel() for m in tiles])
    assert set(np.unique(values)) == {-1, 0, 1}  # the rover's own cells are sent as free
    assert camera.reads == 1  # one image served all of these calls


def test_position_is_withheld_when_the_heading_is_unknown() -> None:
    scene = Scene(marker_positions={}, boxes=[Box(0.40, 0.50, 0.20, 0.23, color=(30, 30, 30))])
    drone = OverheadDrone(FakeCamera(render_top_down(scene)), settings(), clock=lambda: 0.0)

    assert drone.rover_pose() == pytest.approx((0.80, 0.915, None), abs=0.015)
    assert drone.position_drone() is None


def test_drone_is_unavailable_without_an_image() -> None:
    camera = FakeCamera(render_top_down(scene_with_marked_rover()))
    camera.on = False
    drone = OverheadDrone(camera, settings(rover_marker_id=10), clock=lambda: 0.0)

    assert not drone.disponible()
    assert drone.position_drone() is None
    assert drone.tiles() == []
    assert drone.zoomer((4, 0, 0)) == []
    assert drone.zone() is None


def test_image_is_processed_again_only_after_the_period() -> None:
    now = [0.0]
    camera = FakeCamera(render_top_down(scene_with_marked_rover()))
    drone = OverheadDrone(camera, settings(rover_marker_id=10), period=0.5, clock=lambda: now[0])

    drone.disponible()
    drone.position_drone()
    now[0] = 0.4
    drone.position_drone()
    assert camera.reads == 1
    now[0] = 0.6
    drone.position_drone()
    assert camera.reads == 2


def test_zoom_looks_again_and_returns_the_overlapping_tiles() -> None:
    now = [0.0]
    camera = FakeCamera(render_top_down(scene_with_marked_rover()))
    drone = OverheadDrone(camera, settings(rover_marker_id=10), clock=lambda: now[0])
    drone.tiles()

    fine = drone.zoomer((4, 1, 0))  # one finest tile: x 1.6..3.2, y 0..1.6
    coarse = drone.zoomer((2, 0, 0))  # a 20 cm-cell tile: 6.4 m wide, covers the whole view

    assert [(m['x'], m['y']) for m in fine] == [(1, 0)]
    assert {(m['x'], m['y']) for m in coarse} == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert drone.zoomer((4, 5, 5)) == []
    assert camera.reads == 4  # every zoom takes a new image, even within the period
