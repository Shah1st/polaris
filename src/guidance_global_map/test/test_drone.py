import dataclasses
import math
import time

import cv2
import numpy as np
import pytest

from guidance_calib.synthetic import Box, Scene, render_top_down
from guidance_global_map import pipeline
from guidance_global_map.drone import LiveCamera, OverheadDrone, cut_tiles
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


# ---- scale from the camera height follows the actual image size --------------------------


def test_scale_from_height_for_resized_and_cropped_images(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr(pipeline, '_warned', set())
    monkeypatch.setattr(pipeline.log, 'warning', lambda *args: warnings.append(args))
    photo = OverheadSettings(camera_height=1.9)
    full = photo.scale()

    assert photo.scale((4032, 3024)) == pytest.approx(full)
    assert photo.scale((1280, 960)) == pytest.approx(full * 4032 / 1280)  # resized whole view
    assert not warnings
    # another shape is a crop: the sensor's scale, with one warning per image size
    assert photo.scale((2595, 2829)) == pytest.approx(full)
    assert photo.scale((2595, 2829)) == pytest.approx(full)
    assert photo.scale((1920, 1080)) == pytest.approx(full)
    assert len(warnings) == 2 and all('taken as a crop' in w[0] for w in warnings)
    assert OverheadSettings(metres_per_pixel=0.004).scale((640, 480)) == 0.004


def test_a_cropped_photo_keeps_its_scale_and_its_rover() -> None:
    """A photo cut down to another shape, with the camera-height scale: the area and the
    rover's place come out as in the whole photo (the bug: 1.55 times too large, and another
    object taken for the rover)."""
    image = render_top_down(scene_with_marked_rover())  # 780 x 630 at 300 px per metre
    sensor = (image.shape[1], image.shape[0])
    height = 1.9
    focal = 43.27 * height / (math.hypot(*sensor) / 300.0)  # the lens that gives 300 px/m
    whole = OverheadSettings(
        camera_height=height,
        focal_length_35mm=focal,
        sensor_size=sensor,
        floor_window=0.6,
        rover_marker_id=10,
    )
    crop = image[90:, :600]  # cut 0.30 m off the top and 0.60 m off the right

    full, cut = process_image(image, whole), process_image(crop, whole)

    assert cut.bounds.x_max == pytest.approx(2.0, abs=0.01)
    assert cut.bounds.y_max == pytest.approx(1.8, abs=0.01)
    assert cut.rover_position == pytest.approx(full.rover_position, abs=0.01)


def test_objects_and_rescale() -> None:
    """The camera adapter lists what it sees and takes a scale correction."""
    image = render_top_down(scene_with_marked_rover())
    camera = FakeCamera(image)
    drone = OverheadDrone(camera, settings(), clock=lambda: 0.0)  # 1 / 300 m per pixel

    seen = drone.objects()
    box = min(seen, key=lambda o: math.dist((o.x, o.y), (1.625, 1.375)))
    assert (box.x, box.y) == pytest.approx((1.625, 1.375), abs=0.02)
    assert (box.short, box.long) == pytest.approx((0.25, 0.35), abs=0.02)
    assert drone.current_scale() == pytest.approx(1 / 300)

    new = drone.rescale(2.0)

    assert new == pytest.approx(2 / 300) == drone.current_scale()
    assert drone.zone() == pytest.approx((0.0, 0.0, 5.2, 4.2), abs=0.02)
    box = min(drone.objects(), key=lambda o: math.dist((o.x, o.y), (3.25, 2.75)))
    assert (box.short, box.long) == pytest.approx((0.50, 0.70), abs=0.04)


def test_a_resized_frame_covers_the_same_area() -> None:
    image = render_top_down(scene_with_marked_rover())  # 780 x 630, as if the full sensor
    small = image[::2, ::2]
    photo = OverheadSettings(
        camera_height=1.9, sensor_size=(image.shape[1], image.shape[0]), floor_window=0.6
    )

    full, half = process_image(image, photo), process_image(small, photo)

    assert (half.bounds.x_max, half.bounds.y_max) == pytest.approx(
        (full.bounds.x_max, full.bounds.y_max), rel=0.01
    )


# ---- marker parallax ----------------------------------------------------------------------


def test_marker_position_is_moved_towards_the_point_under_the_camera() -> None:
    image = render_top_down(scene_with_marked_rover())  # marker drawn at (0.80, 0.90)
    nadir = (image.shape[1] * SCALE / 2.0, image.shape[0] * SCALE / 2.0)  # (1.30, 1.05)

    flat = process_image(image, settings(rover_marker_id=10))
    raised = process_image(
        image, settings(rover_marker_id=10, marker_height=0.19, camera_height=1.9)
    )

    factor = (1.9 - 0.19) / 1.9
    expected = [n + (p - n) * factor for p, n in zip(flat.rover_position, nadir, strict=True)]
    assert raised.rover_position == pytest.approx(expected, abs=0.002)
    assert raised.rover_heading == pytest.approx(flat.rover_heading, abs=1e-9)
    # The rover's cells cover both where the marker appears and where the rover stands.
    assert cell(raised.grid, 0.50, 0.60) == ROVER
    x, y = raised.rover_position
    assert cell(raised.grid, x - SHIFT, y - SHIFT) == ROVER


# ---- bad frames ---------------------------------------------------------------------------


def obstacle_count(tiles: list[dict]) -> int:
    return int(sum((m['data'] == 1).sum() for m in tiles))


@pytest.mark.parametrize('kind', ['black', 'white', 'grey', 'dark noise', 'covered'])
def test_a_bad_frame_never_reads_as_free_and_the_camera_goes_unavailable(kind) -> None:
    good = render_top_down(scene_with_marked_rover())
    bad = {
        'black': np.zeros_like(good),
        'white': np.full_like(good, 255),
        'grey': np.full_like(good, 128),
        'dark noise': np.random.default_rng(0).integers(0, 25, good.shape, dtype=np.uint8),
        # A hand in front of the lens: dim, blurred, little contrast left.
        'covered': np.clip(np.random.default_rng(1).normal(90.0, 2.0, good.shape), 0, 255).astype(
            np.uint8
        ),
    }[kind]
    now = [0.0]
    camera = FakeCamera(good)
    drone = OverheadDrone(camera, settings(rover_marker_id=10), clock=lambda: now[0])
    before = drone.tiles()
    assert obstacle_count(before) > 0 and drone.rover_pose() is not None

    camera.image = bad
    now[0] = 1.0
    held = drone.tiles(force=True)
    assert obstacle_count(held) == obstacle_count(before)  # the last good map is kept
    assert all(m['t'] == 0.0 for m in held)  # with the time it was seen
    assert drone.disponible()
    assert drone.rover_pose() is None  # the pose may be out of date: withheld
    assert drone.problem

    now[0] = 2.5  # more than bad_frame_hold after the last good frame
    assert drone.refresh(force=True) is None
    assert not drone.disponible() and drone.tiles() == []

    camera.image = good
    now[0] = 3.0
    assert drone.rover_pose() is not None and drone.problem is None


def test_a_lasting_change_of_scene_becomes_the_new_reference() -> None:
    busy = render_top_down(scene_with_marked_rover())
    plain = render_top_down(Scene(marker_positions={}, boxes=[Box(1.20, 0.90, 0.25, 0.35)]))
    plain = cv2.addWeighted(plain, 0.25, np.full_like(plain, 180), 0.75, 0.0)  # low contrast
    now = [0.0]
    camera = FakeCamera(busy)
    drone = OverheadDrone(camera, settings(), clock=lambda: now[0])
    drone.refresh()

    camera.image = plain
    for t in (1.0, 5.0, 10.5):
        now[0] = t
        drone.refresh(force=True)
    assert not drone.disponible()  # rejected while it might be a transient
    now[0] = 12.0
    assert drone.refresh(force=True) is not None and drone.problem is None


# ---- rover not recognised ----------------------------------------------------------------


def test_an_unrecognised_rover_does_not_become_an_obstacle_where_it_was_last_seen() -> None:
    rover = Box(0.40, 0.50, 0.20, 0.23, color=(30, 30, 30))
    alone = render_top_down(Scene(marker_positions={}, boxes=[rover]))
    # The rover now touches a box: one blob, too large to be the rover.
    box = Box(0.60, 0.45, 0.40, 0.35, color=(40, 60, 200))
    stuck = render_top_down(Scene(marker_positions={}, boxes=[rover, box]))
    now = [0.0]
    camera = FakeCamera(alone)
    drone = OverheadDrone(camera, settings(), clock=lambda: now[0])
    assert drone.rover_pose() == pytest.approx((0.80, 0.915, None), abs=0.015)

    camera.image = stuck
    now[0] = 1.0
    tiles = drone.tiles(force=True)

    assert drone.rover_pose() is None
    tile = {(m['x'], m['y']): m['data'] for m in tiles}[(0, 0)]
    assert tile[int(0.915 / 0.05), int(0.80 / 0.05)] == -1  # unseen, not an obstacle
    assert tile[int(0.95 / 0.05), int(1.25 / 0.05)] == 1  # the box's far side stays
    assert tile[int(0.30 / 0.05), int(0.30 / 0.05)] == 0  # floor away from the rover


class FakeCapture:
    """Stands in for cv2.VideoCapture: gives numbered pictures, then breaks if told to."""

    opened = 0

    def __init__(self, source, frames=1000) -> None:
        type(self).opened += 1
        self.left, self.count, self.released = frames, 0, False

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV's name)
        return True

    def read(self):
        time.sleep(0.002)
        if self.left <= 0:
            return False, None
        self.left -= 1
        self.count += 1
        return True, np.full((2, 2, 3), self.count % 250, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_live_camera_gives_the_newest_picture() -> None:
    camera = LiveCamera('stream', open_capture=FakeCapture)
    try:
        assert camera.wait_for_image(2.0)
        first = int(camera()[0, 0, 0])
        time.sleep(0.1)
        later = int(camera()[0, 0, 0])
        assert later != first  # it kept reading in the background
        assert camera.age() < 0.5
    finally:
        camera.close()


def test_live_camera_gives_nothing_when_the_stream_stops_and_reopens_it() -> None:
    FakeCapture.opened = 0
    camera = LiveCamera(
        'stream',
        max_age=0.05,
        retry_delay=0.2,
        open_capture=lambda source: FakeCapture(source, frames=3),
    )
    try:
        assert camera.wait_for_image(2.0)
        time.sleep(0.12)  # the 3 pictures are used up and the last one is too old
        assert camera() is None
        time.sleep(0.3)  # reopened: pictures again
        assert camera.wait_for_image(2.0)
        assert FakeCapture.opened >= 2
    finally:
        camera.close()


def test_live_camera_before_the_first_picture() -> None:
    camera = LiveCamera('stream', open_capture=lambda source: FakeCapture(source, frames=0))
    try:
        assert camera() is None
        assert camera.age() is None
        assert not camera.wait_for_image(0.1)
    finally:
        camera.close()


def test_drone_follows_a_live_camera() -> None:
    image = render_top_down(scene_with_marked_rover())

    class Stream(FakeCapture):
        def read(self):
            time.sleep(0.002)
            return True, image

    camera = LiveCamera('stream', open_capture=Stream)
    try:
        assert camera.wait_for_image(2.0)
        drone = OverheadDrone(camera, settings(rover_marker_id=10))
        assert drone.disponible()
        assert drone.position_drone()[:2] == pytest.approx((0.80, 0.90), abs=0.01)
    finally:
        camera.close()


def test_a_rover_found_by_its_marker_is_among_the_objects() -> None:
    """With a marker the rover is left out of the obstacle blobs; the scale check still needs
    to see it move, so it is listed at its marker's position."""
    camera = FakeCamera(render_top_down(scene_with_marked_rover()))
    drone = OverheadDrone(camera, settings(rover_marker_id=10), clock=lambda: 0.0)

    seen = drone.objects()

    rover = min(seen, key=lambda o: math.dist((o.x, o.y), (0.80, 0.90)))
    assert (rover.x, rover.y) == pytest.approx((0.80, 0.90), abs=0.01)
    assert (rover.short, rover.long) == (0.20, 0.23)
    assert len(seen) == 2  # the rover and the box


# ---- scale read off the rover's marker ---------------------------------------------------------


def test_scale_is_read_off_the_marker_whatever_the_height_says() -> None:
    """The scene is at 300 px per metre and its marker is 10 cm wide. With a camera height
    that is badly wrong, the marker still gives the right scale, area and rover position."""
    image = render_top_down(scene_with_marked_rover())
    wrong = OverheadSettings(camera_height=3.0, floor_window=0.6, rover_marker_id=10)

    assert process_image(image, wrong).bounds.x_max != pytest.approx(2.6, abs=0.2)
    with_marker = dataclasses.replace(wrong, marker_size=0.10)
    result = process_image(image, with_marker)

    # a 30-pixel marker: its side is read to about a pixel, 3 %
    assert pipeline.marker_scale(image, with_marker) == pytest.approx(1 / 300, rel=0.04)
    assert result.bounds.x_max == pytest.approx(2.6, rel=0.04)
    assert result.rover_position == pytest.approx((0.80, 0.90), abs=0.03)


def test_marker_above_the_floor_is_brought_down_to_the_floor() -> None:
    image = render_top_down(scene_with_marked_rover())
    flat = OverheadSettings(camera_height=2.0, rover_marker_id=10, marker_size=0.10)
    raised = dataclasses.replace(flat, marker_height=0.20)

    assert pipeline.marker_scale(image, raised) == pytest.approx(
        pipeline.marker_scale(image, flat) * 2.0 / 1.8
    )


def test_no_marker_scale_without_a_size_an_id_or_a_marker_in_view() -> None:
    image = render_top_down(scene_with_marked_rover())
    empty = render_top_down(Scene(marker_positions={}))

    assert pipeline.marker_scale(image, OverheadSettings(rover_marker_id=10)) is None
    assert pipeline.marker_scale(image, OverheadSettings(marker_size=0.10)) is None
    assert (
        pipeline.marker_scale(empty, OverheadSettings(rover_marker_id=10, marker_size=0.1)) is None
    )
    # a measured scale wins over the marker
    measured = OverheadSettings(metres_per_pixel=0.01, rover_marker_id=10, marker_size=0.10)
    assert process_image(image, measured).bounds.x_max == pytest.approx(7.8)


def test_the_marker_scale_is_kept_while_the_marker_is_hidden() -> None:
    seen = render_top_down(scene_with_marked_rover())
    hidden = render_top_down(Scene(marker_positions={}, boxes=scene_with_marked_rover().boxes))
    camera = FakeCamera(seen)
    now = [0.0]
    drone = OverheadDrone(
        camera,
        OverheadSettings(camera_height=3.0, floor_window=0.6, rover_marker_id=10, marker_size=0.1),
        clock=lambda: now[0],
    )

    assert drone.zone()[2] == pytest.approx(2.6, rel=0.04)
    assert drone.current_scale() == pytest.approx(1 / 300, rel=0.04)
    camera.image, now[0] = hidden, 10.0

    assert drone.zone()[2] == pytest.approx(2.6, rel=0.04)  # not back to the height's scale
    assert drone.rover_pose() is None
