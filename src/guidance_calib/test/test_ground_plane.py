import numpy as np
import pytest

from guidance_calib.aruco_compat import detect_markers, generate_marker
from guidance_calib.generate_markers import marker_with_border
from guidance_calib.ground_plane import (
    MapBounds,
    aspect_differs,
    estimate_homography,
    map_to_top_down,
    metres_per_pixel_for_image,
    metres_per_pixel_for_view,
    metres_per_pixel_from_height,
    pixels_to_map,
    straight_down_homography,
    warp_to_top_down,
)
from guidance_calib.synthetic import Scene, render_camera_view


def test_generated_marker_is_detected() -> None:
    image = marker_with_border(7, 200, 'DICT_4X4_50')

    detections = detect_markers(image)

    assert list(detections) == [7]
    assert detections[7].shape == (4, 2)
    assert generate_marker(7, 200).shape == (200, 200)


def test_homography_recovers_map_coordinates_within_a_centimetre() -> None:
    scene = Scene()
    image, true_homography = render_camera_view(scene)

    homography = estimate_homography(detect_markers(image), scene.marker_positions)

    assert homography is not None
    rows, cols = image.shape[:2]
    pixels = np.array(
        [[cols * 0.3, rows * 0.3], [cols * 0.7, rows * 0.4], [cols * 0.5, rows * 0.8]]
    )
    error = pixels_to_map(homography, pixels) - pixels_to_map(true_homography, pixels)
    assert np.abs(error).max() < 0.01


def test_homography_needs_four_known_markers() -> None:
    scene = Scene()
    image, _ = render_camera_view(scene)
    detections = detect_markers(image)

    del detections[2]

    assert estimate_homography(detections, scene.marker_positions) is None
    assert estimate_homography({}, scene.marker_positions) is None


def test_top_down_view_has_y_up_and_metric_size() -> None:
    bounds = MapBounds(0.0, 0.0, 2.0, 1.5)

    assert bounds.grid_shape(0.01) == (150, 200)
    pixels = map_to_top_down(np.array([[0.0, 1.5], [2.0, 0.0]]), bounds, 0.01)
    np.testing.assert_allclose(pixels, [[0.0, 0.0], [200.0, 150.0]], atol=1e-9)


def test_warp_marks_area_outside_the_image_as_not_valid() -> None:
    scene = Scene()
    image, homography = render_camera_view(scene)

    inside = MapBounds(0.0, 0.0, 2.0, 1.5)
    _, valid_inside = warp_to_top_down(image, homography, inside, 0.01)
    beyond = MapBounds(0.0, 0.0, 6.0, 1.5)
    _, valid_beyond = warp_to_top_down(image, homography, beyond, 0.01)

    assert valid_inside.all()
    assert not valid_beyond[:, -1].any()


def test_scale_from_camera_height() -> None:
    # iPhone main camera (26 mm equivalent, 4032 x 3024) at 1.9 m sees about 2.53 x 1.90 m.
    metres_per_pixel = metres_per_pixel_from_height(1.9, 26.0, 4032, 3024)

    assert metres_per_pixel * 4032 == pytest.approx(2.53, abs=0.01)
    assert metres_per_pixel_from_height(3.8, 26.0, 4032, 3024) == pytest.approx(
        2 * metres_per_pixel
    )


def test_scale_follows_the_image_width() -> None:
    full = metres_per_pixel_from_height(1.9, 26.0, 4032, 3024)

    # The same field of view, in fewer pixels: each pixel covers more ground.
    assert metres_per_pixel_for_image(1.9, 26.0, 4032, 3024, 1280) * 1280 == pytest.approx(
        full * 4032
    )
    assert metres_per_pixel_for_image(1.9, 26.0, 4032, 3024, 4032) == pytest.approx(full)
    assert not aspect_differs((1280, 960), (4032, 3024))
    assert aspect_differs((1920, 1080), (4032, 3024))


def test_straight_down_homography_puts_the_origin_at_the_bottom_left() -> None:
    homography = straight_down_homography(0.001, image_rows=3000)

    corners = pixels_to_map(homography, np.array([[0.0, 3000.0], [4000.0, 0.0]]))

    np.testing.assert_allclose(corners, [[0.0, 0.0], [4.0, 3.0]], atol=1e-9)


def test_warp_of_a_large_image_averages_fine_texture() -> None:
    # One-pixel checkerboard at 1 mm per pixel, warped to 1 cm cells: it must come out grey.
    checker = (np.indices((1000, 1000)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.dstack([checker] * 3)
    homography = straight_down_homography(0.001, image_rows=1000)

    top_down, valid = warp_to_top_down(image, homography, MapBounds(0.0, 0.0, 1.0, 1.0), 0.01)

    assert top_down.shape == (100, 100, 3)
    assert valid.all()
    assert top_down[10:90, 10:90].std() < 5.0


def test_scale_of_a_full_resized_or_cropped_image() -> None:
    sensor = (4032, 3024)
    full = metres_per_pixel_from_height(1.9, 26.0, *sensor)

    assert metres_per_pixel_for_view(1.9, 26.0, sensor, (4032, 3024)) == (full, 'full')
    # the sensor's shape with fewer pixels: the whole view, resized
    scale, relation = metres_per_pixel_for_view(1.9, 26.0, sensor, (1280, 960))
    assert (scale, relation) == (pytest.approx(full * 4032 / 1280), 'resized')
    # another shape: a crop, which keeps the sensor's scale (a photo cut to 2595 x 2829)
    assert metres_per_pixel_for_view(1.9, 26.0, sensor, (2595, 2829)) == (full, 'cropped')
    assert metres_per_pixel_for_view(1.9, 26.0, sensor, (1920, 1080)) == (full, 'cropped')
    # the scale is the height times a constant of the camera mode
    assert metres_per_pixel_for_view(3.8, 26.0, sensor, (2595, 2829))[0] == pytest.approx(2 * full)


def test_a_portrait_photo_is_compared_with_the_sensor_turned() -> None:
    sensor = (4032, 3024)
    full = metres_per_pixel_from_height(1.9, 26.0, *sensor)

    assert metres_per_pixel_for_view(1.9, 26.0, sensor, (3024, 4032)) == (full, 'full')
    scale, relation = metres_per_pixel_for_view(1.9, 26.0, sensor, (960, 1280))
    assert (scale, relation) == (pytest.approx(full * 3024 / 960), 'resized')
    assert metres_per_pixel_for_view(1.9, 26.0, sensor, (2062, 2905)) == (full, 'cropped')
