"""The real-rover maps (fusion_reelle.py): the overhead camera placed on the maps without a
marker, from a move of the rover. Fake rover and perception; real OverheadDrone on rendered
pictures."""

import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip('cv2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fusion_reelle as fr  # noqa: E402

drone = pytest.importorskip('guidance_global_map.drone')
import cv2  # noqa: E402

from guidance_calib.ground_plane import MapBounds  # noqa: E402
from guidance_calib.synthetic import Box, Scene, render_top_down  # noqa: E402
from guidance_global_map.pipeline import OverheadSettings  # noqa: E402

TRUE_SCALE = 1.0 / 300.0  # the rendered pictures: 300 pixels per metre
SHIFT = 0.30  # a scene point (x, y) is at (x + 0.30, y + 0.30) in the picture's frame
BOX = Box(1.20, 0.90, 0.25, 0.35, color=(60, 140, 40))


def picture(rover_x, rover_y=0.50):
    """Seen from above: one box, and the rover as a dark 20 x 23 cm block, with no marker."""
    rover = Box(rover_x - 0.10, rover_y - 0.115, 0.20, 0.23, color=(30, 30, 30))
    return render_top_down(Scene(marker_positions={}, boxes=[BOX, rover]))


class Reader:
    def __init__(self, image):
        self.image = image

    def __call__(self):
        return self.image


def fusion(scale_error=1.0, measured=False):
    """A FusionReelle with no network: its camera is a real OverheadDrone on `Reader`."""
    f = fr.FusionReelle(
        fr.options([]),
        requete=lambda *a: None,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    reader = Reader(picture(0.50))
    settings = OverheadSettings(
        metres_per_pixel=TRUE_SCALE * scale_error,
        rover_marker_id=10,  # as the server sets it; there is no marker in the pictures
        floor_window=0.6,
    )
    f.lecteur = reader
    f.camera = drone.OverheadDrone(reader, settings, period=0.0)
    f.mpp_auto, f.mpp_mesure = False, measured
    return f, reader


def stop_at(f, reader, rover_x, pose):
    """The rover stands still at `pose` (maps' frame) and the camera sees it at `rover_x`."""
    reader.image = picture(rover_x)
    with f.verrou:
        f.pose, f.immobile = pose, True
    f.tour_drone()


def test_the_drone_map_is_placed_by_a_move_of_the_rover_without_a_marker() -> None:
    """The camera's scale is 25 % too large and its frame is turned a quarter turn from the
    maps'. The rover drives 20 cm and stops: the scale is corrected, the camera's obstacles
    land at the right place on the maps, and the rover is not an obstacle under itself."""
    f, reader = fusion(scale_error=1.25)
    heading = math.pi / 2  # the rover drives along +y of the maps, +x of the picture

    stop_at(f, reader, 0.50, (20.0, 20.0, heading))
    assert not f.drone['cale'] and 'waiting for the rover to move' in f.drone['mouvement']
    assert not f.carte.tuiles['drone']

    stop_at(f, reader, 0.70, (20.0, 20.2, heading))
    assert f.calage_mvt is not None
    assert 'scale corrected by -20 %' in f.drone['mouvement']
    assert f.camera.settings.metres_per_pixel == pytest.approx(TRUE_SCALE, rel=0.03)
    assert not f.carte.tuiles['drone']  # that picture was at the old scale: not placed

    stop_at(f, reader, 0.70, (20.0, 20.2, heading))  # the next picture is placed
    assert f.carte.tuiles['drone']
    assert f.drone['cale'] and f.drone['cale_par'] == 'movement'
    rotation, cx, cy, mx, my = f.calage_mvt
    assert rotation == pytest.approx(math.pi / 2, abs=math.radians(3))
    assert (cx, cy) == pytest.approx((0.70 + SHIFT, 0.50 + SHIFT), abs=0.03)

    def on_the_maps(x, y):  # a point of the scene, in the frame of the maps
        dx, dy = x - 0.70, y - 0.50
        return 20.0 - dy, 20.2 + dx  # a quarter turn

    def value(x, y):
        px, py = on_the_maps(x, y)
        g = f.carte.grille(px - 0.05, py - 0.05, px + 0.05, py + 0.05)
        return int(g.val[g.val.shape[0] // 2, g.val.shape[1] // 2])

    assert value(1.32, 1.07) == 1  # the middle of the box
    assert value(0.70, 0.50) == 0  # the rover itself
    assert value(0.30, 1.20) == 0  # open floor


def test_a_measured_scale_is_not_changed_by_the_move() -> None:
    f, reader = fusion(scale_error=1.0, measured=True)

    stop_at(f, reader, 0.50, (20.0, 20.0, 0.0))
    stop_at(f, reader, 0.70, (20.2, 20.0, 0.0))

    assert f.drone['cale_par'] == 'movement'
    assert f.camera.settings.metres_per_pixel == TRUE_SCALE
    assert f.calage_mvt[0] == pytest.approx(0.0, abs=math.radians(3))
    assert f.carte.tuiles['drone']  # same scale: placed at once


def test_a_move_too_short_or_not_seen_does_not_place_the_map() -> None:
    f, reader = fusion()
    stop_at(f, reader, 0.50, (20.0, 20.0, 0.0))

    stop_at(f, reader, 0.55, (20.05, 20.0, 0.0))  # 5 cm: under --drone-pas-min
    assert not f.drone['cale']

    stop_at(f, reader, 0.50, (20.3, 20.0, 0.0))  # the odometry moved, the picture did not
    assert not f.drone['cale']
    assert 'did not see exactly one object move' in f.drone['mouvement']


def test_reset_forgets_the_placement() -> None:
    f, reader = fusion()
    stop_at(f, reader, 0.50, (20.0, 20.0, 0.0))
    stop_at(f, reader, 0.70, (20.2, 20.0, 0.0))
    assert f.calage is not None

    f.reinitialiser()

    assert f.calage is None and f.calage_mvt is None and f.appui is None
    assert f.cale_par is None and f.correction_echelle == 1.0


def test_the_height_given_does_not_matter() -> None:
    """A starting scale three times too large (a height three times too high): the rover's
    move still places the map at the true scale, and the height found is reported."""
    f, reader = fusion(scale_error=3.0)

    stop_at(f, reader, 0.50, (20.0, 20.0, 0.0))
    stop_at(f, reader, 0.70, (20.2, 20.0, 0.0))

    assert f.calage_mvt is not None
    assert f.camera.settings.metres_per_pixel == pytest.approx(TRUE_SCALE, rel=0.04)
    # the picture is 780 px wide at 300 px per metre: 2.6 m seen with a 108 degree lens
    assert f.drone['hauteur_estimee_m'] == pytest.approx(
        2.6 / (2 * math.tan(math.radians(54))), rel=0.05
    )


def test_a_mover_that_is_not_the_size_of_the_rover_is_ignored() -> None:
    f, reader = fusion()
    small = lambda x: render_top_down(  # noqa: E731
        Scene(
            marker_positions={}, boxes=[BOX, Box(x - 0.05, 0.45, 0.10, 0.10, color=(30, 30, 30))]
        )
    )
    reader.image = small(0.50)
    with f.verrou:
        f.pose, f.immobile = (20.0, 20.0, 0.0), True
    f.tour_drone()
    reader.image = small(0.70)
    with f.verrou:
        f.pose = (20.2, 20.0, 0.0)
    f.tour_drone()

    assert not f.drone['cale']
    assert 'not like the rover' in f.drone['mouvement']


def test_the_placement_by_move_is_dropped_when_the_camera_itself_moves() -> None:
    """The phone is taken in hand: everything shifts in its picture. The placement found from
    the rover's move no longer holds, and the last placed view stays, marked as old. (With
    --drone-carte instantanee; cumulee places such a picture by matching it to the last one,
    see test_the_drone_map_grows_when_the_camera_moves.)"""
    f, reader = fusion()
    f.args.drone_carte = 'instantanee'
    stop_at(f, reader, 0.50, (20.0, 20.0, 0.0))
    stop_at(f, reader, 0.70, (20.2, 20.0, 0.0))
    assert f.drone['cale_sur_cette_image'] and f.drone['cale_par'] == 'movement'

    shifted = picture(0.70)
    shifted = np.roll(shifted, 90, axis=1)  # the whole picture slides 30 cm
    reader.image = shifted
    f.tour_drone()

    assert f.calage_mvt is None
    assert not f.drone['cale_sur_cette_image'] and f.drone['cale']  # the old view stays


# ---- the vision model (--drone-detecteur ia) together with the rover's move -------------------


class Trouve:
    """What sim/rover_du_dessus.py returns: the rover's box in the picture (pixels, y down)."""

    def __init__(self, x, y, cap=None):
        self.x, self.y, self.cap = x, y, cap
        self.taille, self.score = 0.215 * 300, 0.8  # sqrt(0.20 x 0.23) m at 300 px per metre


def fusion_ia(found):
    """A FusionReelle in vision-model mode; `found[0]` is what the model answers."""
    f, reader = fusion()
    f.args.drone_taille_rover = 0.215
    f.ia = lambda image: found[0]
    return f, reader


def model_stop(f, reader, found, rover_x, pose, cap=None):
    """The rover stands still at `pose`; the model finds it at scene point (rover_x, 0.50)."""
    image = picture(rover_x)
    found[0] = Trouve((rover_x + SHIFT) * 300, image.shape[0] - (0.50 + SHIFT) * 300, cap)
    reader.image = image
    with f.verrou:
        f.pose, f.immobile = pose, True
    f.tour_drone()


def test_with_the_vision_model_a_rover_without_heading_is_placed_by_its_move() -> None:
    """The model finds the rover in every picture but not which way it faces. Before: never
    placed. Now two stops of the rover give the direction of the camera's frame."""
    found = [None]
    f, reader = fusion_ia(found)
    heading = math.pi / 2

    model_stop(f, reader, found, 0.50, (20.0, 20.0, heading))
    assert not f.drone['cale'] and f.drone['rover_vu'] and not f.drone['cap_vu']
    assert 'drive it a little' in f.drone['mouvement']

    model_stop(f, reader, found, 0.70, (20.0, 20.2, heading))
    assert f.drone['cale_sur_cette_image'] and f.drone['cale_par'] == 'movement'
    assert f.rotation_ia == pytest.approx(math.pi / 2, abs=math.radians(3))
    assert f.carte.tuiles['drone']
    px, py = 20.0 - (1.07 - 0.50), 20.2 + (1.32 - 0.70)  # the middle of the box, on the maps
    g = f.carte.grille(px - 0.05, py - 0.05, px + 0.05, py + 0.05)
    assert int(g.val[g.val.shape[0] // 2, g.val.shape[1] // 2]) == 1
    # the move-by-objects placement stays out of the way in this mode
    assert f.calage_mvt is None and f.appui is None


def test_with_the_vision_model_a_heading_seen_in_the_picture_wins() -> None:
    found = [None]
    f, reader = fusion_ia(found)
    f.args.drone_sans_marqueur = False  # trust the marker

    # heading in the picture: 0 = x right; the maps' frame is the picture's here
    model_stop(f, reader, found, 0.50, (20.0, 20.0, 0.0), cap=0.0)

    assert f.drone['cale_sur_cette_image'] and f.drone['cale_par'] == 'marker'
    assert f.rotation_ia is None  # no move was needed


def test_with_the_vision_model_a_move_that_the_pictures_contradict_is_refused() -> None:
    """The rover drove 20 cm but is 60 cm further in the picture: the camera moved."""
    found = [None]
    f, reader = fusion_ia(found)

    model_stop(f, reader, found, 0.30, (20.0, 20.0, 0.0))
    model_stop(f, reader, found, 0.90, (20.2, 20.0, 0.0))

    assert not f.drone['cale'] and f.rotation_ia is None
    assert 'did the camera move' in f.drone['mouvement']


def test_with_the_vision_model_a_rover_not_found_changes_nothing() -> None:
    found = [None]
    f, reader = fusion_ia(found)
    reader.image = picture(0.50)
    with f.verrou:
        f.pose, f.immobile = (20.0, 20.0, 0.0), True

    f.tour_drone()  # the model answers None

    assert f.drone['etat'] == 'ok' and not f.drone['rover_vu'] and not f.drone['cale']


def test_without_a_marker_a_heading_read_in_the_picture_is_ignored() -> None:
    """--drone-sans-marqueur (on by default): the model reports a heading, from a white patch
    that is not a marker, and it is wrong. It must not place the map; the rover's move does."""
    found = [None]
    f, reader = fusion_ia(found)
    assert f.args.drone_sans_marqueur
    heading = math.pi / 2

    model_stop(f, reader, found, 0.50, (20.0, 20.0, heading), cap=2.0)  # a wrong heading
    assert not f.drone['cale'] and not f.drone['cap_vu']

    model_stop(f, reader, found, 0.70, (20.0, 20.2, heading), cap=2.0)
    assert f.drone['cale_par'] == 'movement'
    assert f.rotation_ia == pytest.approx(math.pi / 2, abs=math.radians(3))


def test_the_marker_is_off_by_default_and_can_be_turned_back_on() -> None:
    assert fr.options([]).drone_sans_marqueur is True
    assert fr.options(['--no-drone-sans-marqueur']).drone_sans_marqueur is False


def test_the_marker_switch_can_be_set_while_running() -> None:
    """POST /marqueur, the dashboard's "No marker" box."""
    f, _ = fusion()

    assert f.regler_marqueur({'sans_marqueur': False}) == (
        200,
        {'ok': True, 'sans_marqueur': False},
    )
    assert f.args.drone_sans_marqueur is False
    assert f.regler_marqueur({'sans_marqueur': True})[1]['sans_marqueur'] is True
    assert f.args.drone_sans_marqueur is True
    for bad in ({}, {'sans_marqueur': 'yes'}, {'sans_marqueur': 1}):
        assert f.regler_marqueur(bad)[0] == 400
    assert f.args.drone_sans_marqueur is True


def test_the_switch_changes_how_the_next_picture_is_placed() -> None:
    found = [None]
    f, reader = fusion_ia(found)

    model_stop(f, reader, found, 0.50, (20.0, 20.0, 0.0), cap=0.0)
    assert not f.drone['cale']  # no marker trusted: one picture does not place the map

    f.regler_marqueur({'sans_marqueur': False})
    model_stop(f, reader, found, 0.50, (20.0, 20.0, 0.0), cap=0.0)
    assert f.drone['cale'] and f.drone['cale_par'] == 'marker'


# ---- the drone map grows as the camera moves (--drone-carte cumulee) ----------------------
BOX2 = Box(2.40, 0.30, 0.25, 0.35, color=(40, 60, 200))  # only in view once the camera moved
WIDE_COLS = 780  # the camera's picture: as wide as the pictures above


def textured(image, seed=1):
    """Faint specks on the floor (about 1.5 cm, a little lighter or darker than it), as a
    real floor has: too faint to be obstacles, enough to match one picture to the next."""
    out = image.copy()
    rng = np.random.default_rng(seed)
    floor = image[0, 0].astype(int)
    for _ in range(1500):
        u, v = int(rng.integers(0, out.shape[1])), int(rng.integers(0, out.shape[0]))
        if (np.abs(out[v, u].astype(int) - floor) > 30).any():
            continue  # on a box or the rover: leave it
        shade = tuple(int(c) for c in np.clip(floor + rng.choice((-28, 28)), 0, 255))
        cv2.circle(out, (u, v), 2, shade, -1)
    return out


def wide_world(rover_x, rover_y=0.50):
    """The whole scene, 3.6 m wide: the camera sees 780 px (2.6 m) of it at a time."""
    rover = Box(rover_x - 0.10, rover_y - 0.115, 0.20, 0.23, color=(30, 30, 30))
    scene = Scene(
        bounds=MapBounds(0.0, 0.0, 3.0, 1.5), marker_positions={}, boxes=[BOX, BOX2, rover]
    )
    return textured(render_top_down(scene))


def camera_at(rover_x, shift_px):
    """What the camera sees when it has slid `shift_px` pixels (1/300 m each) to the right."""
    return np.ascontiguousarray(wide_world(rover_x)[:, shift_px : shift_px + WIDE_COLS])


def drone_value(f, x, y):
    """The drone's layer at the point (x, y) of the maps."""
    with f.verrou:
        couche, x0, y0 = f.couche_drone((x - 0.05, y - 0.05, x + 0.05, y + 0.05))
    i, j = int((y - y0) / 0.05), int((x - x0) / 0.05)
    return int(couche[i, j])


def test_recaler_images_finds_how_the_floor_moved() -> None:
    """Two views of the same floor, the second taken 0.5 m further right and turned by 10
    degrees: the match gives that move back."""
    res = 1.0 / 300.0
    world = wide_world(0.70)
    a = np.ascontiguousarray(world[:, :WIDE_COLS])
    b = np.ascontiguousarray(world[:, 150 : 150 + WIDE_COLS])
    h, w = a.shape[:2]
    bounds = MapBounds(0.0, 0.0, w * res, h * res)

    rotation, scale, tx, ty, n = fr.recaler_images(a, bounds, b, bounds, res)
    assert n >= 12 and scale == pytest.approx(1.0, abs=0.01)
    assert rotation == pytest.approx(0.0, abs=math.radians(0.5))
    assert (tx, ty) == pytest.approx((0.5, 0.0), abs=0.01)  # B's origin is 0.5 m right of A's

    # the same second view, turned by 10 degrees about its centre
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), 10.0, 1.0)  # counter-clockwise on screen
    b_turned = cv2.warpAffine(b, turn, (w, h), borderValue=(200, 200, 200))
    rotation, scale, tx, ty, n = fr.recaler_images(a, bounds, b_turned, bounds, res)
    assert scale == pytest.approx(1.0, abs=0.02)
    # turned counter-clockwise on screen (y down) is turned counter-clockwise in the metric
    # frame (y up) too, so B's points are turned back by -10 degrees to land on A
    assert rotation == pytest.approx(math.radians(-10.0), abs=math.radians(1.0))


def test_recaler_images_on_a_bare_floor_gives_up() -> None:
    res = 1.0 / 300.0
    floor = np.full((630, 780, 3), 200, np.uint8)
    bounds = MapBounds(0.0, 0.0, 780 * res, 630 * res)
    assert fr.recaler_images(floor, bounds, floor, bounds, res) is None


def test_calage_enchaine_puts_a_point_where_the_two_placements_agree() -> None:
    calage_a = (0.7, 1.0, 0.5, 20.0, 21.0)  # placement of the reference picture
    recalage = (0.3, 1.0, 0.4, -0.2, 50)  # p_ref = R(0.3) p + (0.4, -0.2)
    calage_b = fr.calage_enchaine(calage_a, recalage)

    def on_maps(calage, p):
        th, cx, cy, mx, my = calage
        c, s = math.cos(th), math.sin(th)
        return mx + c * (p[0] - cx) - s * (p[1] - cy), my + s * (p[0] - cx) + c * (p[1] - cy)

    for p in ((0.0, 0.0), (1.2, 0.3), (-0.5, 2.0)):
        c, s = math.cos(0.3), math.sin(0.3)
        p_ref = (c * p[0] - s * p[1] + 0.4, s * p[0] + c * p[1] - 0.2)
        assert on_maps(calage_b, p) == pytest.approx(on_maps(calage_a, p_ref), abs=1e-9)


def place_with_a_move(f, reader):
    """The camera at its first place: the rover's move (no marker) places the drone map."""
    for rover_x, pose in ((0.50, (20.0, 20.0, 0.0)), (0.70, (20.2, 20.0, 0.0))):
        reader.image = camera_at(rover_x, 0)
        with f.verrou:
            f.pose, f.immobile = pose, True
        f.tour_drone()
    assert f.drone['cale'] and f.drone['cale_par'] == 'movement'


def test_the_drone_map_grows_when_the_camera_moves() -> None:
    """The camera slides 50 cm to the right while the rover stands still. The rover's move no
    longer holds (everything shifted in the picture), but the picture matches the last placed
    one: it is placed, what was mapped before stays, and a box only seen now lands at its
    place on the maps."""
    f, reader = fusion()
    place_with_a_move(f, reader)
    # scene (x, y) -> maps: the rover went from x 0.50 to 0.70 while its pose went 20.0 -> 20.2
    box = (1.325 + 19.5, 1.075 + 19.5)
    box2 = (2.525 + 19.5, 0.475 + 19.5)
    assert drone_value(f, *box) == 1
    assert drone_value(f, *box2) == -1  # not in view yet
    surface = f.surface_drone()

    reader.image = camera_at(0.70, 150)  # same rover, the camera moved 0.5 m
    f.tour_drone()

    assert f.drone['cale_sur_cette_image'] and f.drone['cale_par'] == 'pictures'
    assert 'placed on the last placed picture' in f.drone['recalage']
    assert drone_value(f, *box2) == 1  # the new area is on the map, at its place
    assert drone_value(f, *box) == 1  # the box seen before is still at the same place
    assert drone_value(f, 0.0 + 19.5 - 0.2, 0.75 + 19.5) == 0  # left edge: out of view now, kept
    assert f.surface_drone() > surface + 0.5  # about 0.5 m x 2.1 m more


def test_the_old_way_keeps_only_the_last_picture() -> None:
    f, reader = fusion()
    f.args.drone_carte = 'instantanee'
    place_with_a_move(f, reader)

    reader.image = camera_at(0.70, 150)
    f.tour_drone()

    assert not f.drone['cale_sur_cette_image']  # no placing by pictures in this mode
    assert drone_value(f, 2.525 + 19.5, 0.475 + 19.5) == -1
