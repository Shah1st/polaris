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
    the rover's move no longer holds, and the last placed view stays, marked as old."""
    f, reader = fusion()
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
