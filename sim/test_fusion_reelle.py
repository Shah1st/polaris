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


# ---- the rover's local map, on or off -----------------------------------------------------------


def test_the_local_map_is_off_by_default_and_the_rover_camera_is_not_asked() -> None:
    asked = []
    f = fr.FusionReelle(
        fr.options([]),
        requete=lambda methode, chemin, corps=None: {
            'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}
        },
        percevoir=lambda *a: asked.append(a),
        percevoir_frais=lambda *a: asked.append(a),
    )
    assert f.args.carte_locale is False

    f.tour()
    f.tour()  # the rover stands still: with the local map on, its camera would be asked now

    assert asked == []
    assert f.status['carte_locale'] is False
    assert 'local map off' in f.status['info']


def test_the_local_map_can_be_turned_on_and_off_while_running() -> None:
    view = (np.zeros((4, 4), np.int8), 20.0, 20.0)
    f = fr.FusionReelle(
        fr.options(['--carte-locale']),
        requete=lambda methode, chemin, corps=None: {
            'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}
        },
        percevoir=lambda *a: view,
        percevoir_frais=lambda *a: view,
    )
    f.tour()
    f.tour()
    assert f.carte.tuiles['rover'] and f.vues == 1  # fused once the rover stood still

    assert f.regler_carte_locale({'carte_locale': False}) == (
        200,
        {'ok': True, 'carte_locale': False},
    )
    assert not f.carte.tuiles['rover'] and f.vues == 0  # emptied
    f.tour()
    assert not f.carte.tuiles['rover']

    assert f.regler_carte_locale({'carte_locale': True})[0] == 200
    f.tour()
    assert f.carte.tuiles['rover']
    assert f.regler_carte_locale({'carte_locale': 'no'})[0] == 400


# ---- missions in long segments (--segment) and a drone map that grows (--drone-cumul) ------


class SegmentRover:
    """What _boucle_segments asks of avancer_esquive.RoverSegments: it follows the points it
    is given and remembers them."""

    def __init__(self, x, y):
        self.est, self.cap, self.segments = (x, y), 0.0, []

    def position(self):
        return self.est

    def suivre(self, points, duree):
        self.segments.append([tuple(p) for p in points])
        self.est = tuple(points[-1])
        return 'done', None

    def signaler_choc(self, impact):
        raise AssertionError('no bump expected')


def with_a_drone_map(extra=()):
    """A FusionReelle in segment mode whose drone has seen a free strip 3 m long and 1 m wide
    along x from the rover, and nothing else. Local map off: the rover's camera is not used."""
    f = fr.FusionReelle(
        fr.options(['--segment', '2.0', '--segment-vu', *extra]),
        requete=lambda *a: None,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    strip = np.zeros((20, 60), np.int8)  # 5 cm cells, all free and seen
    for msg in fr.decouper(strip, 19.5, 19.5, 4, f.carte, 'drone', t=1.0):
        f.carte.recevoir(msg)
    return f


def test_in_segments_the_rover_reaches_the_goal_in_a_few_commands() -> None:
    """1.8 m to drive: 8 stops step by step (25 cm each), one or two segments here."""
    import avancer_esquive as ae

    f = with_a_drone_map()
    f.objectif = (21.8, 20.0)
    rover = SegmentRover(20.0, 20.0)

    assert f._boucle_segments(rover, ae) == 'arrive'
    assert 1 <= len(rover.segments) <= 2
    assert math.dist(rover.est, f.objectif) <= f.args.tolerance + 0.1


def test_in_segments_the_rover_never_leaves_what_has_been_seen() -> None:
    """The goal is 2 m to the side, where the drone has seen nothing: every point of every
    segment stays inside the seen strip."""
    import avancer_esquive as ae

    f = with_a_drone_map()
    f.objectif = (20.5, 22.0)
    rover = SegmentRover(20.0, 20.0)

    assert f._boucle_segments(rover, ae) in ('bloque', 'max_pas')
    points = [p for s in rover.segments for p in s]
    assert all(19.5 <= x <= 22.5 and 19.5 <= y <= 20.5 for x, y in points)


def test_in_segments_without_a_drone_map_the_rover_does_not_move() -> None:
    import avancer_esquive as ae

    f = with_a_drone_map()
    with f.verrou:
        f.carte.tuiles['drone'] = {}
    f.objectif = (21.8, 20.0)
    rover = SegmentRover(20.0, 20.0)

    assert f._boucle_segments(rover, ae) == 'bloque'
    assert rover.segments == []


def test_the_drone_map_grows_with_a_camera_that_moves() -> None:
    """Two placed pictures of a camera that moved 3 m: with --drone-cumul both stay on the
    drone map; without it the second replaces the first."""
    for cumul, seen in ((True, 2 * 1600), (False, 1600)):
        f, _ = fusion()
        f.args.drone_cumul = cumul
        for calage in ((0.0, 1.0, 1.0, 20.0, 20.0), (0.0, 1.0, 1.0, 23.0, 20.0)):
            m = np.zeros((40, 40), np.int8)
            with f.verrou:
                place = f.placer_vue(m, 0.05, calage)
                if not cumul:
                    f.carte.tuiles['drone'], f.carte.maj['drone'] = {}, {}
                for msg in fr.decouper(place[0], place[1], place[2], 4, f.carte, 'drone', t=1.0):
                    f.carte.recevoir(msg)
        total, x0, y0 = f.matrice_drone()
        assert int((total != -1).sum()) == seen


def test_in_segments_unseen_cells_are_crossed_unless_segment_vu() -> None:
    """No drone map at all and no local map: step by step the rover drives to the clicked
    point on its odometry alone; in segments it must do the same (--segment-vu refuses)."""
    import avancer_esquive as ae

    f = fr.FusionReelle(
        fr.options(['--segment', '2.0']),
        requete=lambda *a: None,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    f.objectif = (21.8, 20.6)
    rover = SegmentRover(20.0, 20.0)

    assert f._boucle_segments(rover, ae) == 'arrive'
    assert 1 <= len(rover.segments) <= 2


def test_in_segments_a_segment_goes_round_an_obstacle_of_the_drone() -> None:
    """A box of the drone map across the straight line: no point of a segment, and no chord
    between two points, comes within the rover's radius of it."""
    import avancer_esquive as ae

    f = fr.FusionReelle(
        fr.options(['--segment', '2.0']),
        requete=lambda *a: None,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    m = np.zeros((60, 80), np.int8)  # 4 m x 3 m seen free, a 30 cm box on the way
    m[27:33, 37:43] = 1
    for msg in fr.decouper(m, 19.0, 18.5, 4, f.carte, 'drone', t=1.0):
        f.carte.recevoir(msg)
    f.objectif = (22.0, 20.0)
    rover = SegmentRover(20.0, 20.0)

    assert f._boucle_segments(rover, ae) == 'arrive'
    box = (21.0, 20.0)  # its centre; half side 0.15 m, rover radius 0.20 m
    trace = [(20.0, 20.0)] + [p for s in rover.segments for p in s]
    for a, b in zip(trace, trace[1:], strict=False):
        for k in range(21):
            x, y = a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20
            assert max(abs(x - box[0]), abs(y - box[1])) > 0.15 + 0.15


# ---- calibration: forward and back, seen by the drone (POST /calibrer) ---------------------


def calibrating(turn_deg=70.0, mm_per_px=4.0, camera_moves=False):
    """A FusionReelle whose rover and drone are stand-ins: the camera's frame is turned
    `turn_deg` from the maps' and one pixel is `mm_per_px` on the floor. `_rouler` moves the
    stand-in rover along its heading; `_rover_dans_l_image` says where the camera sees it."""
    posts = []
    f = fr.FusionReelle(
        fr.options([]),
        requete=lambda m, c, corps=None: posts.append((m, c)) or {'ok': True},
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    f.camera = type('Camera', (), {'settings': OverheadSettings(metres_per_pixel=0.0031)})()
    f.ia = lambda image: None
    pose = [20.0, 20.0, math.radians(30)]
    moves = []

    def rouler(distance):
        pose[0] += distance * math.cos(pose[2])
        pose[1] += distance * math.sin(pose[2])
        moves.append(distance)

    def dans_l_image(apres):
        th, k = math.radians(-turn_deg), 1000.0 / mm_per_px  # maps -> picture, pixels per m
        x, y = pose[0] - 18.0, pose[1] - 19.0
        px = (k * (math.cos(th) * x - math.sin(th) * y), k * (math.sin(th) * x + math.cos(th) * y))
        if camera_moves and len(moves) == 2:  # the hand moved before the last picture
            px = (px[0] + 40.0, px[1] + 70.0)
        return px, tuple(pose)

    f._rouler, f._rover_dans_l_image = rouler, dans_l_image
    for msg in fr.decouper(np.zeros((8, 8), np.int8), 20.0, 20.0, 4, f.carte, 'drone', t=1.0):
        f.carte.recevoir(msg)  # an older drone map, placed before the calibration
    return f, moves, posts


def test_the_calibration_drives_forward_and_back_and_finds_direction_and_scale() -> None:
    f, moves, posts = calibrating(turn_deg=70.0, mm_per_px=4.0)

    f._calibrage()

    assert moves == [pytest.approx(0.30), pytest.approx(-0.30)]  # back where it started
    assert f.mission['etat'] == 'arrive' and 'CALIBRATED' in f.mission['texte']
    assert f.rotation_ia == pytest.approx(math.radians(70.0), abs=math.radians(1))
    assert f.camera.settings.metres_per_pixel == pytest.approx(0.004, rel=0.01)
    assert f.mpp_calibre  # the rover's box no longer changes the scale
    assert not f.carte.tuiles['drone']  # the map placed before starts again
    assert ('POST', '/stop') not in posts


def test_the_calibration_is_refused_when_the_camera_moved_meanwhile() -> None:
    f, moves, posts = calibrating(camera_moves=True)

    f._calibrage()

    assert f.mission['etat'] == 'erreur' and 'hold the camera still' in f.mission['texte']
    assert f.rotation_ia is None and not f.mpp_calibre
    assert f.carte.tuiles['drone']  # nothing was thrown away
    assert ('POST', '/stop') in posts


def test_the_calibration_needs_the_drone_camera_and_no_mission() -> None:
    f, _, _ = calibrating()
    f.ia = None
    code, body = f.calibrer()
    assert code == 409 and 'vision model' in body['error']
    assert not f.en_mission()


def test_defaults_for_a_small_arena() -> None:
    """Asked for after the first drives: obstacles were grown too much to pass between them."""
    a = fr.options([])
    assert (a.rayon, a.pas, a.drone_ombres) == (0.15, 0.15, True)
    f, _ = fusion()
    assert f.carte.rayon_robot == 0.15 and f.carte.rayon_effectif == 0.15


# ---- the rover's own obstacle grid (--source-locale rover) and continuous driving ------------


def test_the_rover_grid_is_fused_at_its_place_also_while_the_rover_moves() -> None:
    """The rover is at odometry (1, 2) heading 90 degrees when the maps start: that is
    (20, 20) heading 0 on the maps. Its grid has one obstacle 0.5 m ahead of that pose."""
    cells = [[-1] * 50 for _ in range(50)]
    for i in range(50):
        for j in range(50):
            cells[i][j] = 0
    # odometry point (1, 2.5): 0.5 m ahead of the start pose; grid origin (-0.25, 0.75)
    cells[int((2.52 - 0.75) / 0.05)][int((1.02 + 0.25) / 0.05)] = 1
    local = {'res': 0.05, 'x0': -0.25, 'y0': 0.75, 'cells': cells, 'age_s': 0.3, 'error': None}

    def requete(methode, chemin, corps=None):
        if chemin == '/local_grid':
            return local
        return {'odom': {'x': 1.0, 'y': 2.0, 'heading_deg': 90.0, 'v': 0.2}}  # driving

    f = fr.FusionReelle(
        fr.options(['--carte-locale', '--source-locale', 'rover']),
        requete=requete,
        percevoir=lambda *a: pytest.fail('the perception service is not asked'),
        percevoir_frais=lambda *a: pytest.fail('the perception service is not asked'),
    )
    f.tour()
    g = f.carte.grille(20.4, 19.9, 20.6, 20.1)
    assert (g.val == 1).sum() == 1
    i, j = np.argwhere(g.val == 1)[0]
    x, y = g.xy(i, j)
    assert abs(x - 20.52) < 0.04 and abs(y - 19.98) < 0.04  # ahead of the rover, on its axis
    assert f.status['local']['source'] == 'rover' and f.status['local']['age_s'] == 0.3
    assert 'depth camera' in f.status['source_rover'] and f.status['info'] == "rover's grid fused"

    local['age_s'] = 60.0  # the feed died: nothing more is fused
    vues = f.vues
    f.tour()
    assert f.vues == vues and 'no recent grid' in f.status['info']


def test_continuous_driving_and_the_rover_grid_are_off_by_default() -> None:
    a = fr.options([])
    assert (a.continu, a.source_locale, a.carte_locale) == (False, 'service', False)
    f, _ = fusion()
    assert f.status.get('conduite') in (None, 'pas')


def test_driving_and_the_local_map_source_can_be_switched_while_running() -> None:
    f, _ = fusion()
    assert f.regler_conduite({'continu': True}) == (
        200,
        {'ok': True, 'continu': True, 'source_locale': 'service'},
    )
    assert f.args.continu is True
    code, answer = f.regler_conduite({'source_locale': 'rover', 'continu': False})
    assert code == 200 and (answer['continu'], answer['source_locale']) == (False, 'rover')
    assert f.local['source'] == 'rover' and f.args.source_locale == 'rover'
    for bad in ({}, {'continu': 'yes'}, {'source_locale': 'drone'}, {'continu': 1}):
        assert f.regler_conduite(bad)[0] == 400
    assert (f.args.continu, f.args.source_locale) == (False, 'rover')  # unchanged by those

    class Running:
        def is_alive(self):
            return True

    f.fil_mission = Running()  # a mission reads both: refused while it runs
    assert f.regler_conduite({'continu': True})[0] == 409
    assert f.args.continu is False


def test_an_older_rover_server_without_a_grid_is_said_not_fatal() -> None:
    def requete(methode, chemin, corps=None):
        if chemin == '/local_grid':
            raise fr.ServeurRoverError('GET /local_grid -> HTTP 404 {"error": "not found"}')
        return {'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}}

    f = fr.FusionReelle(
        fr.options(['--carte-locale', '--source-locale', 'rover']),
        requete=requete,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    f.tour()  # no exception: the maps keep working
    assert f.local['source'] == 'rover' and 'HTTP 404' in f.local['erreur']
    assert 'no recent grid' in f.status['info'] and f.status['etat'] == 'ok'


def test_the_mission_speed_follows_the_dashboard_slider() -> None:
    f = fr.FusionReelle(
        fr.options(['--vitesse', '0.3']),
        requete=lambda methode, chemin, corps=None: {
            'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}
        },
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    f.tour()
    assert f.status['vitesse'] == 0.3
    assert f.regler_vitesse({'vitesse': 0.15}) == (200, {'ok': True, 'vitesse': 0.15})
    f.tour()
    assert f.args.vitesse == 0.15 and f.status['vitesse'] == 0.15

    class Driver:  # a mission's rover driver: it reads .vitesse at each POST /path
        vitesse = 0.15

    f.rover_mission = Driver()
    assert f.regler_vitesse({'vitesse': 0.1})[0] == 200
    assert f.rover_mission.vitesse == 0.1  # the running mission gets it too
    for bad in ({}, {'vitesse': 'fast'}, {'vitesse': True}, {'vitesse': float('nan')},
                {'vitesse': 0.01}, {'vitesse': 0.5}):  # fmt: skip
        assert f.regler_vitesse(bad)[0] == 400
    assert f.args.vitesse == 0.1 and f.rover_mission.vitesse == 0.1  # unchanged by those


# ---- a place marked as an obstacle that turns out to be free --------------------------------


def cell(f, x, y):
    """The fused value (1 obstacle, 0 free, -1 unknown) of the 5 cm cell holding map (x, y)."""
    g = f.carte.grille(x - 0.2, y - 0.2, x + 0.2, y + 0.2)
    for i in range(g.val.shape[0]):
        for j in range(g.val.shape[1]):
            cx, cy = g.xy(i, j)
            if abs(cx - x) < 0.025 and abs(cy - y) < 0.025:
                return int(g.val[i, j])
    raise AssertionError('cell not in the window')


def test_an_obstacle_the_rover_sees_free_again_leaves_the_map_with_the_service() -> None:
    """Perception service (--source-locale service): the views of the rover's camera."""
    vues = []
    f = fr.FusionReelle(
        fr.options(['--carte-locale']),
        requete=lambda methode, chemin, corps=None: {
            'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}
        },
        percevoir=lambda *a: vues[-1],
        percevoir_frais=lambda *a: vues[-1],
    )
    m = np.zeros((8, 8), np.int8)
    m[4, 4] = 1  # map point (20.225, 20.225)
    vues.append((m, 20.0, 20.0))
    f.tour()
    f.tour()  # fused once the rover stood still
    assert cell(f, 20.225, 20.225) == 1
    vues.append((np.zeros((8, 8), np.int8), 20.0, 20.0))  # the same place, seen free
    f.tour()
    assert cell(f, 20.225, 20.225) == 0  # the obstacle is gone from the map
    m = np.full((8, 8), -1, np.int8)
    m[4, 4] = 1
    vues.append((m, 20.0, 20.0))
    f.tour()
    vues.append((np.full((8, 8), -1, np.int8), 20.0, 20.0))  # not seen any more: no news
    f.tour()
    assert cell(f, 20.225, 20.225) == 1  # kept: only seeing it free removes it


def test_an_obstacle_the_rover_sees_free_again_leaves_the_map_with_its_own_grid() -> None:
    """The rover's own depth grid (--source-locale rover, its GET /local_grid)."""
    cells = [[0] * 20 for _ in range(20)]
    cells[10][10] = 1  # odometry (0.525, 0.525) = map (20.525, 20.525)
    local = {'res': 0.05, 'x0': 0.0, 'y0': 0.0, 'cells': cells, 'age_s': 0.1, 'error': None}

    def requete(methode, chemin, corps=None):
        if chemin == '/local_grid':
            return local
        return {'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}}

    f = fr.FusionReelle(
        fr.options(['--carte-locale', '--source-locale', 'rover']),
        requete=requete,
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )
    f.tour()
    assert cell(f, 20.525, 20.525) == 1
    cells[10][10] = 0  # the rover's camera now sees the floor there
    f.tour()
    assert cell(f, 20.525, 20.525) == 0
    cells[10][10] = 1
    f.tour()
    cells[10][10] = -1  # the rover forgot it (not seen for local_hold s): no news
    f.tour()
    assert cell(f, 20.525, 20.525) == 1  # the fused map keeps it until it is seen free


def test_an_obstacle_reported_by_the_guard_expires_and_is_not_erased_before(monkeypatch) -> None:
    libre = (np.zeros((8, 8), np.int8), 20.0, 20.0)
    f = fr.FusionReelle(
        fr.options(['--carte-locale', '--duree-obstacle', '30']),
        requete=lambda methode, chemin, corps=None: {
            'odom': {'x': 0.0, 'y': 0.0, 'heading_deg': 0.0, 'v': 0.0}
        },
        percevoir=lambda *a: libre,
        percevoir_frais=lambda *a: libre,
    )
    t = [1000.0]
    monkeypatch.setattr(fr.time, 'monotonic', lambda: t[0])
    f.carte.signaler_obstacle(20.225, 20.225, rayon=0.1, t=t[0], duree=30.0)  # as a mission does
    assert cell(f, 20.225, 20.225) == 1
    t[0] += 5.0
    f.publier(libre)  # a view that says free there (it does not see what the guard saw)...
    assert cell(f, 20.225, 20.225) == 0
    f.carte.reposer_chocs(t[0])  # ...is undone before the next plan, as the mission loops do
    assert cell(f, 20.225, 20.225) == 1
    t[0] += 30.0  # its time is over: gone at the next turn, mission or not
    f.tour()
    f.tour()
    assert cell(f, 20.225, 20.225) == 0 and not f.carte.chocs_temporaires


class GuardedRover(SegmentRover):
    """A rover whose local guard ends segments as 'blocked': it drives up to the obstacle it
    alone sees (a wall at x = `wall`), or does not move when it is already there."""

    def __init__(self, x, y, wall):
        super().__init__(x, y)
        self.wall, self.fin = wall, None

    def suivre(self, points, duree):
        self.segments.append([tuple(p) for p in points])
        x, y = points[-1]
        if x < self.wall - 0.15:  # the segment ends before the wall: driven to its end
            self.est, self.fin = (x, y), {'status': 'done', 'distance': 1.0}
            return 'done', None
        stop = self.wall - 0.15
        moved = max(0.0, stop - self.est[0])
        self.est = (max(self.est[0], stop), self.est[1])
        self.fin = {'status': 'blocked', 'obstacle': [self.wall, self.est[1]], 'distance': moved}
        return 'blocked', None

    def obstacle_vu(self, entree):
        o = entree.get('obstacle')
        return None if entree.get('status') != 'blocked' or not o else (o[0], o[1])


def test_in_segments_what_the_guard_stops_for_goes_on_the_map_and_changes_the_route() -> None:
    """The first real runs: the guard ended the segment ('blocked'), the loop did not read it
    and sent the same segment again and again while the rover stood still."""
    import avancer_esquive as ae

    f = with_a_drone_map(['--max-pas', '8'])
    f.objectif = (21.8, 20.0)
    rover = GuardedRover(20.0, 20.0, wall=21.0)
    issue = f._boucle_segments(rover, ae)
    assert 'OBSTACLE seen by the rover' in f.mission['texte'] or issue == 'bloque'
    assert cell(f, 21.025, 20.025) == 1  # what the guard saw is on the fused map
    assert rover.segments[1] != rover.segments[0]  # a new route, not the same segment again
    assert len(rover.segments) < 8  # and no endless resend


def test_in_segments_three_refusals_before_moving_end_the_mission_blocked() -> None:
    import avancer_esquive as ae

    f = with_a_drone_map(['--max-pas', '20'])
    f.objectif = (21.8, 20.0)

    class Stuck(GuardedRover):  # its guard reports an obstacle the planner cannot plan round:
        def obstacle_vu(self, entree):  # always behind the rover, never on the way
            return None if entree.get('status') != 'blocked' else (19.6, 20.0)

    rover = Stuck(20.85, 20.0, wall=21.0)  # already standing at the wall: 0 m at each try
    assert f._boucle_segments(rover, ae) == 'bloque'
    assert len(rover.segments) == 3  # three tries, then it stops asking
    assert "rover's guard refuses every way" in f.mission['texte']
    assert 'move the rover back by hand' in f.mission['texte']
