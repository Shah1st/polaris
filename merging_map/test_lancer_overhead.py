"""Tests of the overhead-camera launcher, with a fake camera that has the same methods as
`guidance_global_map.drone.OverheadDrone` (branch overhead-camera) and a fake rover server."""

import argparse
import math

import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from lancer_overhead import (
    REGLAGES_PETITE_ZONE,
    PhotoUnique,
    lancer,
    main,
    preparer,
    rafraichir_apres,
    reglages_camera,
)
from test_rover_reel import FauxServeur

RES = 0.05
COTE = 2.4  # the floor seen by the camera, in metres
BOITE = (1.0, 0.9, 1.4, 1.5)  # an obstacle between the rover and the victim (x0, y0, x1, y1)


class FausseCamera:
    """Same interface as OverheadDrone: tiles(), disponible(), zoomer(), rover_pose(), zone().
    Like the real one without a marker on the rover, it gives the rover's heading as None."""

    def __init__(self, rover):
        self.rover = rover
        n = round(COTE / RES)
        self.grille = np.zeros((n, n), np.int8)
        x0, y0, x1, y1 = BOITE
        self.grille[round(y0 / RES) : round(y1 / RES), round(x0 / RES) : round(x1 / RES)] = 1

    def tiles(self, force=False):
        return decouper(self.grille, 0.0, 0.0, 4, CarteFusion(), 'drone', self.rover.t)

    def disponible(self):
        return True

    def zoomer(self, tuile):
        return self.tiles(force=True)

    def rover_pose(self):
        return self.rover.x, self.rover.y, None

    def zone(self):
        return (0.0, 0.0, COTE, COTE)


def dans_boite(x, y):
    x0, y0, x1, y1 = BOITE
    return x0 <= x <= x1 and y0 <= y <= y1


def test_small_area_settings_are_used() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    carte, rover = preparer(FausseCamera(faux), faux.requete)
    assert carte.niveau_plan == REGLAGES_PETITE_ZONE['niveau_plan'] == 4
    assert carte.rayon_robot == 0.155  # circle around the whole 0.20 x 0.23 m rover
    assert carte.tuiles['drone']  # first survey loaded
    assert rover.position_drone() == (0.4, 0.4, None)


def test_points_outside_the_camera_view_are_refused() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    with pytest.raises(ValueError, match='Repere attendu'):
        lancer(FausseCamera(faux), faux.requete, victime=(3.0, 1.0), journal=lambda _: None)


def test_mission_with_the_overhead_camera_as_drone() -> None:
    """Rover seen without heading, start taken from the camera, obstacle in the way: the
    rover goes around it and stops within 15 cm of the victim."""
    faux = FauxServeur(0.4, 0.4, math.pi)  # faces west, away from the goal
    lignes = []
    lancer(
        FausseCamera(faux),
        faux.requete,
        victime=(2.0, 2.0),
        options_rover={'horloge': faux.horloge, 'dormir': faux.dormir},
        journal=lignes.append,
    )
    texte = '\n'.join(lignes)
    assert 'ARRIVE pres de la victime' in texte
    assert 'LIAISON DRONE PERDUE' not in texte  # the camera counts as the drone
    assert math.dist((faux.x, faux.y), (2.0, 2.0)) <= 0.2
    assert not any(dans_boite(x, y) for x, y in faux.trace)  # went around the obstacle


def test_live_camera_image_is_taken_after_the_stream_delay() -> None:
    """A live stream is late: the picture that locates the stopped rover is taken only after
    the stream's delay has passed."""
    ordre = []

    class Camera:
        def refresh(self, force=False):
            ordre.append(('image', force))

    rafraichir_apres(Camera(), 0.5, lambda s: ordre.append(('attente', s)))()

    assert ordre == [('attente', 0.5), ('image', True)]


def test_a_live_camera_refresh_given_by_the_caller_is_kept() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    camera = FausseCamera(faux)
    camera.refresh = lambda force=False: None  # like OverheadDrone
    mien = lambda: None  # noqa: E731

    _, rover = preparer(camera, faux.requete, options_rover={'rafraichir_drone': mien})

    assert rover.rafraichir_drone is mien


def test_camera_settings_from_the_command_line() -> None:
    args = argparse.Namespace(hauteur=2.4, metres_par_pixel=None, marqueur=None)
    assert reglages_camera(args) == {'camera_height': 2.4}
    args = argparse.Namespace(hauteur=1.9, metres_par_pixel=0.0012, marqueur=10)
    assert reglages_camera(args) == {
        'camera_height': 1.9,
        'metres_per_pixel': 0.0012,
        'rover_marker_id': 10,
    }


def test_a_live_camera_needs_its_own_scale(capsys) -> None:
    pytest.importorskip('guidance_global_map')
    with pytest.raises(SystemExit):
        main(['--camera', 'http://phone/video', '--rover', 'http://r', '--victime', '1', '1'])
    assert '--metres-par-pixel' in capsys.readouterr().err


def test_camera_and_image_cannot_both_be_given(capsys) -> None:
    with pytest.raises(SystemExit):
        main(['--camera', '0', '--image', 'a.jpg', '--rover', 'http://r', '--victime', '1', '1'])
    assert 'not allowed with' in capsys.readouterr().err


# -- a single overhead photo (no live camera) --------------------------------------------


def test_single_photo_gives_the_start_once_then_nothing() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    photo = PhotoUnique(FausseCamera(faux), cap=math.pi / 2)
    assert photo.disponible() and photo.rover_pose() == (0.4, 0.4, math.pi / 2)
    assert photo.tiles() and photo.zone() == (0.0, 0.0, COTE, COTE)
    photo.prise()  # the rover's first look, stopped at its start
    assert photo.rover_pose() == (0.4, 0.4, math.pi / 2)
    photo.prise()  # after a step: the photo would show the rover where it started
    assert not photo.disponible() and photo.rover_pose() is None
    assert photo.zoomer((4, 0, 0)) == []
    assert photo.tiles()  # the map stays


def test_mission_on_odometry_after_a_single_photo() -> None:
    """Start and heading from the photo only, then wheels + gyro: the rover still goes around
    the obstacle and reaches the victim, and the photo is never read as a new position."""
    faux = FauxServeur(0.4, 0.4, math.pi)  # faces west, away from the goal
    photo = PhotoUnique(FausseCamera(faux), cap=math.pi)
    lignes = []
    lancer(
        photo,
        faux.requete,
        victime=(2.0, 2.0),
        options_rover={
            'horloge': faux.horloge,
            'dormir': faux.dormir,
            'rafraichir_drone': photo.prise,
        },
        journal=lignes.append,
    )
    assert any('ARRIVE pres de la victime' in ligne for ligne in lignes)
    assert math.dist((faux.x, faux.y), (2.0, 2.0)) <= 0.2
    assert not any(dans_boite(x, y) for x, y in faux.trace)
    assert photo.mesures >= 2  # asked again after steps, and given nothing


def test_single_photo_needs_a_heading(capsys) -> None:
    with pytest.raises(SystemExit):
        main(['--image', 'x.png', '--photo-unique', '--victime', '1', '1', '--rover', 'http://r'])
    assert '--cap' in capsys.readouterr().err
