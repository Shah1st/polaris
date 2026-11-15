"""Tests of the overhead-camera launcher, with a fake camera that has the same methods as
`guidance_global_map.drone.OverheadDrone` (branch overhead-camera) and a fake rover server."""

import argparse
import math

import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from lancer_overhead import (
    REGLAGES_PETITE_ZONE,
    etalonner_echelle,
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


def test_a_live_camera_without_a_scale_needs_the_rover_check(capsys) -> None:
    """--camera with only a height is fine: the rover measures the scale at the start. Without
    that check, the scale has to be given."""
    with pytest.raises(SystemExit):
        main(
            ['--camera', 'http://phone/video', '--rover', 'http://r', '--victime', '1', '1']
            + ['--sans-etalonnage']
        )
    assert '--metres-par-pixel' in capsys.readouterr().err


def test_camera_and_image_cannot_both_be_given(capsys) -> None:
    with pytest.raises(SystemExit):
        main(['--camera', '0', '--image', 'a.jpg', '--rover', 'http://r', '--victime', '1', '1'])
    assert 'not allowed with' in capsys.readouterr().err


# ---- the camera's scale checked on a move of the rover ----------------------------------------


class Vu:
    """Like guidance_global_map.scale_check.Seen."""

    def __init__(self, x, y, short, long):
        self.x, self.y, self.short, self.long = x, y, short, long


class Reglages:
    def __init__(self, metres_per_pixel):
        self.metres_per_pixel = metres_per_pixel
        self.rover_size = (0.20, 0.23)
        self.camera_height, self.focal_length_35mm = 1.9, 26.0
        self.sensor_size = (4032, 3024)


class CameraAEchelle(FausseCamera):
    """A camera whose scale is wrong by `erreur`: everything it reports is `erreur` times too
    far and too large, as with a wrong camera height. Has what etalonner_echelle needs."""

    def __init__(self, rover, erreur=1.0, donnee=False, taille=(0.20, 0.23)):
        super().__init__(rover)
        self.erreur, self.taille = erreur, taille
        self.settings = Reglages(0.001 if donnee else None)
        self.immobile = False  # True: the camera does not see the rover move

    def objects(self, force=False):
        e = self.erreur
        x, y = (0.4, 0.4) if self.immobile else (self.rover.x, self.rover.y)
        return [
            Vu(1.2 * e, 1.2 * e, 0.4 * e, 0.6 * e),  # the box, which does not move
            Vu(1.9 * e, 0.5 * e, 0.13 * e, 0.15 * e),  # a pack nearly the rover's size
            Vu(x * e, y * e, self.taille[0] * e, self.taille[1] * e),
        ]

    def current_scale(self):
        return 0.001 * self.erreur

    def rescale(self, correction):
        self.erreur *= correction
        self.settings.metres_per_pixel = self.current_scale()
        return self.current_scale()

    def rover_pose(self):
        return self.rover.x * self.erreur, self.rover.y * self.erreur, None


def etalonner(camera, faux, lignes):
    return etalonner_echelle(camera, faux.requete, lignes.append, faux.dormir, faux.horloge)


def test_scale_is_measured_on_the_rover_move_when_only_the_height_was_given() -> None:
    """The scale from an approximate height is 1.55 times too large. The rover drives 15 cm
    and back; the camera's scale is corrected, and the rover is back where it started."""
    faux = FauxServeur(0.4, 0.4, math.radians(40))
    camera = CameraAEchelle(faux, erreur=1.55)
    lignes = []

    resultat = etalonner(camera, faux, lignes)

    assert resultat['adoptee']
    assert camera.erreur == pytest.approx(1.0, abs=0.03)  # the wheels read 2 % long
    assert resultat['metres_par_pixel'] == pytest.approx(0.001, abs=0.00003)
    assert math.dist((faux.x, faux.y), (0.4, 0.4)) < 0.01
    assert [c['distance'] for _, m, ch, c in faux.requetes if ch == '/move'] == [0.15, -0.15]
    texte = '\n'.join(lignes)
    assert 'Echelle mesuree' in texte and '--metres-par-pixel 0.00' in texte
    assert 'ATTENTION' not in texte


def test_a_measured_scale_is_kept_when_the_rover_confirms_it() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    camera = CameraAEchelle(faux, erreur=1.04, donnee=True)
    lignes = []

    resultat = etalonner(camera, faux, lignes)

    assert not resultat['adoptee'] and camera.erreur == 1.04
    assert 'Echelle confirmee' in '\n'.join(lignes)


def test_a_measured_scale_that_the_rover_contradicts_stops_the_mission() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    camera = CameraAEchelle(faux, erreur=1.55, donnee=True)

    with pytest.raises(RuntimeError, match='Echelle fausse'):
        etalonner(camera, faux, [])
    assert camera.erreur == 1.55


def test_no_scale_check_when_the_camera_does_not_see_the_rover_move() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    camera = CameraAEchelle(faux)
    camera.immobile = True

    with pytest.raises(RuntimeError, match="n'a pas vu le rover bouger"):
        etalonner(camera, faux, [])


def test_scale_is_measured_backwards_when_the_rover_faces_a_wall() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0, mur_x=0.42)
    camera = CameraAEchelle(faux, erreur=0.8)
    lignes = []

    resultat = etalonner(camera, faux, lignes)

    assert resultat['adoptee'] and camera.erreur == pytest.approx(1.0, abs=0.03)
    assert 'non mesuree' in '\n'.join(lignes)


def test_a_mover_that_is_not_the_size_of_the_rover_is_reported() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    camera = CameraAEchelle(faux, taille=(0.10, 0.12))
    lignes = []

    etalonner(camera, faux, lignes)

    assert 'pas la taille du rover' in '\n'.join(lignes)


def test_mission_starts_with_the_scale_check_and_plans_at_the_corrected_scale() -> None:
    faux = FauxServeur(0.4, 0.4, math.pi)
    camera = CameraAEchelle(faux, erreur=1.3)
    lignes = []

    lancer(
        camera,
        faux.requete,
        victime=(2.0, 2.0),
        options_rover={'horloge': faux.horloge, 'dormir': faux.dormir},
        journal=lignes.append,
    )

    texte = '\n'.join(lignes)
    assert texte.index('Echelle mesuree') < texte.index('Zone vue par la camera')
    assert camera.erreur == pytest.approx(1.0, abs=0.03)
    assert 'ARRIVE pres de la victime' in texte
    assert math.dist((faux.x, faux.y), (2.0, 2.0)) <= 0.25


def test_the_scale_check_can_be_skipped() -> None:
    faux = FauxServeur(0.4, 0.4, math.pi)
    camera = CameraAEchelle(faux)
    lignes = []

    lancer(
        camera,
        faux.requete,
        victime=(2.0, 2.0),
        options_rover={'horloge': faux.horloge, 'dormir': faux.dormir},
        journal=lignes.append,
        etalonner=False,
    )

    assert not any('Echelle mesuree' in ligne or 'confirmee' in ligne for ligne in lignes)
    assert [c for _, m, ch, c in faux.requetes if ch == '/move' and c['distance'] < 0] == []


def test_a_rover_server_restart_during_the_scale_check_is_reported() -> None:
    faux = FauxServeur(0.4, 0.4, 0.0)
    faux.boot_id = 'a'
    camera = CameraAEchelle(faux)
    envoyer = faux.requete

    def requete(methode, chemin, corps=None):
        reponse = envoyer(methode, chemin, corps)
        if chemin == '/move':
            faux.boot_id = 'b'  # the server restarts right after taking the order
        return reponse

    with pytest.raises(RuntimeError, match='redemarre pendant la verification'):
        etalonner_echelle(camera, requete, lambda _: None, faux.dormir, faux.horloge)
