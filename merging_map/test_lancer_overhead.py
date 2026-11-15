"""Tests of the overhead-camera launcher, with a fake camera that has the same methods as
`guidance_global_map.drone.OverheadDrone` (branch overhead-camera) and a fake rover server."""

import math

import numpy as np
import pytest
from fusion_carte import CarteFusion, decouper
from lancer_overhead import REGLAGES_PETITE_ZONE, lancer, preparer
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
