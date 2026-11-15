"""Tests of the real-rover adapter against a fake rover that obeys the ESP32 commands."""

import math

import cerveau
import numpy as np
from fusion_carte import CarteFusion, decouper
from rover_reel import RaspRoverReel, angle_normalise, commande_texte


class FauxRover:
    """What the ESP32 would do: a differential robot driven by T:13 / T:1 commands, with
    wheel odometry, an optional wall it bumps into, and a fake clock."""

    def __init__(self, x, y, cap, mur_x=None, voie=0.125):
        self.x, self.y, self.cap, self.mur_x, self.voie = x, y, cap, mur_x, voie
        self.v = self.w = 0.0
        self.odl = self.odr = 0.0
        self.t = 0.0
        self.commandes = []  # (t, commande)
        self.joignable = True
        self.drone_ok = True

    def envoyer(self, commande):
        self.commandes.append((self.t, commande))
        if not self.joignable:
            return False
        if commande['T'] == 13:
            self.v, self.w = commande['X'], commande['Z']
        elif commande['T'] == 1:
            self.v = self.w = 0.0
        return True

    def dormir(self, dt):
        for _ in range(10):
            h = dt / 10
            nx = self.x + self.v * math.cos(self.cap) * h
            if self.mur_x is not None and nx >= self.mur_x:
                continue  # against the wall: the wheels stall, nothing moves
            self.x, self.y = nx, self.y + self.v * math.sin(self.cap) * h
            self.cap += self.w * h
            self.odl += (self.v - self.w * self.voie / 2) * h
            self.odr += (self.v + self.w * self.voie / 2) * h
        self.t += dt

    def horloge(self):
        return self.t

    def odometrie(self):
        return self.odl, self.odr

    def vu_par_le_drone(self):
        return (self.x, self.y, self.cap) if self.drone_ok else None


class FauxDrone:
    def __init__(self, rover):
        self.rover = rover

    def disponible(self):
        return self.rover.drone_ok

    def zoomer(self, tuile):
        return []


def robot(faux, carte=None):
    return RaspRoverReel(
        carte or CarteFusion(),
        faux.envoyer,
        faux.odometrie,
        faux.vu_par_le_drone,
        lambda position, cap: None,  # no camera in these tests
        FauxDrone(faux),
        horloge=faux.horloge,
        dormir=faux.dormir,
    )


def test_command_line_has_no_spaces() -> None:
    assert commande_texte({'T': 13, 'X': 0.2, 'Z': 0}) == 'base -c {"T":13,"X":0.2,"Z":0}'


def test_angle_normalise() -> None:
    assert math.isclose(angle_normalise(3 * math.pi / 2), -math.pi / 2)


def test_drives_to_the_target_with_the_drone() -> None:
    faux = FauxRover(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert r.deplacer((1.3, 1.2)) is None
    assert math.dist((faux.x, faux.y), (1.3, 1.2)) < 0.05
    assert r.position() == (faux.x, faux.y)  # the drone gives the position
    assert faux.commandes[-1][1] == {'T': 1, 'L': 0, 'R': 0}  # stops at the end
    instants = [t for t, c in faux.commandes if c['T'] == 13]
    assert max(b - a for a, b in zip(instants, instants[1:], strict=False)) < 3.0  # heartbeat


def test_drives_on_odometry_without_the_drone() -> None:
    faux = FauxRover(1.0, 1.0, math.pi / 2)
    faux.drone_ok = False
    r = robot(faux)
    r.placer((1.0, 1.0))
    assert not r.drone_disponible()
    r.deplacer((1.0, 1.3))
    r.deplacer((1.25, 1.5))
    assert math.dist(r.position(), (faux.x, faux.y)) < 0.03  # odometry follows the truth
    assert r.incertitude() > 0  # and the uncertainty grows


def test_bump_is_detected_from_the_wheels_not_moving() -> None:
    faux = FauxRover(1.0, 1.0, 0.0, mur_x=1.2)
    r = robot(faux)
    r.placer((1.0, 1.0))
    impact = r.deplacer((1.5, 1.0))
    assert impact is not None and 1.2 <= impact[0] < 1.45
    assert faux.commandes[-1][1] == {'T': 1, 'L': 0, 'R': 0}


def test_link_loss_is_reported() -> None:
    faux = FauxRover(1.0, 1.0, 0.0)
    r = robot(faux)
    r.placer((1.0, 1.0))
    faux.joignable = False
    r.deplacer((1.3, 1.0))
    assert not r.rover_joignable()


def test_the_brain_drives_the_real_adapter() -> None:
    """The same mission code as in the simulation, driving the real-rover adapter."""
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    faux = FauxRover(1.0, 1.0, math.pi / 2)
    lignes = []
    cerveau.mission(
        robot(faux, carte),
        carte,
        (1.0, 1.0),
        (3.0, 4.0),
        zone=(0, 0, 6.4, 6.4),
        journal=lignes.append,
    )
    assert 'ARRIVE pres de la victime' in '\n'.join(lignes)
    assert math.dist((faux.x, faux.y), (3.0, 4.0)) <= 0.6
