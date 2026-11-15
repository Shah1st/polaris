"""suivi_continu.py: rounding corners, deciding when to send a new path, planning ahead of the
rover, and placing the rover's own grid on the maps."""

import math

import numpy as np
import pytest
import suivi_continu as sc
from fusion_carte import CarteFusion, decouper


def toujours(a, b):
    return True


def virages(points):
    return [sc.virage(a, b, c) for a, b, c in zip(points, points[1:], points[2:], strict=False)]


def test_a_right_angle_becomes_bends_the_rover_does_not_stop_on() -> None:
    coin = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    lisse = sc.lisser(coin, toujours)
    assert lisse[0] == coin[0] and lisse[-1] == coin[-1]
    assert max(virages(lisse)) <= math.radians(45.0)
    # it stays close to the corner: never more than the cut away from the original path
    for p in lisse:
        assert min(sc.projeter(coin, p)[2], 1.0) < 0.08
    assert sc.longueur(lisse) < sc.longueur(coin)


def test_a_gentle_path_and_a_blocked_corner_are_left_alone() -> None:
    doux = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.3)]
    assert sc.lisser(doux, toujours) == doux
    coin = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert sc.lisser(coin, lambda a, b: False) == coin  # the short cut touches an obstacle
    assert sc.lisser([(0.0, 0.0)], toujours) == [(0.0, 0.0)]


def test_the_short_cut_is_refused_next_to_an_obstacle_or_in_unseen_cells() -> None:
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.15)
    libre = np.zeros((40, 40), np.int8)
    libre[18:20, 14:16] = 1  # a box inside the corner at (11, 11): centre (10.75, 10.95)
    for msg in decouper(libre, 10.0, 10.0, 4, carte, 'drone', t=1.0):
        carte.recevoir(msg)
    g = carte.grille(10.0, 10.0, 12.0, 12.0)
    corde = sc.corde_libre(carte.grille_plan(g), g)
    assert corde((10.2, 10.2), (11.5, 10.2))  # far from the box
    assert not corde((10.4, 10.95), (11.2, 10.95))  # through it
    assert not corde((10.85, 10.8), (10.9, 10.85))  # within the rover's radius of it
    assert not corde((11.9, 11.9), (12.3, 11.9))  # leaves what has been seen
    assert sc.corde_libre(carte.grille_plan(g), g, vu=False)((11.9, 10.2), (11.95, 10.2))
    # under the rover at the start nothing is checked
    assert sc.corde_libre(carte.grille_plan(g), g, depart=(10.75, 10.95), rayon=0.3)(
        (10.7, 10.95), (10.8, 10.95)
    )


def test_projection_and_pieces_of_a_path() -> None:
    chemin = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    s, q, d = sc.projeter(chemin, (0.4, 0.1))
    assert s == pytest.approx(0.4) and q == pytest.approx((0.4, 0.0)) and d == pytest.approx(0.1)
    assert sc.projeter(chemin, (1.2, 0.5))[0] == pytest.approx(1.5)
    assert np.allclose(sc.entre(chemin, 0.5, 1.5), [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5)])
    assert np.allclose(sc.entre(chemin, 1.5, 9.0), [(1.0, 0.5), (1.0, 1.0)])
    assert np.allclose(sc.reste(chemin, (0.4, 0.1)), [(0.4, 0.0), (1.0, 0.0), (1.0, 1.0)])
    morceau, depart = sc.prolonger(chemin, (0.4, 0.1), 0.2)
    assert np.allclose(morceau, [(0.4, 0.0), (0.6, 0.0)]) and np.allclose(depart, (0.6, 0.0))
    # near the end of the path the piece stops at its end
    assert np.allclose(sc.prolonger(chemin, (1.0, 0.95), 0.2)[1], (1.0, 1.0))


def test_a_new_path_is_sent_only_when_it_is_better() -> None:
    courant = [(0.0, 0.0), (1.0, 0.0)]
    pos, but = (0.3, 0.0), (3.0, 0.0)
    assert not sc.a_remplacer(courant, [(0.3, 0.0), (1.0, 0.0)], pos, but)  # the same
    assert not sc.a_remplacer(courant, [(0.3, 0.01), (1.05, 0.02)], pos, but)  # noise
    assert not sc.a_remplacer(courant, [(0.3, 0.0), (0.6, 0.2), (1.0, 0.0)], pos, but)  # other
    assert not sc.a_remplacer(courant, [(0.3, 0.0), (0.6, 0.0)], pos, but)  # ends earlier
    assert sc.a_remplacer(courant, [(0.3, 0.0), (1.4, 0.0)], pos, but)  # more floor seen
    detour = [(0.0, 0.0), (0.5, 0.6), (1.0, 0.0)]
    assert sc.a_remplacer(detour, [(0.05, 0.06), (1.0, 0.0)], (0.05, 0.06), but)  # shorter


def test_the_rover_grid_is_placed_on_the_maps() -> None:
    # the rover's odometry frame: origin at (20, 20) of the maps, x axis turned by 90 degrees
    cells = [[-1] * 4 for _ in range(4)]
    cells[1][2] = 1  # odometry cell x 0.10..0.15, y 0.05..0.10
    cells[0][0] = 0
    rep = {'res': 0.05, 'x0': 0.0, 'y0': 0.0, 'cells': cells, 'age_s': 0.2}
    m, x0, y0 = sc.grille_locale(rep, (20.0, 20.0), math.pi / 2)
    # odometry (0.125, 0.075) -> maps (20 - 0.075, 20 + 0.125)
    i, j = math.floor((20.125 - y0) / 0.05), math.floor((19.925 - x0) / 0.05)
    assert m[i, j] == 1 and (m == 1).sum() == 1 and (m == 0).sum() == 1
    assert sc.grille_locale({**rep, 'age_s': 5.0}, (20.0, 20.0), 0.0) is None  # too old
    assert sc.grille_locale({**rep, 'age_s': None}, (20.0, 20.0), 0.0) is None  # no picture yet
    assert sc.grille_locale(None, (20.0, 20.0), 0.0) is None
