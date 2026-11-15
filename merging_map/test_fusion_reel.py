"""Tests of the map core for the real mission (audit mm-core F1, F2, F3, F5; interface 7):
leaving the start without cutting the robot radius, forgetting rover observations, an
unreachable victim at field scale, a victim seen as an obstacle, several devices per role."""

import math
import time

import fusion_carte
import numpy as np
import pytest
from fusion_carte import OBSTACLE, SRC_DRONE, SRC_ROVER, CarteFusion, decouper

ZONE = (0.0, 0.0, 4.8, 4.8)


def clearance(x, y):
    """Distance from (x, y) to the 0.5 m box at x, y in [2.0, 2.5]."""
    return math.hypot(max(2.0 - x, x - 2.5, 0.0), max(2.0 - y, y - 2.5, 0.0))


def map_with_a_box(source='rover', **reglages):
    carte = CarteFusion(**reglages)
    monde = np.zeros((96, 96), np.int8)
    monde[40:50, 40:50] = OBSTACLE  # x, y in [2.0, 2.5]
    for msg in decouper(monde, 0.0, 0.0, 4, carte, source, t=1.0):
        carte.recevoir(msg)
    return carte


# ----------------------------------------------------------------------------
# F1: leaving the start
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'reglages, marge_drone',
    [
        ({}, 0.0),  # field
        ({}, 0.5),  # field, drone lost
        ({'niveau_plan': 4, 'rayon_robot': 0.155}, 0.0),  # arena
        ({'niveau_plan': 4, 'rayon_robot': 0.155}, 0.15),  # arena, drone lost
    ],
)
def test_a_clear_start_never_gives_a_path_into_the_robot_radius(reglages, marge_drone) -> None:
    """Starts on a 5 cm lattice next to a box the rover saw, at least the robot radius away:
    every vertex of the path keeps the robot radius (audit: 0.05 m with the old release)."""
    pires = []
    for sx in np.arange(1.5, 2.0, 0.05):
        for sy in np.arange(1.5, 2.05, 0.05):
            carte = map_with_a_box(**reglages)
            carte.marge_drone = marge_drone
            depart = (round(float(sx), 3), round(float(sy), 3))
            if clearance(*depart) < carte.rayon_effectif + 0.01:
                continue
            dec = carte.decider(depart, (3.0, 3.2), zone=ZONE, tolerance=0.15)
            assert dec['action'] == 'SUIVRE'
            pires.append(min(clearance(x, y) for x, y in dec['chemin'][1:]))
            # the path just planned also passes the check made after every step
            assert carte.chemin_valide(dec['chemin'], zone=ZONE)
    assert pires and min(pires) >= carte.rayon_effectif - 1e-9


def test_a_start_inside_the_inflation_can_leave_without_getting_closer() -> None:
    """After a bump the rover stands 5 cm from the obstacle, deep inside the inflation: it can
    still leave, and the first part of the path never gets closer to the box."""
    carte = map_with_a_box(niveau_plan=4, rayon_robot=0.155)
    depart = (1.925, 2.225)  # 7.5 cm from the box face
    dec = carte.decider(depart, (1.0, 2.25), zone=ZONE, tolerance=0.15)
    assert dec['action'] == 'SUIVRE'
    distances = [clearance(x, y) for x, y in dec['chemin']]
    sortie = [d for d in distances if d < carte.rayon_effectif]
    assert sortie == sorted(sortie)  # monotone exit
    assert carte.chemin_valide(dec['chemin'], zone=ZONE)
    # going along the box, closer than the start, is refused
    assert not carte.chemin_valide([depart, (1.975, 2.225)], zone=ZONE)


def test_a_free_start_keeps_the_inflation_in_the_path_check() -> None:
    """The old check ignored the first ~0.5 m of the path without the drone."""
    carte = map_with_a_box(niveau_plan=4, rayon_robot=0.155)
    carte.marge_drone = 0.15
    depart = (1.6, 2.25)  # 0.4 m from the box, outside the inflation
    assert carte.premier_point_bloque([depart, (1.9, 2.25)], zone=ZONE) is not None


# ----------------------------------------------------------------------------
# F2: forgetting rover observations
# ----------------------------------------------------------------------------
def gap_map():
    """Arena wall at x = 1.2 m with a 45 cm gap seen by the overhead camera."""
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    monde = np.zeros((48, 48), np.int8)
    monde[:, 24:26] = OBSTACLE
    monde[21:30, 24:26] = 0  # gap y in [1.05, 1.5]
    for msg in decouper(monde, 0.0, 0.0, 4, carte, 'drone', t=0.0):
        carte.recevoir(msg)
    return carte


def decide(carte):
    return carte.decider((0.4, 1.25), (2.0, 1.25), zone=(0, 0, 2.4, 2.4), tolerance=0.15)


def test_forgetting_a_false_bump_reopens_the_passage() -> None:
    carte = gap_map()
    assert decide(carte)['action'] == 'SUIVRE'
    carte.signaler_obstacle(1.22, 1.25, 0.1, t=5.0)  # a false 'stalled' in the gap
    assert decide(carte)['action'] == 'OBJECTIF_DEGRADE'
    assert carte.oublier_rover(zone=(0.0, 0.0, 1.0, 2.4)) == 0  # elsewhere: nothing
    assert decide(carte)['action'] == 'OBJECTIF_DEGRADE'
    assert carte.oublier_rover(zone=(1.0, 1.0, 1.5, 1.5)) > 0
    assert decide(carte)['action'] == 'SUIVRE'


def test_forgetting_everything_keeps_the_drone_map() -> None:
    carte = gap_map()
    carte.signaler_obstacle(1.22, 1.25, 0.1, t=5.0)
    carte.signaler_obstacle(0.5, 0.5, 0.1, t=5.0, source='rover:b')
    assert carte.oublier_rover() > 0
    assert not any(carte.tuiles[s] for s in carte.sources('rover'))
    g = carte.grille(0, 0, 2.4, 2.4)
    assert (g.src != SRC_ROVER).all() and (g.val == OBSTACLE).any()


def test_a_bump_with_a_lifetime_expires() -> None:
    carte = gap_map()
    carte.duree_choc = 20.0
    carte.signaler_obstacle(1.22, 1.25, 0.1, t=5.0)
    assert carte.expirer_chocs(10.0) == 0
    assert decide(carte)['action'] == 'OBJECTIF_DEGRADE'
    assert carte.expirer_chocs(25.0) == 1
    assert decide(carte)['action'] == 'SUIVRE'


def test_bumps_last_forever_by_default() -> None:
    carte = gap_map()
    carte.signaler_obstacle(1.22, 1.25, 0.1, t=5.0)
    assert carte.expirer_chocs(1e9) == 0
    assert decide(carte)['action'] == 'OBJECTIF_DEGRADE'


# ----------------------------------------------------------------------------
# F3: unreachable victim at field scale
# ----------------------------------------------------------------------------
@pytest.mark.skipif(fusion_carte.cv2 is None, reason='the fast path needs OpenCV')
def test_an_enclosed_victim_is_decided_fast_at_field_scale() -> None:
    """48 x 48 m surveyed at z4 (0.66 M fine cells), victim inside a closed ring: the old
    decider explored everything in Python (~4 s on 64 x 57.6 m)."""
    carte = CarteFusion()
    monde = np.zeros((960, 960), np.int8)
    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, 960, size=(300, 2)):
        monde[max(i - 4, 0) : i + 4, max(j - 4, 0) : j + 4] = OBSTACLE  # trunks
    yy, xx = np.mgrid[0:960, 0:960]
    d = np.hypot(yy - 800, xx - 800)
    monde[(d > 20) & (d < 26)] = OBSTACLE  # ring of 1 m radius around (40, 40)
    monde[(d <= 20)] = 0
    for msg in decouper(monde, 0.0, 0.0, 4, carte, 'drone', t=0.0):
        carte.recevoir(msg)
    depart, victime = (3.0, 3.0), (40.0, 40.0)
    monde_depart = monde[55:65, 55:65]
    assert (monde_depart == 0).all()
    debut = time.perf_counter()
    dec = carte.decider(depart, victime, drone_disponible=True)
    duree = time.perf_counter() - debut
    assert dec['action'] == 'OBJECTIF_DEGRADE'
    assert 1.0 < dec['distance_restante'] < 2.0
    assert duree < 2.0, duree  # ~0.3 s measured; generous for a loaded machine


def test_degraded_goal_is_the_nearest_reachable_cell() -> None:
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    monde = np.zeros((48, 48), np.int8)
    monde[:, 24:26] = OBSTACLE  # closed wall at x = 1.2 m
    for msg in decouper(monde, 0.0, 0.0, 4, carte, 'drone', t=0.0):
        carte.recevoir(msg)
    dec = carte.decider((0.4, 1.0), (2.0, 1.025), zone=(0, 0, 2.4, 2.4), tolerance=0.15)
    assert dec['action'] == 'OBJECTIF_DEGRADE'
    # last free column before the inflated wall: 1.2 - 0.155 -> cell centre 1.025
    assert dec['point'] == pytest.approx((1.025, 1.025))
    assert dec['distance_restante'] == pytest.approx(0.975)
    assert dec['chemin'][-1] == pytest.approx(dec['point'])


# ----------------------------------------------------------------------------
# F5: victim seen as an obstacle (arena tolerance 0.15 < robot radius 0.155)
# ----------------------------------------------------------------------------
def test_a_victim_seen_as_an_obstacle_is_still_reached() -> None:
    carte = CarteFusion(niveau_plan=4, rayon_robot=0.155)
    monde = np.zeros((48, 48), np.int8)
    monde[38:40, 38:40] = OBSTACLE  # 10 x 10 cm object at (1.95, 1.95)
    for msg in decouper(monde, 0.0, 0.0, 4, carte, 'drone', t=0.0):
        carte.recevoir(msg)
    victime = (1.95, 1.95)
    dec = carte.decider((0.4, 0.4), victime, zone=(0, 0, 2.4, 2.4), tolerance=0.15)
    assert dec['action'] == 'SUIVRE'
    fin = dec['chemin'][-1]
    assert carte.rayon_effectif <= math.dist(fin, victime) <= 0.15 + 0.155 + 0.05 + 1e-9
    # a free victim keeps the asked tolerance
    dec = carte.decider((0.4, 0.4), (1.0, 1.0), zone=(0, 0, 2.4, 2.4), tolerance=0.15)
    assert math.dist(dec['chemin'][-1], (1.0, 1.0)) <= 0.15


# ----------------------------------------------------------------------------
# Interface 7: several devices per role
# ----------------------------------------------------------------------------
def tile(source, t, z, x, y, valeur, cases):
    data = np.full((32, 32), -1, np.int8)
    for i, j in cases:
        data[i, j] = valeur
    return {'source': source, 't': t, 'z': z, 'x': x, 'y': y, 'data': data}


def test_sources_with_a_device_id_are_accepted() -> None:
    carte = CarteFusion()
    for source in ('drone', 'rover', 'drone:a', 'rover:2', 'drone:overhead-1'):
        carte.recevoir(tile(source, 0.0, 4, 0, 0, 0, [(0, 0)]))
    for source in ('plane', 'drone:', ':a', 'rovers:1', None):
        with pytest.raises(ValueError):
            carte.recevoir(tile(source, 0.0, 4, 0, 0, 0, [(0, 0)]))
    assert set(carte.sources('drone')) == {'drone', 'drone:a', 'drone:overhead-1'}
    assert set(carte.sources('rover')) == {'rover', 'rover:2'}


def test_two_drones_and_two_rovers() -> None:
    carte = CarteFusion()
    # drone a sees cell (0, 0) as an obstacle at t=1; drone b sees it free at t=2
    carte.recevoir(tile('drone:a', 1.0, 4, 0, 0, OBSTACLE, [(0, 0), (1, 1)]))
    carte.recevoir(tile('drone:b', 2.0, 4, 0, 0, 0, [(0, 0)]))
    # drone b is older on cell (2, 2): drone a's later tile wins there
    carte.recevoir(tile('drone:b', 0.5, 4, 1, 0, OBSTACLE, [(2, 2)]))
    carte.recevoir(tile('drone:a', 3.0, 4, 1, 0, 0, [(2, 2)]))
    # a coarser but more recent drone tile does not override a finer one (level rule)
    carte.recevoir(tile('drone:c', 9.0, 3, 0, 0, 0, [(0, 0)]))
    # each device's tiles are kept apart
    assert carte.tuiles['drone:a'][(4, 0, 0)][0, 0] == OBSTACLE
    assert carte.tuiles['drone:b'][(4, 0, 0)][0, 0] == 0
    g = carte.grille(0.0, 0.0, 3.2, 1.6)
    assert g.val[0, 0] == 0 and g.src[0, 0] == SRC_DRONE  # most recent drone wins
    assert g.val[1, 1] == OBSTACLE  # only drone a knows it
    assert g.val[2, 32 + 2] == 0  # most recent wins on the second tile too
    assert g.val[1, 1] == OBSTACLE and g.zdrone[1, 1] == 4
    # rovers: rover role beats every drone; the most recent rover wins between rovers
    carte.recevoir(tile('rover:1', 5.0, 4, 0, 0, 0, [(1, 1)]))
    carte.recevoir(tile('rover:2', 4.0, 4, 0, 0, OBSTACLE, [(1, 1), (3, 3)]))
    g = carte.grille(0.0, 0.0, 3.2, 1.6)
    assert g.val[1, 1] == 0 and g.src[1, 1] == SRC_ROVER
    assert g.val[3, 3] == OBSTACLE and g.src[3, 3] == SRC_ROVER
    # the surveyed area is the union of every drone's tiles
    carte.recevoir(tile('drone:b', 0.0, 4, 3, 2, 0, [(0, 0)]))
    assert carte.zone_survolee() == pytest.approx((0.0, 0.0, 6.4, 4.8))
    # forgetting one rover keeps the other
    carte.oublier_rover(source='rover:2')
    g = carte.grille(0.0, 0.0, 3.2, 1.6)
    assert g.val[1, 1] == 0 and g.val[3, 3] == -1


def test_plain_names_behave_as_before() -> None:
    """'drone' and 'rover' alone: same fused map as two devices named that way."""
    a, b = CarteFusion(), CarteFusion()
    for carte, (d, r) in ((a, ('drone', 'rover')), (b, ('drone:x', 'rover:y'))):
        carte.recevoir(tile(d, 1.0, 2, 0, 0, OBSTACLE, [(0, 0)]))
        carte.recevoir(tile(d, 0.5, 4, 0, 0, 0, [(0, 0), (5, 5)]))
        carte.recevoir(tile(r, 2.0, 4, 0, 0, OBSTACLE, [(5, 5)]))
    ga, gb = a.grille(0, 0, 6.4, 6.4), b.grille(0, 0, 6.4, 6.4)
    assert (ga.val == gb.val).all() and (ga.src == gb.src).all() and (ga.zdrone == gb.zdrone).all()
    assert ga.val[0, 0] == 0  # finer level wins although older
