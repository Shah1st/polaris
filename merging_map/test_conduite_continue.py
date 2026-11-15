"""Continuous driving end to end (sim/fusion_reelle.py --continu) against the REAL rover
server (rover_server + rover_core, simulated ESP32: banc_reel.Banc), in real time, ports
127.0.0.1:8280-8284.

The depth camera is a stand-in: ten times a second, a perfect grid of what is within 1.25 m in
front of the rover's TRUE pose goes into the rover's local map (Rover.feed_grid), as the feed
from rover_mapper/d405_server.py does on the robot. The drone has seen a free strip and no
obstacle: what is in the way is only seen by the rover."""

import math
import sys
import threading

import numpy as np
import pytest

pytest.importorskip('cv2')

import banc_reel  # noqa: E402
from banc_reel import RACINE, Box, boite_touchee  # noqa: E402

if str(RACINE / 'sim') not in sys.path:
    sys.path.append(str(RACINE / 'sim'))

import fusion_reelle as fr  # noqa: E402

PORTS_BANC = range(8280, 8285)
RES, PORTEE, DEMI_CHAMP = 0.05, 1.25, 43.0


@pytest.fixture
def banc(monkeypatch):
    monkeypatch.setattr(banc_reel, 'PORTS', PORTS_BANC)
    b = banc_reel.Banc()
    b.cfg.dist_scale = 0.98  # the simulated board's ground_ratio
    b.attendre_pret()
    yield b
    b.fermer()


class Yeux:
    """The stand-in depth camera, and a recorder of the commanded speed."""

    def __init__(self, banc, boites):
        self.banc, self.boites = banc, boites
        self.vitesses = []  # (true x, commanded v)
        self._fin = threading.Event()
        self._fil = threading.Thread(target=self._boucle, daemon=True)
        self._fil.start()

    def grille(self):
        """What a depth camera sees from the true pose: the floor free within 43 degrees of
        the axis, the boxes, and nothing behind a box."""
        b = self.banc.board
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        n = round(PORTEE / RES)
        xs = (np.arange(n) + 0.5) * RES
        ys = (np.arange(2 * n) + 0.5) * RES - PORTEE
        X, Y = np.meshgrid(xs, ys)  # noqa: N806
        gx, gy = b.x + c * X - s * Y, b.y + s * X + c * Y  # the cells, in the board's frame
        obstacle = np.zeros(X.shape, bool)
        for bo in self.boites:
            obstacle |= (
                (gx >= bo.x) & (gx <= bo.x + bo.width) & (gy >= bo.y) & (gy <= bo.y + bo.height)
            )
        cap, dist = np.degrees(np.arctan2(Y, X)), np.hypot(X, Y)
        vu = (np.abs(cap) <= DEMI_CHAMP) & (dist <= PORTEE)
        secteur = np.clip(((cap + 90) / 2).astype(int), 0, 89)
        proche = np.full(90, np.inf)
        for k, d in zip(secteur[obstacle & vu], dist[obstacle & vu], strict=True):
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < 90:
                    proche[kk] = min(proche[kk], d)
        cells = np.full(X.shape, -1, np.int8)
        cells[vu & (dist <= proche[secteur])] = 0
        cells[vu & obstacle & (dist <= proche[secteur] + 2 * RES)] = 1
        return {'res': RES, 'x0': 0.0, 'y0': -PORTEE, 'cells': cells.tolist(), 'age_s': 0.0}

    def _boucle(self):
        while not self._fin.wait(0.1):
            self.banc.rover.feed_grid(self.grille())
            self.vitesses.append((self.banc.board.x, self.banc.rover.ctrl.v_cmd))

    def fermer(self):
        self._fin.set()
        self._fil.join(timeout=2.0)


def mission(banc, boites, but, extra=()):
    """A FusionReelle in continuous mode on the bench; the drone has seen a free strip 3 m
    long and 1.2 m wide along the rover's x axis. Returns (fusion, eyes)."""
    banc.cfg.local_guard = 1
    f = fr.FusionReelle(
        fr.options(
            ['--rover', banc.url, '--continu', '--carte-locale', '--source-locale', 'rover',
             '--vitesse', '0.2', '--periode-plan', '0.25', '--delai-http', '2', *extra]
        ),
        percevoir=lambda *a: None,
        percevoir_frais=lambda *a: None,
    )  # fmt: skip
    bande = np.zeros((24, 60), np.int8)
    for msg in fr.decouper(bande, 19.5, 19.4, 4, f.carte, 'drone', t=1.0):
        f.carte.recevoir(msg)
    f.objectif = (20.0 + but[0], 20.0 + but[1])
    return f, Yeux(banc, boites)


def test_a_clear_way_is_driven_without_a_stop(banc) -> None:
    """Paths of 0.8 m at most: the rover is sent several, each taking over from the last
    while it moves. Its speed never drops between the start and the approach of the goal."""
    f, yeux = mission(banc, [], (1.8, 0.0), ['--segment', '0.8'])
    try:
        issue = f._mission_continue()
    finally:
        yeux.fermer()
    assert issue == 'arrive', f.mission
    assert math.dist((banc.board.x, banc.board.y), (1.8, 0.0)) <= f.args.tolerance + 0.1
    chemins = [r for r in banc.recus if r[1] == '/path']
    assert len(chemins) >= 3 and all(r[2].get('id') for r in chemins)
    en_route = [v for x, v in yeux.vitesses if 0.3 < x < 1.2]
    assert en_route and min(en_route) > 0.12, min(en_route)
    statuts = [h['status'] for h in banc.rover.ctrl.history if h['kind'] == 'path']
    assert statuts[0] == 'done' and set(statuts[1:]) <= {'replaced by a new command'}


def test_an_obstacle_only_the_rover_sees_is_driven_round_without_contact(banc) -> None:
    boite = Box(0.9, -0.1, 0.15, 0.2, color=(0, 0, 0))  # across the straight line
    f, yeux = mission(banc, [boite], (1.9, 0.0))
    depart = len(banc.poses)
    try:
        issue = f._mission_continue()
    finally:
        yeux.fermer()
    assert issue == 'arrive', f.mission
    assert math.dist((banc.board.x, banc.board.y), (1.9, 0.0)) <= f.args.tolerance + 0.1
    # true poses, in the frame of the simulated board (the bench's map frame is turned)
    trace = banc.board.trail
    assert not any(boite_touchee((x, y, 0.0), boite, 0.2, 0.2) for x, y in trace)
    assert max(abs(y) for _, y in trace) > 0.2  # it went round, not through
    assert len(banc.poses) > depart
    # the box is on the fused map, put there by the rover's grid
    g = f.carte.grille(20.85, 19.85, 21.1, 20.15)
    assert (g.val == 1).sum() >= 6 and (g.src[g.val == 1] == 2).all()


def test_a_wall_across_the_strip_stops_the_rover_and_ends_blocked(banc) -> None:
    mur = Box(1.0, -0.7, 0.1, 1.4, color=(0, 0, 0))  # the whole width of the mission's zone
    f, yeux = mission(banc, [mur], (1.9, 0.0), ['--marge-zone', '0.5', '--duree-max', '90'])
    try:
        issue = f._mission_continue()
    finally:
        yeux.fermer()
    assert issue == 'bloque', f.mission
    assert banc.board.x < 1.0 - 0.1 - 0.05  # nose (10 cm ahead of the centre) short of it
    assert abs(banc.rover.ctrl.v_cmd) < 1e-6
