"""Tests for fusion_carte: tile fusion, planning and blocking decisions."""

import numpy as np
import pytest
from fusion_carte import (
    INCONNU,
    LIBRE,
    OBSTACLE,
    SRC_DRONE,
    SRC_ROVER,
    CarteFusion,
    decouper,
    dilater,
    reduire,
)


# ----------------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------------
def test_reduire_is_conservative() -> None:
    tuile = np.array(
        [
            [0, 0, 0, -1],
            [0, 1, -1, -1],
            [0, 0, -1, -1],
            [0, 0, -1, -1],
        ],
        dtype=np.int8,
    )
    grossier = reduire(tuile, 2)
    # one obstacle blocks the block; free + unknown -> free; all unknown -> unknown
    assert grossier.tolist() == [[OBSTACLE, LIBRE], [LIBRE, INCONNU]]


def test_reduire_pads_with_unknown() -> None:
    grossier = reduire(np.zeros((3, 3), np.int8), 2)
    assert grossier.shape == (2, 2)
    assert (grossier == LIBRE).all()


def test_dilater_grows_by_radius() -> None:
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    d = dilater(m, 2)
    assert d[3, 1] and d[3, 5] and d[1, 3] and d[5, 3]
    assert not d[1, 1]  # corner is outside the disc
    assert d.sum() == 13


# ----------------------------------------------------------------------------
# Reception and fusion
# ----------------------------------------------------------------------------
def petite_carte() -> CarteFusion:
    # 8x8 tiles; cells: z2 = 10 cm, z1 = 20 cm, z0 = 40 cm
    return CarteFusion(n=8, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1, rayon_robot=0.0)


def test_rover_wins_over_drone_with_negative_coordinates() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'drone', 't': 0, 'z': 0, 'x': -1, 'y': -1, 'data': np.ones((8, 8))})
    rover = np.full((8, 8), INCONNU)
    rover[7, 7] = LIBRE
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': -1, 'y': -1, 'data': rover})

    g = carte.grille(-0.4, -0.4, 0.0, 0.0)
    assert g.val.shape == (4, 4)
    assert g.val[-1, -1] == LIBRE and g.src[-1, -1] == SRC_ROVER
    assert g.val[0, 0] == OBSTACLE and g.src[0, 0] == SRC_DRONE


def test_drone_tiles_are_kept_when_rover_overrides() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'drone', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': np.ones((8, 8))})
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))})
    assert (carte.grille(0, 0, 0.8, 0.8).val == LIBRE).all()
    assert (carte.tuiles['drone'][(2, 0, 0)] == OBSTACLE).all()


def test_unknown_never_overrides() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'rover', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': np.ones((8, 8))})
    carte.recevoir(
        {'source': 'rover', 't': 1, 'z': 2, 'x': 0, 'y': 0, 'data': np.full((8, 8), INCONNU)}
    )
    assert (carte.grille(0, 0, 0.8, 0.8).val == OBSTACLE).all()


def test_older_message_only_fills_gaps() -> None:
    carte = petite_carte()
    recent = np.full((8, 8), INCONNU)
    recent[0, 0] = LIBRE
    carte.recevoir({'source': 'rover', 't': 5, 'z': 2, 'x': 0, 'y': 0, 'data': recent})
    ancien = np.ones((8, 8))
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': 0, 'y': 0, 'data': ancien})
    g = carte.grille(0, 0, 0.8, 0.8)
    assert g.val[0, 0] == LIBRE  # the recent value is kept
    assert g.val[3, 3] == OBSTACLE  # the old message fills the unknown cells


def test_finer_level_wins_within_a_source() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'drone', 't': 0, 'z': 0, 'x': 0, 'y': 0, 'data': np.ones((8, 8))})
    carte.recevoir({'source': 'drone', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))})
    g = carte.grille(0, 0, 1.6, 1.6)
    assert (g.val[:8, :8] == LIBRE).all()  # area covered by the fine tile
    assert (g.val[8:, 8:] == OBSTACLE).all()  # rest of the coarse tile


@pytest.mark.parametrize(
    'msg, erreur',
    [
        ({'source': 'x', 'z': 0, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))}, 'source'),
        ({'source': 'rover', 'z': 3, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))}, 'niveau'),
        ({'source': 'rover', 'z': 0, 'x': 0, 'y': 0, 'data': np.zeros((4, 4))}, 'tuile'),
        ({'source': 'rover', 'z': 0, 'x': 0, 'y': 0, 'data': np.full((8, 8), 2)}, 'valeurs'),
    ],
)
def test_recevoir_rejects_invalid_messages(msg: dict, erreur: str) -> None:
    with pytest.raises(ValueError, match=erreur):
        petite_carte().recevoir(msg)


def test_decouper_round_trip() -> None:
    rng = np.random.default_rng(0)
    grande = rng.choice([INCONNU, LIBRE, OBSTACLE], size=(20, 13)).astype(np.int8)
    carte = petite_carte()
    for msg in decouper(grande, -0.7, -1.2, 2, carte, 'rover'):
        carte.recevoir(msg)
    g = carte.grille(-0.7, -1.2, -0.7 + 1.3, -1.2 + 2.0)
    assert (g.val == grande).all()


def test_decouper_skips_fully_unknown_tiles() -> None:
    carte = petite_carte()
    m = np.full((16, 16), INCONNU)
    m[0, 0] = LIBRE
    assert len(decouper(m, 0, 0, 2, carte, 'rover')) == 1


def test_signaler_obstacle_marks_rover_cells() -> None:
    carte = petite_carte()
    carte.signaler_obstacle(0.35, 0.35, rayon=0.1)
    g = carte.grille(0, 0, 0.8, 0.8)
    assert g.val[3, 3] == OBSTACLE and g.src[3, 3] == SRC_ROVER


# ----------------------------------------------------------------------------
# Planning and decisions
#   World of 6.4 m: a wall at y = 3.2..3.6 m across the whole width.
#   The drone first sees it with 40 cm cells (z0), then can zoom with 20 cm cells (z1).
# ----------------------------------------------------------------------------
DEPART, VICTIME = (1.0, 1.0), (3.2, 5.5)


def carte_mur() -> CarteFusion:
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1, rayon_robot=0.1)
    vue_z0 = np.zeros((16, 16), np.int8)
    vue_z0[8, :] = OBSTACLE  # y = 3.2..3.6
    for msg in decouper(vue_z0, 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    return carte


def vue_zoom(avec_passage: bool) -> np.ndarray:
    vue_z1 = np.zeros((32, 32), np.int8)
    vue_z1[16:18, :] = OBSTACLE  # same wall, 20 cm cells
    if avec_passage:
        vue_z1[16:18, 10:13] = LIBRE  # 60 cm gap at x = 2.0..2.6
    return vue_z1


def jouer(carte: CarteFusion, vue_z1: np.ndarray, max_tours: int = 6) -> tuple[dict, list]:
    """Answers the drone zoom requests until the module takes another decision."""
    demandees = []
    for _ in range(max_tours):
        dec = carte.decider(DEPART, VICTIME, zone=(0, 0, 6.4, 6.4))
        if dec['action'] != 'DEMANDER_DRONE':
            return dec, demandees
        for z, tx, ty in dec['tuiles']:
            demandees.append((z, tx, ty))
            data = vue_z1[ty * 16 : (ty + 1) * 16, tx * 16 : (tx + 1) * 16]
            carte.recevoir({'source': 'drone', 't': 1, 'z': z, 'x': tx, 'y': ty, 'data': data})
    raise AssertionError('le module demande des zooms sans fin')


def test_open_field_gives_a_path() -> None:
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    dec = carte.decider(DEPART, VICTIME, zone=(0, 0, 6.4, 6.4))
    assert dec['action'] == 'SUIVRE'
    x, y = dec['chemin'][-1]
    assert np.hypot(x - VICTIME[0], y - VICTIME[1]) <= 0.5


def test_blocked_wall_asks_drone_to_zoom() -> None:
    dec = carte_mur().decider(DEPART, VICTIME, zone=(0, 0, 6.4, 6.4))
    assert dec['action'] == 'DEMANDER_DRONE'
    assert dec['tuiles'] and all(z == 1 for z, _, ty in dec['tuiles'])
    assert all(ty == 1 for _, _, ty in dec['tuiles'])  # tiles that contain the wall


def test_zoom_reveals_the_gap() -> None:
    dec, demandees = jouer(carte_mur(), vue_zoom(avec_passage=True))
    assert dec['action'] == 'SUIVRE'
    assert demandees
    # the path crosses the wall through the gap
    traversee = [x for x, y in dec['chemin'] if 3.2 <= y < 3.6]
    assert traversee and all(2.0 <= x <= 2.6 for x in traversee)


def test_closed_wall_falls_back_to_closest_point() -> None:
    carte = carte_mur()
    dec, demandees = jouer(carte, vue_zoom(avec_passage=False))
    assert dec['action'] == 'OBJECTIF_DEGRADE'
    assert len(demandees) == len(set(demandees))  # never the same tile twice
    assert dec['point'][1] < 3.2  # stays below the wall
    assert dec['distance_restante'] > 0.5


def test_rover_obstacle_invalidates_path() -> None:
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    zone = (0, 0, 6.4, 6.4)
    chemin = carte.decider(DEPART, VICTIME, zone=zone)['chemin']
    assert carte.chemin_valide(chemin, zone)
    x, y = chemin[len(chemin) // 2]
    carte.signaler_obstacle(x, y, rayon=0.3)
    assert not carte.chemin_valide(chemin, zone)
    nouveau = carte.decider(DEPART, VICTIME, zone=zone)
    assert nouveau['action'] == 'SUIVRE' and nouveau['chemin'] != chemin
