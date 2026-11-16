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
    gonfler,
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


def ecart_au_carre(res: float, di: int, dj: int) -> float:
    """Distance (m) du centre d'une case, decalee de (di, dj) cases, au carre de la case 0."""
    return res * float(np.hypot(max(abs(di) - 0.5, 0.0), max(abs(dj) - 0.5, 0.0)))


@pytest.mark.parametrize('res', [0.05, 0.1, 0.2, 0.4])
@pytest.mark.parametrize('rayon', [0.12, 0.15, 0.25, 0.5])
def test_gonfler_keeps_the_robot_radius_clear(res: float, rayon: float) -> None:
    """Le robot est un point : une case libre apres gonflage est TOUJOURS a au moins `rayon`
    de l'obstacle, et une case bloquee l'est toujours a raison (ni trop, ni trop peu)."""
    n = 41
    c = n // 2
    m = np.zeros((n, n), bool)
    m[c, c] = True
    bloque = gonfler(m, rayon, res)
    for i in range(n):
        for j in range(n):
            ecart = ecart_au_carre(res, i - c, j - c)
            if bloque[i, j]:
                assert ecart <= rayon + 1e-9, f'case ({i},{j}) bloquee a tort'
            else:
                assert ecart > rayon, f'case ({i},{j}) trop pres : {ecart:.3f} m < {rayon} m'


def test_gonfler_beats_dilater_on_coarse_cells() -> None:
    """Regression : dilater(ceil(r / res)) laissait passer le rover a 14 cm d'un coin avec
    des cases de 20 cm et un rayon de 15 cm (diagonale)."""
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert not dilater(m, 1)[5, 5]  # ancien comportement : diagonale libre, a 14,1 cm
    assert gonfler(m, 0.15, 0.2)[5, 5]  # maintenant : bloquee


def test_gonfler_does_not_close_passages_for_nothing() -> None:
    """Avec des cases de 40 cm, ceil(0.15 / 0.4) = 1 case gonflait de 28 cm : le passage entre
    deux obstacles distants de 3 cases (1,2 m) disparaissait pour un rover de 15 cm."""
    m = np.zeros((1, 9), bool)
    m[0, 1] = m[0, 7] = True
    libre = ~gonfler(m, 0.15, 0.4)
    assert libre[0, 3:6].all()  # le couloir du milieu reste ouvert, au centre


def test_gonfler_zero_radius_is_identity() -> None:
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    assert (gonfler(m, 0.0, 0.1) == m).all()


def test_astar_keeps_clearance_from_inflated_obstacles() -> None:
    """Chemin d'un robot-point : toutes ses cases sont a au moins rayon_effectif de l'obstacle."""
    carte = CarteFusion(
        n=16,
        res_fine=0.1,
        zmax=2,
        niveau_plan=2,
        niveau_zoom=1,
        rayon_robot=0.1,
        marge_securite=0.1,
    )
    sol = np.zeros((16, 16), np.int8)
    sol[4:12, 7] = OBSTACLE  # mur vertical
    carte.recevoir({'source': 'rover', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': sol})
    dec = carte.decider((0.25, 0.8), (1.35, 0.8), zone=(0, 0, 1.6, 1.6), tolerance=0.1)
    assert dec['action'] == 'SUIVRE'
    # le mur occupe x dans [0.7, 0.8] et y dans [0.4, 1.2] : distance d'un point a ce rectangle
    for x, y in dec['chemin']:
        pres = float(np.hypot(max(0.7 - x, x - 0.8, 0.0), max(0.4 - y, y - 1.2, 0.0)))
        assert pres >= carte.rayon_effectif - 1e-9, f'({x:.2f},{y:.2f}) a {pres:.3f} m du mur'


def test_marge_securite_widens_blocked_area() -> None:
    sol = np.zeros((16, 16), np.int8)
    sol[8, 8] = OBSTACLE
    bloques = []
    for marge in (0.0, 0.2):
        carte = CarteFusion(
            n=16,
            res_fine=0.1,
            zmax=2,
            niveau_plan=2,
            niveau_zoom=1,
            rayon_robot=0.1,
            marge_securite=marge,
        )
        carte.recevoir({'source': 'rover', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': sol})
        plan = carte.grille_plan(carte.grille(0, 0, 1.6, 1.6))
        bloques.append(int(plan.bloque.sum()))
    assert bloques[1] > bloques[0]


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


def test_default_zone_is_the_surveyed_area() -> None:
    carte = carte_mur()
    assert carte.zone_survolee() == (0.0, 0.0, 6.4, 6.4)
    for z, tx, ty in [(1, 0, 1), (1, 1, 1)]:
        data = vue_zoom(avec_passage=True)[ty * 16 : (ty + 1) * 16, tx * 16 : (tx + 1) * 16]
        carte.recevoir({'source': 'drone', 't': 1, 'z': z, 'x': tx, 'y': ty, 'data': data})
    dec = carte.decider(DEPART, VICTIME)  # no zone given
    assert dec['action'] == 'SUIVRE'
    # the path stays on the surveyed map and goes through the gap, not around the map edge
    assert all(0.0 <= x <= 6.4 and 0.0 <= y <= 6.4 for x, y in dec['chemin'])
    traversee = [x for x, y in dec['chemin'] if 3.2 <= y < 3.6]
    assert traversee and all(2.0 <= x <= 2.6 for x in traversee)


def test_default_zone_without_drone_tiles_uses_a_margin() -> None:
    zone = petite_carte().zone_par_defaut((1.0, 1.0), (3.0, 2.0), marge=2.0)
    assert zone == (-1.0, -1.0, 5.0, 4.0)


def test_window_check_matches_a_full_map_check() -> None:
    """premier_point_bloque only renders a window around the path: it must find the same
    blocked point as a check on the whole map, obstacles just off the path included."""
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1, rayon_robot=0.1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    chemin = [(1.0 + 0.1 * k, 1.0 + 0.05 * k) for k in range(40)]
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(1.0, 5.0, size=(12, 2)):
        carte.signaler_obstacle(float(x), float(y), rayon=0.1)
    zone = (0.0, 0.0, 6.4, 6.4)
    plan = carte.grille_plan(carte.grille(*zone))
    attendu = None
    for x, y in chemin[1:]:
        i, j = plan.ij(x, y)
        if plan.dedans(i, j) and plan.bloque[i, j]:
            attendu = (x, y)
            break
    assert carte.premier_point_bloque(chemin, zone) == attendu
    assert carte.chemin_valide(chemin, zone) == (attendu is None)


# ----------------------------------------------------------------------------
# Without the drone
# ----------------------------------------------------------------------------
def test_without_drone_the_rover_inspects_instead_of_asking_the_drone() -> None:
    carte = carte_mur()
    dec = carte.decider(DEPART, VICTIME, zone=(0, 0, 6.4, 6.4), drone_disponible=False)
    assert dec['action'] == 'INSPECTER'
    assert carte.demandes == set()  # nothing asked to the drone
    cx, cy = dec['cible']
    vx, vy = dec['point_de_vue']
    assert 3.2 <= cy < 3.6  # the target is on the wall
    distance_min = carte.rayon_robot + 0.35  # outside the safety margin, with a reserve
    assert vy < 3.2 and distance_min - 1e-9 <= np.hypot(cx - vx, cy - vy) <= 1.2
    assert np.hypot(dec['chemin'][-1][0] - vx, dec['chemin'][-1][1] - vy) < 1e-9


def test_inspected_sectors_move_the_search_elsewhere() -> None:
    carte = carte_mur()
    zone = (0, 0, 6.4, 6.4)
    premier = carte.decider(DEPART, VICTIME, zone=zone, drone_disponible=False)
    carte.inspectees.add(premier['secteur'])
    second = carte.decider(DEPART, VICTIME, zone=zone, drone_disponible=False)
    assert second['action'] == 'INSPECTER' and second['secteur'] != premier['secteur']


def test_drone_margin_only_grows_drone_obstacles_far_from_rover_views() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'drone', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))})
    d = np.zeros((8, 8))
    d[4, 4] = OBSTACLE  # drone obstacle at (0.45, 0.45)
    carte.recevoir({'source': 'drone', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': d})
    carte.rayon_robot = 0.1
    normal = carte.grille_plan(carte.grille(0, 0, 0.8, 0.8)).bloque.sum()
    carte.marge_drone = 0.2
    elargi = carte.grille_plan(carte.grille(0, 0, 0.8, 0.8)).bloque.sum()
    assert elargi > normal
    # once the rover has looked around there itself, its own view is trusted again
    r = np.full((8, 8), INCONNU)
    r[2:7, 2:7] = LIBRE
    r[4, 4] = INCONNU
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': 0, 'y': 0, 'data': r})
    assert carte.grille_plan(carte.grille(0, 0, 0.8, 0.8)).bloque.sum() == normal


def test_rover_snapshot_and_restore() -> None:
    carte = petite_carte()
    carte.recevoir({'source': 'rover', 't': 0, 'z': 2, 'x': 0, 'y': 0, 'data': np.zeros((8, 8))})
    instant = carte.instantane_rover()
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': 0, 'y': 0, 'data': np.ones((8, 8))})
    carte.recevoir({'source': 'rover', 't': 1, 'z': 2, 'x': 1, 'y': 0, 'data': np.ones((8, 8))})
    carte.restaurer_rover(instant)
    assert (carte.grille(0, 0, 0.8, 0.8).val == LIBRE).all()
    assert (carte.grille(0.8, 0, 1.6, 0.8).val == INCONNU).all()


def test_no_go_zone_is_avoided_like_an_obstacle() -> None:
    carte = CarteFusion(n=16, res_fine=0.1, zmax=2, niveau_plan=2, niveau_zoom=1, rayon_robot=0.1)
    for msg in decouper(np.zeros((16, 16)), 0, 0, 0, carte, 'drone'):
        carte.recevoir(msg)
    zone = (0, 0, 6.4, 6.4)
    direct = carte.decider(DEPART, VICTIME, zone=zone)['chemin']
    carte.interdire_zone(2.1, 3.2, 0.8)  # right on the straight path
    dec = carte.decider(DEPART, VICTIME, zone=zone)
    assert dec['action'] == 'SUIVRE' and dec['chemin'] != direct
    assert all(np.hypot(x - 2.1, y - 3.2) > 0.8 for x, y in dec['chemin'])
    # it is not an obstacle of the map: nothing is written in the tiles
    assert (carte.grille(1.3, 2.4, 2.9, 4.0).val != OBSTACLE).all()


def test_superposition_un_libre_du_rover_n_efface_pas_un_obstacle_du_drone():
    """rover_efface_drone False : la carte du drone reste la carte, le rover ajoute ses
    obstacles et ne dit « libre » que la ou le drone n'a rien vu ; roles_ignores laisse une
    des deux cartes de cote."""
    carte = CarteFusion(niveau_plan=4)
    drone = np.full((32, 32), -1, np.int8)
    drone[0, 0:3] = (1, 0, 0)
    rover = np.full((32, 32), -1, np.int8)
    rover[0, 0:4] = (0, 1, -1, 0)
    carte.recevoir({'source': 'drone', 't': 1.0, 'z': 4, 'x': 0, 'y': 0, 'data': drone})
    carte.recevoir({'source': 'rover:d405', 't': 2.0, 'z': 4, 'x': 0, 'y': 0, 'data': rover})
    assert list(carte.grille(0, 0, 0.2, 0.05).val[0]) == [0, 1, 0, 0]  # regle historique
    carte.rover_efface_drone = False
    g = carte.grille(0, 0, 0.2, 0.05)
    assert list(g.val[0]) == [1, 1, 0, 0]
    assert list(g.src[0]) == [1, 2, 1, 2]  # drone, rover, drone, rover
    assert list(carte.couche('drone', 0, 0, 0.2, 0.05)[0]) == [1, 0, 0, -1]
    assert list(carte.couche('rover', 0, 0, 0.2, 0.05)[0]) == [0, 1, -1, 0]
    carte.roles_ignores = {'drone'}
    assert list(carte.grille(0, 0, 0.2, 0.05).val[0]) == [0, 1, -1, 0]
    carte.roles_ignores = {'rover'}
    assert list(carte.grille(0, 0, 0.2, 0.05).val[0]) == [1, 0, 0, -1]
