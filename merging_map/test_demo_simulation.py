"""Tests for the simulated rover camera used by the demo (needs matplotlib)."""

import numpy as np
import pytest

pytest.importorskip('matplotlib')

import demo_fusion as demo  # noqa: E402
from fusion_carte import INCONNU, LIBRE, OBSTACLE, CarteFusion  # noqa: E402


def vue_rover(occlusion: bool) -> np.ndarray:
    """Rover at (4.5, 6.0), a thin wall 0.5 m in front of it (x = 5.0..5.2)."""
    carte = CarteFusion()
    verite = np.zeros((demo.NC, demo.NC), np.int8)
    demo.rect(verite, 5.0, 5.5, 5.2, 6.5)
    simu = demo.Simu(carte, verite, verite.copy(), occlusion=occlusion)
    simu.capteur_rover(4.5, 6.0)
    return carte.grille(4.0, 5.0, 7.0, 7.0).val


def case(g: np.ndarray, x: float, y: float) -> int:
    return int(g[int((y - 5.0) / 0.05), int((x - 4.0) / 0.05)])


def test_camera_cannot_see_behind_obstacles() -> None:
    g = vue_rover(occlusion=True)
    assert case(g, 4.8, 6.0) == LIBRE  # between the rover and the wall
    assert case(g, 5.05, 6.0) == OBSTACLE  # front face of the wall
    assert case(g, 5.6, 6.0) == INCONNU  # hidden behind the wall
    assert case(g, 4.5, 6.9) == LIBRE  # beside the rover, nothing in the way


def test_xray_mode_sees_everything_in_range() -> None:
    g = vue_rover(occlusion=False)
    assert case(g, 5.6, 6.0) == LIBRE
    assert case(g, 6.8, 6.0) == INCONNU  # beyond the 2 m range


def test_camera_sees_a_face_at_a_grazing_angle() -> None:
    """Regression: seen at a shallow angle, each face cell used to hide its neighbour."""
    carte = CarteFusion()
    verite = np.zeros((demo.NC, demo.NC), np.int8)
    demo.rect(verite, 4.0, 6.0, 14.0, 6.5)  # long log
    demo.Simu(carte, verite, verite.copy(), occlusion=True).capteur_rover(12.3, 5.4)
    g = carte.grille(12.0, 6.0, 14.0, 6.05)  # the log's south face, x = 12..14
    assert (g.val == OBSTACLE).all()


def test_narrow_field_of_view_sees_only_in_front() -> None:
    carte = CarteFusion()
    verite = np.zeros((demo.NC, demo.NC), np.int8)
    demo.Simu(carte, verite, verite.copy(), champ=60).capteur_rover(6.0, 6.0, cap=0.0)  # east
    g = carte.grille(4.0, 5.0, 7.0, 7.0).val
    assert case(g, 6.9, 6.0) == LIBRE  # in front
    assert case(g, 4.5, 6.0) == INCONNU  # behind
    assert case(g, 6.0, 6.9) == INCONNU  # on the side, 90 degrees away


def test_contact_stops_the_rover_before_an_unseen_obstacle() -> None:
    verite = np.zeros((demo.NC, demo.NC), np.int8)
    demo.rect(verite, 5.0, 5.0, 5.2, 7.0)
    simu = demo.Simu(CarteFusion(), verite, verite.copy())
    arret, impact = simu.contact((4.8, 6.0), (5.1, 6.0))
    assert 5.0 <= impact[0] < 5.05
    assert arret[0] < 5.0
    assert simu.contact((4.0, 6.0), (4.5, 6.0)) is None


@pytest.mark.parametrize(
    'monde, occlusion, champ',
    [
        (demo.monde_1, True, 360),
        (demo.monde_1, False, 360),
        (demo.monde_1, True, 60),
        (demo.monde_2, True, 360),
        (demo.monde_3, True, 60),
        (demo.monde_4, True, 160),  # drone lost for 110 s
        (demo.monde_5, True, 160),  # drone lost for good, hidden gap closer
    ],
)
def test_rover_never_drives_through_a_real_obstacle(monde, occlusion: bool, champ: int) -> None:
    """Even with a narrow camera or without the drone (drifting odometry), the rover never
    ends up inside an obstacle: it sees it in time, or bumps into it, stops and replans.
    The check uses the REAL trajectory, not the one the rover believes it followed."""
    _, simu, depart, victime = monde()
    simu.occlusion, simu.champ = occlusion, champ
    simu.survol_initial(z=0)
    traj = demo.mission(simu, depart, victime, journal=lambda _: None)['fin']['traj_vraie']
    for (x0, y0), (x1, y1) in zip(traj, traj[1:], strict=False):
        for s in np.linspace(0.0, 1.0, 40):
            x, y = x0 + (x1 - x0) * s, y0 + (y1 - y0) * s
            assert simu.verite[int(y / demo.RES), int(x / demo.RES)] != OBSTACLE, (x, y)


@pytest.mark.parametrize('monde', [demo.monde_4, demo.monde_5])
def test_rover_reaches_the_victim_without_the_drone(monde) -> None:
    lignes = []
    _, simu, depart, victime = monde()
    simu.champ = 160
    simu.survol_initial(z=0)
    demo.mission(simu, depart, victime, journal=lignes.append)
    texte = '\n'.join(lignes)
    assert 'LIAISON DRONE PERDUE' in texte
    assert "tour d'horizon" in texte  # it inspected obstacles by itself
    assert 'ARRIVE pres de la victime' in texte
    assert abs(complex(*simu.vraie) - complex(*victime)) < 0.8  # really there, not just believed


def test_odometry_drifts_only_without_the_drone() -> None:
    _, simu, _, _ = demo.monde_4(((5.0, None),))
    simu.placer((3.0, 3.0))
    avant = simu.vraie
    assert simu.deplacement_reel((3.0, 3.0), (3.0, 4.0), avant) == (3.0, 4.0)  # t=0: drone OK
    simu.t = 10.0
    reel = simu.deplacement_reel((3.0, 3.0), (3.0, 4.0), avant)
    assert reel != (3.0, 4.0)  # slip and heading drift
    assert abs(complex(*reel) - complex(3.0, 4.0)) < 0.05


def test_relocalisation_moves_provisional_observations() -> None:
    """Observations made without the drone are rewritten at their corrected place."""
    carte = CarteFusion()
    verite = np.zeros((demo.NC, demo.NC), np.int8)
    demo.rect(verite, 5.0, 5.0, 5.2, 7.0)  # a wall
    simu = demo.Simu(carte, verite, verite.copy(), coupures=((0.0, 10.0),))
    simu.placer((4.0, 6.0))
    simu.est = (4.0, 5.5)  # the rover believes it is 50 cm south of where it really is
    simu.dist_fix = 1.0
    simu.observer(tour_complet=True)
    # the wall really starts at y = 5.0, but it was written 50 cm too low: a ghost at y = 4.6
    assert carte.grille(5.0, 4.6, 5.05, 4.65).val[0, 0] == OBSTACLE
    simu.t = 10.0  # the drone is back and sees the rover where it really is
    simu.recaler()
    assert carte.grille(5.0, 4.55, 5.2, 4.95).val.max() < OBSTACLE  # ghost gone
    assert (carte.grille(5.0, 5.05, 5.05, 6.95).val == OBSTACLE).all()  # wall at its place
