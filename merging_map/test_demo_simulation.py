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
