"""Tests of the two sets of mission settings: FIELD (the former constants) and ARENA."""

import inspect
import math

from cerveau import mission
from fusion_carte import CarteFusion
from lancer_overhead import TOLERANCE_PETITE_ZONE, lancer, preparer
from mission_settings import ARENA, FIELD
from test_lancer_overhead import FausseCamera
from test_rover_reel import FauxServeur

VICTIM = (2.0, 2.0)


def test_field_keeps_the_former_values() -> None:
    assert (FIELD.step, FIELD.careful_step, FIELD.drone_wait_max) == (0.3, 0.15, 300.0)
    assert (FIELD.uncertainty_budget, FIELD.link_backoff, FIELD.no_go_radius) == (0.6, 0.5, 1.5)
    assert (FIELD.check_horizon, FIELD.inspect_reserve, FIELD.inspect_max_distance) == (
        3.0,
        0.35,
        1.2,
    )
    carte = CarteFusion()  # a map used without a mission keeps the field values too
    assert (carte.inspect_reserve, carte.inspect_max_distance) == (0.35, 1.2)


def test_only_the_size_of_the_place_changes_in_the_arena() -> None:
    assert (ARENA.step, ARENA.careful_step, ARENA.drone_wait_max) == (
        FIELD.step,
        FIELD.careful_step,
        FIELD.drone_wait_max,
    )
    assert ARENA.no_go_radius == 0 and ARENA.uncertainty_budget < FIELD.uncertainty_budget


def test_the_overhead_launcher_uses_the_arena_settings() -> None:
    assert inspect.signature(lancer).parameters['settings'].default is ARENA


def run_with_a_wifi_cut(settings):
    """Overhead-camera mission in the 2.4 m arena, around a box. The Wi-Fi drops every answer
    for 3 s in the middle (more than the adapter's 2 s of patience: a real link loss)."""
    faux = FauxServeur(0.4, 0.4, math.pi)
    faux.coupure = (8.0, 11.0)
    camera = FausseCamera(faux)
    options = {'horloge': faux.horloge, 'dormir': faux.dormir}
    carte, rover = preparer(camera, faux.requete, options_rover=options)
    lines = []
    mission(
        rover,
        carte,
        (0.4, 0.4),
        VICTIM,
        zone=camera.zone(),
        tolerance=TOLERANCE_PETITE_ZONE,
        journal=lines.append,
        settings=settings,
    )
    return faux, carte, '\n'.join(lines)


def test_a_wifi_cut_in_the_arena_does_not_end_the_mission() -> None:
    faux, carte, log = run_with_a_wifi_cut(ARENA)
    assert 'LIAISON ROVER PERDUE' in log  # the cut did happen
    assert 'sans zone interdite' in log and not carte.zones_interdites
    assert 'ARRIVE pres de la victime' in log
    assert math.dist((faux.x, faux.y), VICTIM) <= 0.2
    assert (carte.inspect_reserve, carte.inspect_max_distance) == (0.15, 0.6)


def test_the_same_cut_with_the_field_settings_blocks_the_arena() -> None:
    """Why ARENA exists: the field's 1.5 m no-go zone covers most of the 2.4 m arena."""
    faux, carte, log = run_with_a_wifi_cut(FIELD)
    assert [z[2] for z in carte.zones_interdites] == [1.5]
    assert 'AUCUN PASSAGE' in log and 'ARRIVE pres de la victime' not in log
    assert math.dist((faux.x, faux.y), VICTIM) > 0.5
