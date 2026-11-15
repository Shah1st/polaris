"""Two simulated drones (two halves of the area) and two simulated rovers, one JSON file, ONE
CarteFusion: both drones' tiles and both rovers' observations end up in the fused map, and a
mission can be planned (and driven, one rover at a time) for each rover."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from devices_helpers import CarteAvecIds
from polaris_devices import build, fusion_accepts_ids
from polaris_devices.registry import FUSION_ROLES

CONFIG = Path(__file__).resolve().parents[1] / 'examples' / 'fleet_sim.json'
VICTIM = (8.0, 20.0)


def quiet(_):
    pass


@pytest.fixture
def fleet():
    return build(CONFIG, log=quiet)


def test_every_device_is_plugged(fleet):
    assert set(fleet.devices) == {'drone-west', 'drone-east', 'rover-a', 'rover-b'}
    assert not fleet.skipped
    assert len(fleet.sources(role='drone')) == 2
    assert set(fleet.mobiles()) == {'rover-a', 'rover-b'}


def test_each_drone_covers_its_half_and_the_map_holds_both(fleet):
    counts = fleet.survey()
    assert counts['drone-west'] > 0 and counts['drone-east'] > 0
    for device_id, (lo, hi) in {'drone-west': (0, 12.8), 'drone-east': (12.8, 25.6)}.items():
        for msg in fleet.devices[device_id].tiles():
            x, _, side = fleet.carte.rect_tuile((msg['z'], msg['x'], msg['y']))
            assert lo - 1e-9 <= x and x + side <= hi + 1e-9
    g = fleet.carte.grille(0.0, 0.0, 25.6, 25.6)
    west, east = g.val[:, : g.val.shape[1] // 2], g.val[:, g.val.shape[1] // 2 :]
    assert (west != -1).mean() > 0.9 and (east != -1).mean() > 0.9
    assert fleet.zone() == (0.0, 0.0, 25.6, 25.6)


def test_both_rovers_observations_are_in_the_map(fleet):
    from fusion_carte import SRC_ROVER

    fleet.survey()
    for rover in fleet.mobiles().values():
        rover.observer(tour_complet=True)
    g = fleet.carte.grille(0.0, 0.0, 25.6, 25.6)
    for start in ((3.0, 3.0), (22.0, 3.0)):
        i, j = int(start[1] / g.res), int(start[0] / g.res)
        around = g.src[i - 20 : i + 20, j - 20 : j + 20]
        assert (around == SRC_ROVER).mean() > 0.5, start


def test_a_mission_is_planned_for_each_rover(fleet):
    fleet.survey()
    for rover_id in ('rover-a', 'rover-b'):
        dec = fleet.plan(rover_id, VICTIM)
        assert dec['action'] == 'SUIVRE', rover_id
        assert math.dist(dec['chemin'][-1], VICTIM) <= 0.5 + 0.1
        start = fleet.start_of(rover_id)
        assert math.dist(dec['chemin'][0], start) < 0.15


def test_each_drone_locates_only_the_rovers_in_its_half(fleet):
    west, east = fleet.devices['drone-west'], fleet.devices['drone-east']
    assert west.pose_of('rover-a')[:2] == (3.0, 3.0) and west.pose_of('rover-b') is None
    assert east.pose_of('rover-b')[:2] == (22.0, 3.0) and east.pose_of('rover-a') is None
    assert fleet.pose_of('rover-b')[:2] == (22.0, 3.0)


def test_zoom_goes_to_the_drone_that_sees_the_tile(fleet):
    msgs = fleet.drone_view().zoomer((3, 2, 2))  # 3.2 m tile at x 6.4-9.6: the west half
    assert len(msgs) == 1 and msgs[0]['z'] == 3
    assert msgs[0]['source'] in ('drone', 'drone:drone-west')
    assert fleet.drone_view().zoomer((3, 5, 2))[0]['source'] in ('drone', 'drone:drone-east')


def test_drive_one_rover_after_the_other(fleet):
    fleet.survey()
    for rover_id in ('rover-a', 'rover-b'):
        fin = fleet.run_mission(rover_id, VICTIM, journal=quiet)['fin']
        assert math.dist(fin['traj'][-1], VICTIM) <= 0.5 + 1e-6, rover_id


def test_drone_loss_is_seen_by_the_rovers(fleet):
    for drone in fleet.sources(role='drone'):
        drone.up = False
    assert not fleet.drone_view().disponible()
    assert not fleet.devices['rover-a'].drone_disponible()


def test_sources_carry_role_and_device_id_when_the_fusion_accepts_them():
    carte = CarteAvecIds()
    assert fusion_accepts_ids(carte)
    fleet = build(CONFIG, carte=carte, log=quiet)
    fleet.survey()
    for rover in fleet.mobiles().values():
        rover.observer()
    assert {'drone:drone-west', 'drone:drone-east', 'rover:rover-a', 'rover:rover-b'} <= set(
        carte.recu
    )
    assert all(s.split(':')[0] in FUSION_ROLES for s in carte.recu)
    assert fleet.plan('rover-b', VICTIM)['action'] == 'SUIVRE'


def test_the_real_fusion_accepts_device_ids():
    fleet = build(CONFIG, log=quiet)
    assert fleet.ids
    fleet.survey()
    assert fleet.carte.grille(0.0, 0.0, 25.6, 25.6).val.max() == 1


def test_config_round_trip_from_a_dict():
    cfg = json.loads(CONFIG.read_text())
    cfg['devices'] = cfg['devices'][:1] + cfg['devices'][2:3]
    fleet = build(cfg, log=quiet)
    assert set(fleet.devices) == {'drone-west', 'rover-a'}
    fleet.survey()
    g = fleet.carte.grille(12.8, 0.0, 25.6, 25.6)
    assert (g.val == -1).all()  # nobody looked at the east half
    assert np.isin(fleet.carte.grille(0.0, 0.0, 12.8, 25.6).val, (0, 1)).mean() > 0.9
