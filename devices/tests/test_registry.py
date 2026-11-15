"""The registry: configuration checks, a missing dependency only disables its kind, rover
tiles get the rover's id, real-hardware adapters are built without talking to the hardware."""

import importlib.util
from http.server import BaseHTTPRequestHandler

import numpy as np
import pytest
from devices_helpers import CarteAvecIds, serve
from polaris_devices import (
    DeviceSpec,
    DeviceUnavailable,
    Locator,
    MapSource,
    Mobile,
    TaggedMap,
    build,
    register,
)
from polaris_devices.registry import KINDS
from polaris_devices.roles import roles_of


def quiet(_):
    pass


SIM_DRONE = {'id': 'd', 'role': 'drone', 'kind': 'sim-drone', 'settings': {'level': 2}}


def test_spec_rejects_what_would_silently_do_nothing():
    with pytest.raises(ValueError, match='missing'):
        DeviceSpec.from_dict({'id': 'x', 'role': 'drone'})
    with pytest.raises(ValueError, match='role'):
        DeviceSpec.from_dict({'id': 'x', 'role': 'camera', 'kind': 'tile-http'})
    with pytest.raises(ValueError, match='unknown keys'):
        DeviceSpec.from_dict({'id': 'x', 'role': 'drone', 'kind': 'k', 'adress': 'typo'})
    with pytest.raises(ValueError, match='twice'):
        build({'devices': [SIM_DRONE, SIM_DRONE]}, log=quiet)


def test_a_missing_dependency_only_disables_that_kind():
    def needs_a_missing_module(spec, fleet):
        import module_that_is_not_installed  # noqa: F401

    register('test-missing-module', needs_a_missing_module)
    try:
        cfg = {
            'devices': [
                {'id': 'broken', 'role': 'drone', 'kind': 'test-missing-module'},
                {'id': 'nokind', 'role': 'drone', 'kind': 'no-such-kind'},
                SIM_DRONE,
            ]
        }
        fleet = build(cfg, log=quiet)
        assert set(fleet.devices) == {'d'}
        assert set(fleet.skipped) == {'broken', 'nokind'}
        assert 'module_that_is_not_installed' in fleet.skipped['broken']
        assert fleet.survey()['d'] > 0
        with pytest.raises(ImportError):
            build(cfg, log=quiet, strict=True)
    finally:
        KINDS.pop('test-missing-module')


def test_rover_perception_is_plugged_only_when_its_module_exists():
    cfg = {
        'devices': [
            {'id': 'eye', 'role': 'rover', 'kind': 'rover-perception', 'address': 'http://x'}
        ]
    }
    fleet = build(cfg, log=quiet)
    if importlib.util.find_spec('perception_rover') is None:
        assert 'eye' in fleet.skipped  # merging_map/perception_rover.py not in this tree yet
    else:
        assert callable(fleet.devices['eye'].percevoir)


def test_roles_are_recognised_from_the_methods():
    fleet = build({'devices': [SIM_DRONE]}, log=quiet)
    drone = fleet.devices['d']
    assert isinstance(drone, MapSource) and isinstance(drone, Locator)
    assert not isinstance(drone, Mobile)
    assert roles_of(drone) == ['map', 'locator']


def test_tagged_map_relabels_only_the_rovers_own_tiles():
    carte = CarteAvecIds()
    vue = TaggedMap(carte, 'rover', 'rover:r1')
    data = np.zeros((32, 32), np.int8)
    vue.recevoir({'source': 'rover', 't': 1.0, 'z': 4, 'x': 0, 'y': 0, 'data': data})
    vue.recevoir({'source': 'drone:d', 't': 1.0, 'z': 4, 'x': 1, 'y': 0, 'data': data})
    vue.signaler_obstacle(0.5, 0.5)
    assert carte.recu[:2] == ['rover:r1', 'drone:d']
    assert vue.zmax == carte.zmax
    vue.marge_drone = 0.3  # attributes written through the view land on the shared map
    assert carte.marge_drone == 0.3


def test_tagged_map_snapshot_covers_a_separate_layer():
    """A fusion that stores 'rover:<id>' apart: restoring the snapshot taken during a drone
    loss must also remove what that rover published since (ghost obstacles)."""

    class CarteParSource(CarteAvecIds):
        def recevoir(self, msg):
            s = msg['source']
            self.tuiles.setdefault(s, {})
            self.maj.setdefault(s, {})
            k = (int(msg['z']), int(msg['x']), int(msg['y']))
            self.tuiles[s][k] = np.asarray(msg['data'], np.int8)
            self.maj[s][k] = msg['t']

    carte = CarteParSource()
    vue = TaggedMap(carte, 'rover', 'rover:r1')
    data = np.ones((32, 32), np.int8)
    instantane = vue.instantane_rover()
    vue.recevoir({'source': 'rover', 't': 1.0, 'z': 4, 'x': 0, 'y': 0, 'data': data})
    assert carte.tuiles['rover:r1']
    vue.restaurer_rover(instantane)
    assert carte.tuiles['rover:r1'] == {}
    vue.recevoir({'source': 'rover', 't': 2.0, 'z': 4, 'x': 3, 'y': 0, 'data': data})
    instantane = vue.instantane_rover()
    vue.recevoir({'source': 'rover', 't': 3.0, 'z': 4, 'x': 4, 'y': 0, 'data': data})
    vue.restaurer_rover(instantane)
    assert set(carte.tuiles['rover:r1']) == {(4, 3, 0)}


class Counter(BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):  # noqa: N802
        Counter.hits.append(self.path)
        self.send_error(500)

    do_POST = do_GET  # noqa: N815

    def log_message(self, *args):
        pass


def test_rasprover_is_built_without_talking_to_the_rover():
    Counter.hits.clear()
    with serve(Counter, 8172) as url:
        cfg = {
            'map': {'niveau_plan': 4, 'rayon_robot': 0.155},
            'devices': [
                SIM_DRONE,
                {
                    'id': 'r1',
                    'role': 'rover',
                    'kind': 'rasprover',
                    'address': url,
                    'settings': {'locator': 'd', 'options': {'vitesse': 0.15}},
                },
            ],
        }
        fleet = build(cfg, log=quiet, strict=True)
        rover = fleet.devices['r1']
        assert isinstance(rover, Mobile) and rover.vitesse == 0.15
        assert rover.drone.disponible()  # the fleet's drones, through DroneView
        assert rover.position_drone() is None  # the sim drone does not see a real rover
        assert rover.carte._carte is fleet.carte
    assert Counter.hits == []


def test_rasprover_needs_its_declared_locator_and_sensor():
    for settings, missing in (({'locator': 'cam'}, 'cam'), ({'perception': 'eye'}, 'eye')):
        cfg = {
            'devices': [
                {'id': 'r1', 'role': 'rover', 'kind': 'rasprover', 'address': 'http://x'},
            ]
        }
        cfg['devices'][0]['settings'] = settings
        with pytest.raises(DeviceUnavailable, match=missing):
            build(cfg, log=quiet, strict=True)


def test_overhead_camera_from_a_still_image(tmp_path):
    cv2 = pytest.importorskip('cv2')
    image = np.full((600, 800, 3), 200, np.uint8)
    cv2.rectangle(image, (380, 280), (420, 320), (20, 20, 20), -1)
    path = tmp_path / 'overhead.png'
    cv2.imwrite(str(path), image)
    cfg = {
        'devices': [
            {
                'id': 'cam',
                'role': 'drone',
                'kind': 'overhead-camera',
                'source': str(path),
                'settings': {'overhead': {'metres_per_pixel': 0.003}},
            }
        ]
    }
    fleet = build(cfg, log=quiet, strict=True)
    cam = fleet.devices['cam']
    assert roles_of(cam) == ['map', 'locator']
    assert cam.available()
    counts = fleet.survey()
    assert counts['cam'] > 0
    assert fleet.zone() is not None
    cam.refresh()
    cam.close()
