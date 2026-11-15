"""A device with no code in this repository: the 40-line example serves /map over HTTP. Also
the MuJoCo simulator's drone (sim/run_world.py API), against a mock of that API."""

import importlib.util
import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
from devices_helpers import CarteAvecIds, serve
from polaris_devices import build
from polaris_devices.tile_http import level_of, messages_from_answer

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'tile_device.py'


def quiet(_):
    pass


def example_handler(x0, y0):
    spec = importlib.util.spec_from_file_location('tile_device_example', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_handler(x0, y0)


def tile_http(url, **settings):
    spec = {'id': 'box', 'role': 'drone', 'kind': 'tile-http', 'address': url}
    return {**spec, 'settings': settings}


def test_the_example_device_feeds_the_fusion():
    with serve(example_handler(1.0, 0.5), 8171) as url:
        carte = CarteAvecIds(niveau_plan=4)
        fleet = build({'devices': [tile_http(url)]}, carte=carte, log=quiet, strict=True)
        assert fleet.devices['box'].available()
        assert fleet.survey()['box'] > 0
        assert set(carte.recu) == {'drone:box'}
        g = carte.grille(1.0, 0.5, 2.0, 1.1)
        assert g.val.shape == (12, 20)
        assert (g.val[4:8, 8:12] == 1).all()  # the box, at x 1.4-1.6, y 0.7-0.9
        assert (g.val == 1).sum() == 16 and (g.val == 0).sum() == 12 * 20 - 16
        # a zoom re-reads /map and keeps what covers the tile
        z4 = fleet.drone_view().zoomer((4, 0, 0))
        assert z4 and all(m['source'] == 'drone:box' for m in z4)
        assert fleet.drone_view().zoomer((4, 5, 5)) == []


class Unavailable(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_error(503)

    def log_message(self, *args):
        pass


def test_a_device_answering_503_is_unavailable_and_publishes_nothing():
    with serve(Unavailable, 8173) as url:
        fleet = build({'devices': [tile_http(url)]}, log=quiet, strict=True)
        assert not fleet.devices['box'].available()
        assert fleet.survey() == {'box': 0}
        assert not fleet.drone_view().disponible()


def test_a_dead_device_is_unavailable():
    fleet = build({'devices': [tile_http('http://127.0.0.1:8179', timeout=0.3)]}, log=quiet)
    assert not fleet.devices['box'].available()
    assert fleet.survey() == {'box': 0}


class WithPose(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = {
            '/map': {'res': 0.2, 'x0': 0.0, 'y0': 0.0, 'cells': [[0, 1], [-1, 0]]},
            '/pose': {'r1': [1.0, 2.0, None], 'r2': [3.0, 4.0, 0.5]},
        }.get(self.path)
        if body is None:
            return self.send_error(404)
        data = json.dumps(body).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_a_device_can_also_locate_and_send_coarse_cells():
    with serve(WithPose, 8174) as url:
        fleet = build({'devices': [tile_http(url, locates=True)]}, log=quiet, strict=True)
        assert fleet.pose_of('r1') == (1.0, 2.0, None)
        assert fleet.pose_of('r2') == (3.0, 4.0, 0.5)
        assert fleet.pose_of('r3') is None
        fleet.survey()
        g = fleet.carte.grille(0.0, 0.0, 0.4, 0.4)
        assert g.val[0, 4] == 1 and g.val[4, 0] == -1 and g.val[0, 0] == 0


def test_levels_and_pre_cut_tiles():
    carte = CarteAvecIds()
    assert level_of(0.05, carte) == 4 and level_of(0.8, carte) == 0
    with pytest.raises(ValueError):
        level_of(0.07, carte)
    tile = np.zeros((32, 32), int).tolist()
    before = time.monotonic()
    msgs = messages_from_answer({'t': 5, 'tiles': [{'z': 2, 'x': 1, 'y': 0, 'data': tile}]}, carte)
    assert before <= msgs[0].pop('t') <= time.monotonic()  # received now, not the device's t
    assert msgs == [{'z': 2, 'x': 1, 'y': 0, 'data': tile}]


# ---- the MuJoCo simulator's drone ------------------------------------------------------------
def mujoco_handler():
    cv2 = pytest.importorskip('cv2')
    grid = np.zeros((64, 64), np.uint8)  # 3.2 m square, origin (-1.6, -1.6), y up in the image
    grid[10:20, 40:50] = 255  # obstacle: image rows 10-19 from the top = y 0.6-1.1 m
    grid[50:, :] = 128  # not seen
    png = cv2.imencode('.png', grid)[1].tobytes()
    state = {
        'rover': {'x': 0.3, 'y': -0.2, 'yaw': 1.0},
        'grid': {'res': 0.05, 'origin': [-1.6, -1.6], 'shape': [64, 64]},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data, kind = (
                (png, 'image/png')
                if self.path.startswith('/grid/overhead.png')
                else (json.dumps(state).encode(), 'application/json')
            )
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def test_the_mujoco_simulator_drone():
    with serve(mujoco_handler(), 8170) as url:
        spec = {
            'id': 'simdrone',
            'role': 'drone',
            'kind': 'sim-drone',
            'address': url,
            'settings': {'level': 4, 'tracks': 'simrover'},
        }
        fleet = build({'devices': [spec]}, log=quiet, strict=True)
        drone = fleet.devices['simdrone']
        assert drone.available()
        assert fleet.survey()['simdrone'] > 0
        assert drone.pose_of('simrover') == (0.3, -0.2, 1.0)
        assert drone.pose_of('other') is None
        g = fleet.carte.grille(-1.6, -1.6, 1.6, 1.6)
        # image row r from the top is matrix row 63 - r from the south
        assert (g.val[44:54, 40:50] == 1).all()
        assert (g.val[:14, :] == -1).all()
        assert (g.val[14:44, :] == 0).all()
        assert fleet.zone() is not None
        assert drone.zoom((4, -1, -1))
