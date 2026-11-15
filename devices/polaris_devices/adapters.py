"""Adapters for the devices that exist today. Thin wrappers: the real work stays in
merging_map, guidance_global_map and sim/. Every import is inside the factory, so a missing
module (OpenCV, the overhead camera package, the rover perception) only disables its kind.

| kind               | role  | plays            | wraps                                          |
|--------------------|-------|------------------|------------------------------------------------|
| overhead-camera    | drone | map, locator     | guidance_global_map.drone.OverheadDrone        |
| rasprover          | rover | mobile           | merging_map/rover_reel.RaspRoverReel           |
| rover-perception   | rover | (sensor)         | merging_map/perception_rover.percevoir_http    |
| sim-drone          | drone | map, locator     | merging_map/demo_fusion worlds, or sim/ HTTP   |
| sim-rover          | rover | mobile           | merging_map/demo_fusion.Simu                   |
| tile-http          | any   | map (+ locator)  | any process serving GET /map (tile_http.py)    |
"""

from __future__ import annotations

import os
import time

from . import tile_http
from .paths import ensure_repo_paths
from .registry import DeviceUnavailable, register


def _tuples(d: dict) -> dict:
    """JSON has no tuples; the overhead settings expect them for sizes."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def _in_area(x, y, area) -> bool:
    return area is None or (area[0] <= x < area[2] and area[1] <= y < area[3])


# ---- overhead camera (the arena's "drone") -------------------------------------------------
class OverheadCamera:
    """OverheadDrone as MapSource + Locator. It sees ONE rover (its marker, or the blob of the
    rover's size): settings.tracks names it; without it, it answers for the only rover."""

    def __init__(self, camera, reader, tracks, fleet) -> None:
        self.camera, self.reader, self.tracks, self.fleet = camera, reader, tracks, fleet

    def tiles(self):
        return self.camera.tiles()

    def available(self):
        return self.camera.disponible()

    def zoom(self, tile):
        return self.camera.zoomer(tile)

    def pose_of(self, device_id):
        if self.tracks is not None and device_id != self.tracks:
            return None
        if self.tracks is None and len(self.fleet.mobiles()) > 1:
            return None  # two rovers and no marker id: the camera cannot tell them apart
        return self.camera.rover_pose()

    def zone(self):
        """The image bounds (what lancer_overhead plans in), not the 1.6 m tiles around it."""
        return self.camera.zone()

    def refresh(self):
        self.camera.refresh(force=True)

    def close(self):
        close = getattr(self.reader, 'close', None)
        if close:
            close()


def make_overhead_camera(spec, fleet):
    ensure_repo_paths()
    from guidance_global_map import drone
    from guidance_global_map.pipeline import OverheadSettings

    source = spec.source or spec.address
    if source is None:
        raise DeviceUnavailable(f'{spec.id}: overhead-camera needs "source" (camera or image)')
    camera_reader = getattr(drone, 'camera_reader', None)
    if camera_reader is not None:
        reader = camera_reader(source, max_age=float(spec.settings.get('max_age', 2.0)))
    elif os.path.isfile(str(source)):
        reader = drone.image_file_reader(source)  # still photo: the rover never seems to move
    else:
        raise DeviceUnavailable(
            f'{spec.id}: no live camera reader in this tree (guidance_global_map.drone.'
            f'camera_reader), and {source!r} is not an image file'
        )
    settings = OverheadSettings(**_tuples(spec.settings.get('overhead', {})))
    camera = drone.OverheadDrone(reader, settings, period=float(spec.settings.get('period', 0.5)))
    return OverheadCamera(camera, reader, spec.settings.get('tracks'), fleet)


# ---- the real RaspRover ----------------------------------------------------------------------
def make_rasprover(spec, fleet):
    ensure_repo_paths()
    from rover_reel import RaspRoverReel, client_http

    if not spec.address:
        raise DeviceUnavailable(f'{spec.id}: rasprover needs "address" (rover_server.py URL)')
    percevoir = lambda position, cap: None  # noqa: E731  (no rover camera: drone map only)
    name = spec.settings.get('perception')
    if name in fleet.skipped:
        # Same as lancer_overhead without --perception: the rover drives on the drone's map.
        fleet.log(f'{spec.id}: perception {name!r} skipped, the rover will not see by itself')
    elif name is not None:
        sensor = fleet.devices.get(name)
        if sensor is None:
            raise DeviceUnavailable(f'{spec.id}: perception {name!r} is not declared before it')
        percevoir = sensor.percevoir
    locator = fleet.devices.get(spec.settings.get('locator'))
    if locator is not None and callable(getattr(locator, 'refresh', None)):
        refresh = locator.refresh
    else:
        refreshers = [d.refresh for d in fleet.locators() if callable(getattr(d, 'refresh', None))]
        refresh = (lambda: [r() for r in refreshers]) if refreshers else None
    return RaspRoverReel(
        fleet.map_for(spec),
        client_http(spec.address, float(spec.settings.get('timeout', 1.0))),
        position_drone=fleet.locator_for(spec),
        percevoir=percevoir,
        drone=fleet.drone_view(),
        rafraichir_drone=refresh,
        **spec.settings.get('options', {}),
    )


class RoverPerception:
    """The rover's own camera, as a service (rover_mapper/perception_service.py). Not a role
    by itself: a rover uses it through settings.perception = this device's id."""

    def __init__(self, percevoir) -> None:
        self.percevoir = percevoir


def make_rover_perception(spec, fleet):
    ensure_repo_paths()
    from perception_rover import percevoir_http  # merging_map/perception_rover.py

    if not spec.address:
        raise DeviceUnavailable(f'{spec.id}: rover-perception needs "address"')
    return RoverPerception(percevoir_http(spec.address, float(spec.settings.get('timeout', 3.0))))


# ---- simulation ---------------------------------------------------------------------------
def _world(fleet, name):
    """One simulated world (ground truth) shared by every simulated device that names it."""
    ensure_repo_paths()
    import demo_fusion

    worlds = fleet.shared.setdefault('worlds', {})
    if name not in worlds:
        factory = getattr(demo_fusion, name, None)
        if not (name.startswith('monde_') and callable(factory)):
            raise DeviceUnavailable(f'unknown simulated world {name!r} (demo_fusion.monde_N)')
        _, simu, _, _ = factory()
        worlds[name] = simu  # its verite / vu_drone / inconnu_drone; its own map is unused
    return worlds[name]


class SimWorldDrone:
    """A drone over a merging_map world, seeing only `area` (x0, y0, x1, y1), at `level`. It
    locates the simulated rovers that are inside its area (their true position)."""

    def __init__(self, fleet, world, area, level, first_unknown=True) -> None:
        self.fleet, self.world, self.area, self.level = fleet, world, area, level
        self.first_unknown = first_unknown
        self.up = True

    def _view(self, z, survey):
        import numpy as np
        from fusion_carte import INCONNU

        carte = self.fleet.carte
        f = 2 ** (carte.zmax - z)
        vu = self.world.vu_drone
        n = vu.shape[0] // f
        m = vu.reshape(n, f, n, f).max(axis=(1, 3)).astype(np.int8)
        r = carte.res(z)
        if survey and self.first_unknown and self.world.inconnu_drone:
            x0, y0, x1, y1 = self.world.inconnu_drone  # what Simu.survol_initial hides too
            m[int(y0 / r) : int(y1 / r), int(x0 / r) : int(x1 / r)] = INCONNU
        if self.area is not None:
            c = (np.arange(n) + 0.5) * r
            inside = (c[None, :] >= self.area[0]) & (c[None, :] < self.area[2])
            inside = inside & (c[:, None] >= self.area[1]) & (c[:, None] < self.area[3])
            m[~inside] = INCONNU
        return m

    def available(self):
        return self.up

    def tiles(self):
        from fusion_carte import decouper

        m = self._view(self.level, survey=True)
        return decouper(m, 0.0, 0.0, self.level, self.fleet.carte, 'drone', time.monotonic())

    def zoom(self, tile):
        z, tx, ty = tile
        n = self.fleet.carte.n
        m = self._view(z, survey=False)
        if not (0 <= ty * n < m.shape[0] and 0 <= tx * n < m.shape[1]):
            return []
        data = m[ty * n : (ty + 1) * n, tx * n : (tx + 1) * n]
        if (data == -1).all():
            return []
        return [{'t': time.monotonic(), 'z': z, 'x': tx, 'y': ty, 'data': data}]

    def pose_of(self, device_id):
        rover = self.fleet.devices.get(device_id)
        where = getattr(rover, 'position_reelle', lambda: None)()
        if where is None or not _in_area(*where, self.area):
            return None
        return where[0], where[1], getattr(rover, 'cap', None)


class SimHttpDrone:
    """The drone of the MuJoCo simulator (sim/run_world.py): GET /grid/overhead.png and the
    grid origin from GET /state; it locates the simulator's single rover (settings.tracks)."""

    def __init__(self, fleet, url, level, tracks, timeout=2.0) -> None:
        self.fleet, self.url, self.level = fleet, url.rstrip('/'), level
        self.tracks, self.timeout = tracks, timeout
        self.grid = None  # (fine matrix, origin)

    def _state(self):
        return tile_http.get_json(self.url + '/state', self.timeout)

    def _read(self):
        import urllib.request

        from fusion_feed import RES, drone_grid_from_png

        state = self._state()
        if not state or 'grid' not in state:
            return None
        if abs(float(state['grid']['res']) - RES) > 1e-9:
            raise ValueError(f'simulator grid cell {state["grid"]["res"]} m, expected {RES}')
        try:
            with urllib.request.urlopen(self.url + '/grid/overhead.png', timeout=10) as r:
                png = r.read()
        except OSError:
            return None
        self.grid = (drone_grid_from_png(png), tuple(float(v) for v in state['grid']['origin']))
        return self.grid

    def _messages(self, z):
        from fusion_carte import decouper, reduire

        if self.grid is None and self._read() is None:
            return []
        m, (ox, oy) = self.grid
        coarse = reduire(m, 2 ** (self.fleet.carte.zmax - z))
        return decouper(coarse, ox, oy, z, self.fleet.carte, 'drone', time.monotonic())

    def available(self):
        return self._state() is not None

    def tiles(self):
        self._read()
        return self._messages(self.level)

    def zoom(self, tile):
        return [m for m in self._messages(tile[0]) if (m['x'], m['y']) == tuple(tile[1:])]

    def pose_of(self, device_id):
        if self.tracks is not None and device_id != self.tracks:
            return None
        rover = (self._state() or {}).get('rover')
        if not rover:
            return None
        return float(rover['x']), float(rover['y']), float(rover['yaw'])


def make_sim_drone(spec, fleet):
    s = spec.settings
    area = s.get('area')
    if spec.address or s.get('url'):
        ensure_repo_paths()
        import fusion_feed  # noqa: F401  (needs OpenCV: fail now, not at the first survey)

        return SimHttpDrone(
            fleet, spec.address or s['url'], int(s.get('level', 2)), s.get('tracks')
        )
    world = _world(fleet, s.get('world', 'monde_1'))
    return SimWorldDrone(
        fleet,
        world,
        tuple(area) if area else None,
        int(s.get('level', 0)),
        bool(s.get('first_unknown', True)),
    )


def make_sim_rover(spec, fleet):
    ensure_repo_paths()
    from demo_fusion import Simu

    s = spec.settings
    world = _world(fleet, s.get('world', 'monde_1'))

    class SimRover(Simu):
        """Simu whose zoom requests go to the fleet's drones, not to its own world's drone."""

        def demander_zoom(self, tuile):
            for msg in fleet.drone_view().zoomer(tuile):
                self.carte.recevoir(msg)

        def drone_disponible(self):
            return self.connecte() and fleet.drone_view().disponible()

    rover = SimRover(
        fleet.map_for(spec),
        world.verite,
        world.vu_drone,
        occlusion=bool(s.get('occlusion', True)),
        champ=float(s.get('champ', 360)),
        coupures=[tuple(c) for c in s.get('coupures', [])],
        graine=int(s.get('seed', 0)),
    )
    start = s.get('start')
    if start is None:
        raise DeviceUnavailable(f'{spec.id}: sim-rover needs settings.start [x, y]')
    rover.placer(tuple(start))
    if 'heading' in s:
        rover.cap = float(s['heading'])
    return rover


register('overhead-camera', make_overhead_camera)
register('rasprover', make_rasprover)
register('rover-perception', make_rover_perception)
register('sim-drone', make_sim_drone)
register('sim-rover', make_sim_rover)
register('tile-http', tile_http.make)
