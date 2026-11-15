"""Registry of device kinds, and the fleet built from a JSON configuration.

    from polaris_devices import build
    fleet = build('devices/examples/fleet_sim.json')   # one CarteFusion, every device plugged
    fleet.survey()                                     # every map source -> the fusion
    fleet.plan('rover-a', victim=(8.0, 20.0))          # CarteFusion.decider for that rover
    fleet.run_mission('rover-a', victim=(8.0, 20.0))   # cerveau.mission, one rover at a time

Configuration (JSON)::

    {"map": {"niveau_plan": 3, "rayon_robot": 0.15},          # CarteFusion arguments
     "devices": [
       {"id": "cam", "role": "drone", "kind": "overhead-camera", "source": "0",
        "settings": {...}},
       {"id": "r1", "role": "rover", "kind": "rasprover", "address": "http://...:8080",
        "settings": {"locator": "cam", "perception": "r1-eye"}}]}

`role` is the device's role in the fusion: "drone" (sees from above: its tiles are the global
map) or "rover" (on the ground: its tiles win over the drone's where it has seen). Its tiles are
published with source "<role>:<id>" (or plain "<role>" when the fusion in this tree does not
accept ids yet). `kind` picks the adapter (adapters.py, or your own: register()).
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from .paths import ensure_repo_paths
from .roles import Mobile, roles_of

FUSION_ROLES = ('drone', 'rover')

#: kind -> factory(spec, fleet) -> device object. Filled by adapters.py and register().
KINDS: dict[str, Callable] = {}


class DeviceUnavailable(RuntimeError):  # noqa: N818  (reads better at the raise site)
    """The device's kind cannot be used here (a missing module, a missing service)."""


def register(kind: str, factory: Callable) -> None:
    """Make `kind` usable in a configuration. `factory(spec, fleet)` returns the device object;
    `spec` is the device's JSON entry (a DeviceSpec) and `fleet` the Fleet being built (its
    `carte`, already-built `devices`, `drone_view()`...). Raise DeviceUnavailable (or let an
    ImportError through) when a dependency is missing: only that device is skipped."""
    KINDS[kind] = factory


@dataclass
class DeviceSpec:
    id: str
    role: str
    kind: str
    address: str | None = None
    source: str | None = None
    settings: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> DeviceSpec:
        missing = [k for k in ('id', 'role', 'kind') if k not in d]
        if missing:
            raise ValueError(f'device {d!r}: missing {", ".join(missing)}')
        if d['role'] not in FUSION_ROLES:
            raise ValueError(f'device {d["id"]!r}: role must be one of {FUSION_ROLES}')
        unknown = set(d) - {'id', 'role', 'kind', 'address', 'source', 'settings', 'comment'}
        if unknown:
            raise ValueError(f'device {d["id"]!r}: unknown keys {sorted(unknown)}')
        return cls(
            str(d['id']),
            d['role'],
            d['kind'],
            d.get('address'),
            d.get('source'),
            dict(d.get('settings') or {}),
        )


def fusion_accepts_ids(carte) -> bool:
    """Does this fusion accept sources 'drone:<id>' (interface 7)? Probed on a throwaway map of
    the same class, so the real map is not touched."""
    import numpy as np

    probe = type(carte)(n=carte.n, res_fine=carte.res_fine, zmax=carte.zmax)
    tile = np.full((carte.n, carte.n), -1, np.int8)
    try:
        probe.recevoir({'source': 'drone:probe', 't': 0.0, 'z': 0, 'x': 0, 'y': 0, 'data': tile})
    except ValueError:
        return False
    return True


class TaggedMap:
    """The fusion as seen by one rover: what it publishes as plain 'rover' is re-labelled
    'rover:<id>'. Everything else goes straight to the shared map.

    The rover adapters keep a snapshot of the rover tiles to rewrite them after a drone loss
    (robot.RobotOdometrie._reprojeter). If the fusion stores 'rover:<id>' apart from 'rover',
    that layer is snapshotted and restored here too."""

    def __init__(self, carte, role: str, tag: str) -> None:
        object.__setattr__(self, '_carte', carte)
        object.__setattr__(self, '_role', role)
        object.__setattr__(self, '_tag', tag)

    def __getattr__(self, name):
        return getattr(self._carte, name)

    def __setattr__(self, name, value):
        setattr(self._carte, name, value)

    def recevoir(self, msg):
        if self._tag != self._role and msg.get('source') == self._role:
            msg = {**msg, 'source': self._tag}
        return self._carte.recevoir(msg)

    # signaler_obstacle() goes to the fusion unchanged (plain 'rover'): it keeps the fusion's
    # own rule that a felt obstacle is always the newest rover value of its tile.

    def _layer(self):
        tuiles, maj = getattr(self._carte, 'tuiles', {}), getattr(self._carte, 'maj', {})
        if self._tag == self._role or self._tag not in tuiles:
            return None
        return {k: v.copy() for k, v in tuiles[self._tag].items()}, dict(maj.get(self._tag, {}))

    def instantane_rover(self):
        return self._carte.instantane_rover(), self._layer()

    def restaurer_rover(self, instantane):
        base, own = instantane
        self._carte.restaurer_rover(base)
        tuiles, maj = getattr(self._carte, 'tuiles', {}), getattr(self._carte, 'maj', {})
        if self._tag in tuiles:  # created after the snapshot: empty it
            tuiles[self._tag] = {k: v.copy() for k, v in own[0].items()} if own else {}
            if self._tag in maj:
                maj[self._tag] = dict(own[1]) if own else {}


class DroneView:
    """All the fleet's map sources as the single `drone` object merging_map expects
    (RaspRoverReel(drone=...)): disponible() and zoomer(tile)."""

    def __init__(self, fleet: Fleet) -> None:
        self.fleet = fleet

    def disponible(self) -> bool:
        return any(
            self.fleet._safe(d, 'available', False) for d in self.fleet.sources(role='drone')
        )

    def zoomer(self, tuile) -> list[dict]:
        out = []
        for device in self.fleet.sources():
            if not callable(getattr(device, 'zoom', None)):
                continue
            if not self.fleet._safe(device, 'available', False):
                continue
            spec = self.fleet.specs[self.fleet.id_of(device)]
            for msg in device.zoom(tuile) or []:
                out.append({**msg, 'source': self.fleet.tag(spec)})
        return out


class Fleet:
    """Every device of a configuration around ONE fusion map."""

    def __init__(self, carte, log: Callable[[str], None] = print) -> None:
        self.carte = carte
        self.log = log
        self.devices: dict[str, object] = {}
        self.specs: dict[str, DeviceSpec] = {}
        self.skipped: dict[str, str] = {}  # device id -> why it is not plugged
        self.shared: dict = {}  # state shared by adapters of one kind (simulated worlds...)
        self.ids = fusion_accepts_ids(carte)
        self._rects: dict[str, list[tuple[float, float, float, float]]] = {}

    # ---- naming -------------------------------------------------------------------------
    def tag(self, spec: DeviceSpec) -> str:
        return f'{spec.role}:{spec.id}' if self.ids else spec.role

    def id_of(self, device) -> str:
        for k, d in self.devices.items():
            if d is device:
                return k
        raise KeyError(device)

    def map_for(self, spec: DeviceSpec):
        """The fusion as a rover adapter should hold it (its plain 'rover' tiles get its id)."""
        return TaggedMap(self.carte, spec.role, self.tag(spec))

    # ---- views by role --------------------------------------------------------------------
    def sources(self, role: str | None = None) -> list:
        return [
            d
            for k, d in self.devices.items()
            if 'map' in roles_of(d) and (role is None or self.specs[k].role == role)
        ]

    def locators(self) -> list:
        return [d for d in self.devices.values() if 'locator' in roles_of(d)]

    def mobiles(self) -> dict[str, object]:
        return {k: d for k, d in self.devices.items() if isinstance(d, Mobile)}

    def drone_view(self) -> DroneView:
        return DroneView(self)

    def _safe(self, device, method, default):
        try:
            return getattr(device, method)()
        except Exception as e:  # a broken device must not stop the others
            self.log(f'device {self.id_of(device)}: {method}() failed: {e}')
            return default

    # ---- map ---------------------------------------------------------------------------------
    def publish(self, spec: DeviceSpec, messages) -> int:
        n = 0
        for msg in messages or []:
            self.carte.recevoir({**msg, 'source': self.tag(spec)})
            if spec.role == 'drone':
                x, y, side = self.carte.rect_tuile((int(msg['z']), msg['x'], msg['y']))
                self._rects.setdefault(spec.id, []).append((x, y, x + side, y + side))
            n += 1
        return n

    def survey(self) -> dict[str, int]:
        """Every available map source publishes its tiles once. Returns tiles per device."""
        counts = {}
        for device in self.sources():
            device_id = self.id_of(device)
            if not self._safe(device, 'available', False):
                counts[device_id] = 0
                continue
            tiles = self._safe(device, 'tiles', [])
            counts[device_id] = self.publish(self.specs[device_id], tiles)
        return counts

    def zone(self):
        """Planning area (x0, y0, x1, y1): the union of what the drones cover, or None. A
        drone that knows its own area (OverheadCamera.zone(): the image bounds) gives it;
        otherwise the bounds of the tiles it published. Computed here because
        CarteFusion.zone_survolee() only looks at the plain 'drone' layer."""
        boxes = []
        for device_id, rects in self._rects.items():
            own = getattr(self.devices.get(device_id), 'zone', None)
            box = own() if callable(own) else None
            boxes.append(box or _union(rects))
        return _union(boxes) if boxes else self.carte.zone_survolee()

    # ---- where things are ----------------------------------------------------------------------
    def pose_of(self, device_id: str):
        """First answer of the locators, in declaration order; None if nobody sees it."""
        for locator in self.locators():
            try:
                pose = locator.pose_of(device_id)
            except Exception as e:
                self.log(f'locator {self.id_of(locator)}: pose_of({device_id}) failed: {e}')
                continue
            if pose is not None:
                return pose
        return None

    def locator_for(self, spec: DeviceSpec) -> Callable[[], object]:
        """position_drone() for a rover adapter: its declared locator, else every locator."""
        name = spec.settings.get('locator')
        if name is None:
            return lambda: self.pose_of(spec.id)
        locator = self.devices.get(name)
        if locator is None:
            raise DeviceUnavailable(f'{spec.id}: locator {name!r} is not plugged')
        return lambda: locator.pose_of(spec.id)

    # ---- missions --------------------------------------------------------------------------
    def start_of(self, rover_id: str):
        pose = self.pose_of(rover_id)
        if pose is not None:
            return (pose[0], pose[1])
        start = self.specs[rover_id].settings.get('start')
        if start is not None:
            return tuple(start)
        return self.devices[rover_id].position()

    def plan(self, rover_id: str, victim, zone=None, **kwargs) -> dict:
        """One decision of the planner for this rover (CarteFusion.decider), no driving."""
        return self.carte.decider(
            self.start_of(rover_id), tuple(victim), zone=zone or self.zone(), **kwargs
        )

    def run_mission(self, rover_id: str, victim, zone=None, **kwargs):
        """cerveau.mission for ONE rover. Run them one after another: the brain does not
        coordinate several rovers moving at the same time (no cell reservation)."""
        ensure_repo_paths()
        from cerveau import mission

        rover = self.mobiles()[rover_id]
        return mission(
            rover,
            self.carte,
            self.start_of(rover_id),
            tuple(victim),
            zone=zone or self.zone(),
            **kwargs,
        )


def _union(boxes):
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def load_config(config) -> dict:
    if isinstance(config, dict):
        return config
    with open(os.fspath(config), encoding='utf-8') as f:
        return json.load(f)


def build(config, carte=None, strict: bool = False, log: Callable[[str], None] = print) -> Fleet:
    """Build every device of `config` (a path to a JSON file, or the same dict) around one
    CarteFusion (`carte`, or one made from config['map']).

    Devices are built in declaration order, so declare locators and sensors before the rovers
    that use them. A device whose kind is unknown or unavailable (missing module or service) is
    skipped with a message (fleet.skipped), unless strict=True."""
    from . import adapters  # noqa: F401  (registers the built-in kinds)

    cfg = load_config(config)
    if carte is None:
        ensure_repo_paths()
        from fusion_carte import CarteFusion

        carte = CarteFusion(**cfg.get('map', {}))
    fleet = Fleet(carte, log)
    if not fleet.ids:
        log('fusion: device ids not accepted yet, tiles published as plain "drone"/"rover"')
    seen = set()
    for entry in cfg.get('devices', []):
        spec = DeviceSpec.from_dict(entry)
        if spec.id in seen:
            raise ValueError(f'device id {spec.id!r} declared twice')
        seen.add(spec.id)
        factory = KINDS.get(spec.kind)
        try:
            if factory is None:
                raise DeviceUnavailable(f'unknown kind {spec.kind!r} (known: {sorted(KINDS)})')
            fleet.specs[spec.id] = spec
            fleet.devices[spec.id] = factory(spec, fleet)
        except (DeviceUnavailable, ImportError, OSError) as e:
            fleet.specs.pop(spec.id, None)
            fleet.devices.pop(spec.id, None)
            if strict:
                raise
            fleet.skipped[spec.id] = f'{type(e).__name__}: {e}'
            log(f'device {spec.id} ({spec.kind}) skipped: {e}')
            continue
        roles = roles_of(fleet.devices[spec.id])
        log(f'device {spec.id}: {spec.kind}, {spec.role}, plays {", ".join(roles) or "nothing"}')
    return fleet


def main(argv=None) -> int:
    """`python -m polaris_devices fleet.json`: build, survey, print what is plugged."""
    import argparse

    p = argparse.ArgumentParser(description='Plug the devices of a configuration and survey.')
    p.add_argument('config')
    p.add_argument('--victim', type=float, nargs=2, metavar=('X', 'Y'))
    args = p.parse_args(argv)
    fleet = build(args.config)
    for device_id, n in fleet.survey().items():
        print(f'{device_id}: {n} tiles')
    print(f'zone: {fleet.zone()}')
    if args.victim:
        for rover_id in fleet.mobiles():
            dec = fleet.plan(rover_id, args.victim)
            path = dec.get('chemin') or []
            length = sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1))
            print(f'{rover_id}: {dec["action"]}, path {length:.2f} m')
    return 1 if fleet.skipped else 0


if __name__ == '__main__':
    sys.exit(main())
