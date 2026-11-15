"""The three roles a device can play. Each docstring is the specification to implement.

A device is any object; the registry only checks which of these methods it has. One device may
play several roles (the overhead camera is a MapSource AND a Locator). Everything is in the
MAP frame: metres, x right / east, y up / north, origin chosen by whoever set up the map (today:
the bottom-left corner of the overhead camera's image).

Tile message (merging_map/README.md, `CarteFusion.recevoir`)::

    {'source': str, 't': seconds, 'z': level, 'x': tx, 'y': ty, 'data': N x N of -1/0/1}

1 = obstacle, 0 = free AND actually seen, -1 = not seen. A hidden cell sent as 0 erases the
obstacles other devices saw: when in doubt, send -1. Row 0 is SOUTH. The registry overwrites
'source' with '<role>:<device id>', so a device does not need to know its own name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Pose = tuple[float, float, 'float | None']
TileKey = tuple[int, int, int]


@runtime_checkable
class MapSource(Protocol):
    """Anything that produces map tiles: an overhead camera, a drone, a rover's own sensor, a
    fixed camera on a wall, another team's map.

    Required:
      tiles() -> list[dict]
          Everything this device currently knows, as tile messages (format above). Called once
          for the first survey and again whenever the fleet refreshes. Return [] when it has
          nothing. Must not block for more than about a second.
      available() -> bool
          True while the device is alive and its tiles are current. False makes the brain go
          into "drone lost" mode if no other drone is available.

    Optional:
      zoom(tile: (z, x, y)) -> list[dict]
          The brain wants a closer / fresher look at tile (z, x, y) (CarteFusion.decider's
          DEMANDER_DRONE). Return the tile messages that cover it, possibly at a finer level;
          [] if this device cannot see it. Without zoom(), the device is simply not asked.
    """

    def tiles(self) -> list[dict]: ...

    def available(self) -> bool: ...


@runtime_checkable
class Locator(Protocol):
    """Anything that can say where a device is: an overhead camera that sees an ArUco marker, a
    motion-capture system, a simulator's ground truth.

      pose_of(device_id: str) -> (x, y, heading) | None
          Pose of the device `device_id` in the map frame, heading in radians (0 = +x, counter-
          clockwise) or None when it is not known (rover found by its size, no marker). None
          when this locator does not see that device right now. Several locators may answer for
          the same device: the fleet takes the first one that answers, in declaration order.

    Optional:
      refresh() -> None
          Take a new measurement now (the rover has just stopped). Without it, the rover
          adapter waits a little before reading pose_of().
    """

    def pose_of(self, device_id: str) -> Pose | None: ...


@runtime_checkable
class Mobile(Protocol):
    """Anything the brain (merging_map/cerveau.py, `mission`) can send somewhere.

    This is exactly `merging_map/robot.Robot`: the brain calls nothing else. The simplest way to
    implement it is to subclass `robot.RobotOdometrie` (position bookkeeping, buffering during a
    link loss, re-projection after a drone loss are done for you) and write: temps(),
    attendre(s), drone_disponible(), rover_joignable(), recaler(), deplacer(cible) and
    observer(tour_complet=False). See merging_map/rover_reel.py (real RaspRover) and
    merging_map/demo_fusion.py `Simu` (simulated) for two complete examples.

    Names are French because the brain is (Paco's code). Positions are (x, y) in the map frame.
      temps() -> float                      current time (s)
      attendre(duree)                       stand still for `duree` s
      drone_disponible() -> bool            is some drone / locator seeing the rover?
      rover_joignable() -> bool             does the link to the rover work?
      position() -> (x, y)                  estimated position
      incertitude() -> float                possible position error (m), 0 when located
      position_reelle() -> (x, y) | None    ground truth, simulation only
      placer(pos)                           start of mission: the rover is at `pos`
      recaler() -> float                    re-locate from the drone; returns the error (m)
      trace_depuis_recalage()               (last located point, positions since then)
      deplacer(cible) -> (x, y) | None      drive straight to `cible` (~30 cm); impact or None
      observer(tour_complet=False)          publish what the rover's camera sees (tiles)
      demander_zoom(tuile)                  ask the drones for a closer look at (z, x, y)
      signaler_choc(impact)                 mark a felt obstacle on the map
      garder_a_bord() / vider_tampon()      buffer observations during a rover link loss
      attendre_retour_embarque(chemin)      only if retour_embarque is True
    Attributes: portee_camera (m), retour_embarque (bool).
    """

    portee_camera: float
    retour_embarque: bool

    def temps(self) -> float: ...

    def attendre(self, duree: float) -> None: ...

    def drone_disponible(self) -> bool: ...

    def rover_joignable(self) -> bool: ...

    def position(self) -> tuple[float, float]: ...

    def incertitude(self) -> float: ...

    def position_reelle(self) -> tuple[float, float] | None: ...

    def placer(self, pos) -> None: ...

    def recaler(self) -> float: ...

    def trace_depuis_recalage(self): ...

    def deplacer(self, cible): ...

    def observer(self, tour_complet: bool = False) -> None: ...

    def demander_zoom(self, tuile) -> None: ...

    def signaler_choc(self, impact) -> None: ...

    def garder_a_bord(self) -> None: ...

    def vider_tampon(self) -> None: ...


def roles_of(device) -> list[str]:
    """Names of the roles a device object plays, from the methods it has."""
    out = []
    if callable(getattr(device, 'tiles', None)) and callable(getattr(device, 'available', None)):
        out.append('map')
    if callable(getattr(device, 'pose_of', None)):
        out.append('locator')
    if isinstance(device, Mobile):
        out.append('mobile')
    return out
