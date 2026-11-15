# devices: plug any drone, rover or sensor into Polaris

## The idea

1. One map (`merging_map.CarteFusion`) takes tiles of `-1 / 0 / 1` from anyone.
2. A device plays one or more of three roles: it **makes map tiles**, it **says where a device
   is**, or it **can be sent somewhere**.
3. A JSON file lists the devices; `build()` plugs them all around that one map.
4. Each device's tiles carry its name: source `drone:<id>` or `rover:<id>`.
5. The brain (`cerveau.mission`) is unchanged: it drives one rover through the same interface,
   in simulation or for real.

```bash
python devices/examples/run_fleet_sim.py --drive     # 2 drones + 2 rovers, no hardware
python -m pytest devices -q -p no:cacheprovider      # 27 tests
```

## The three roles (`polaris_devices/roles.py`: the docstrings are the specification)

| Role | Methods | Used by | Today |
|---|---|---|---|
| `MapSource` | `tiles()`, `available()`, optional `zoom(tile)` | first survey, `DEMANDER_DRONE` zooms | overhead camera, simulated drones, any `tile-http` device |
| `Locator` | `pose_of(device_id) -> (x, y, heading or None)`, optional `refresh()` | the rover adapter's `position_drone` | overhead camera, simulated drones |
| `Mobile` | exactly `merging_map/robot.Robot` (`deplacer`, `observer`, `position`...) | `cerveau.mission` | `RaspRoverReel`, `Simu` |

All positions are in the map frame (metres; today the overhead camera's: origin at the image's
bottom-left corner, x right, y up). Tile format: `merging_map/README.md`. **A cell you did not
really see is `-1`, never `0`**: a `0` erases the obstacles other devices saw.

## Configuration

```json
{"map": {"niveau_plan": 4, "rayon_robot": 0.155},
 "devices": [
   {"id": "overhead", "role": "drone", "kind": "overhead-camera", "source": "0",
    "settings": {"tracks": "rasprover",
                 "overhead": {"metres_per_pixel": 0.0030, "marker_height": 0.0,
                              "rover_marker_id": 10}}},
   {"id": "rasprover-eye", "role": "rover", "kind": "rover-perception",
    "address": "http://127.0.0.1:8095"},
   {"id": "rasprover", "role": "rover", "kind": "rasprover", "address": "http://ROVER-IP:8080",
    "settings": {"locator": "overhead", "perception": "rasprover-eye"}}]}
```

* `map`: arguments of `CarteFusion` (the arena: `niveau_plan` 4, `rayon_robot` 0.155).
* `id`: unique name. `role`: `drone` (seen from above: the global map) or `rover` (on the
  ground: what it saw wins over the drone where it looked). `kind`: the adapter.
* `address` / `source`: where the device is (URL, camera index, file). `settings`: per kind.
* Declare locators and sensors **before** the rover that uses them.
* A kind whose module is missing (OpenCV, a package not in this tree) is skipped with a
  message, the others are plugged (`fleet.skipped`); `build(..., strict=True)` raises instead.

```python
from polaris_devices import build
fleet = build('devices/examples/fleet_sim.json')
fleet.survey()                                   # every map source -> the one map
fleet.plan('rover-a', victim=(8.0, 20.0))        # CarteFusion.decider for that rover
fleet.run_mission('rover-a', victim=(8.0, 20.0)) # cerveau.mission, settings=... as usual
```
`python -m polaris_devices CONFIG.json --victim X Y` (from `devices/`) builds, surveys and
plans for every rover.

| kind | role | plays | settings |
|---|---|---|---|
| `overhead-camera` | drone | map, locator | `source` (camera index, URL or image file), `tracks` (rover id), `overhead` (`OverheadSettings`; a live camera is refused without a measured `metres_per_pixel`, as `lancer_overhead.py --camera`), `max_age`, `period` |
| `rasprover` | rover | mobile | `address` (rover_server.py), `locator`, `perception`, `timeout`, `options` (`RaspRoverReel` arguments) |
| `rover-perception` | rover | sensor of a rover | `address` (perception service), `timeout` |
| `sim-drone` | drone | map, locator | `world` (`monde_1`...), `area` [x0, y0, x1, y1], `level`; or `address` of the MuJoCo sim (`sim/run_world.py`) and `tracks` |
| `sim-rover` | rover | mobile | `world`, `start` [x, y], `champ`, `occlusion`, `coupures` |
| `tile-http` | any | map (+ locator with `locates: true`) | `address`, `timeout`, `period` |

## Add a device in ten minutes

**Way 1: serve tiles over HTTP (no code in this repository).** Any program, any language:

```
GET /map  -> 200 {"t": 1731681234.5, "res": 0.05, "x0": 1.20, "y0": 0.40,
                  "cells": [[-1, 0, 1, ...], ...]}      or 503 when it has nothing fresh
GET /pose -> 200 {"rasprover": [x, y, heading_rad_or_null]}   (optional, "locates": true)
```

`cells[i][j]` is the cell at x = x0 + j·res, y = y0 + i·res (row 0 = south), in the map frame.
`res` is 0.05 m times a power of two (0.05, 0.1, 0.2, 0.4, 0.8); x0, y0 are multiples of
`res`. A device that already cuts tiles may answer `{"t": ..., "tiles": [{"z", "x", "y",
"data"}]}`. `t` is optional and ignored: the fleet stamps every tile with the time it is
received (`time.monotonic()` on the fusion's machine, the clock of every other adapter and of
the rover), so a device's clock or a Unix time cannot make an old tile win over a fresh one.
Your own adapter (way 2) must stamp its tiles with `time.monotonic()` too. A complete device
in 40 lines: `examples/tile_device.py`. Declare it:
`{"id": "wall-cam", "role": "drone", "kind": "tile-http", "address": "http://127.0.0.1:8178"}`.

**Way 2: write an adapter class.** Implement the role's methods (`roles.py`), then register a
factory before `build()`:

```python
from polaris_devices import register

class Lidar:                                   # a MapSource
    def __init__(self, port): ...
    def available(self): return self.connected
    def tiles(self):                           # messages; 'source' is set by the fleet
        return decouper(self.grid, self.x0, self.y0, 4, carte, '', t=time.monotonic())

register('lidar', lambda spec, fleet: Lidar(spec.address))
```

Import heavy modules inside the factory: if they are missing, only that kind is disabled. For
a new rover, subclass `merging_map/robot.RobotOdometrie` (see `rover_reel.py`), give it
`fleet.map_for(spec)` as its map and `fleet.drone_view()` as its drone.

## What works today, what does not

Works (tested here without hardware, `tests/`): the JSON fleet; two simulated drones on two
halves plus two simulated rovers feeding one map, a plan for each rover and both missions
driven one after the other; the example HTTP device; a mock of the MuJoCo simulator's API; the
overhead camera on a still image; the RaspRover adapter built without contacting the robot.

Not yet:
* **Never run on the real robot or the live camera.** The tested way to run a real mission is
  still `merging_map/lancer_overhead.py`; `fleet_arena.json` is the same wiring, untried.
* Devices of one role are layered by time: where two drones (or two rovers) know the same
  tile, the most recently received one wins (sources `drone:<id>` / `rover:<id>`; with an
  older fusion that refuses ids, the fleet falls back to plain `drone` / `rover`).
* **One rover per mission.** The brain drives one rover at a time; rovers do not see each other
  as obstacles. A fleet moving together needs cell reservation along each path, priorities at
  crossings, and each rover's position published as a moving obstacle for the others.
* One overhead camera locates one rover (`tracks`); two rovers need two markers.
* `rover-perception` uses `merging_map/perception_rover.py` and a running
  `rover_mapper/perception_service.py` (port 8095 by default). Without a perception the
  RaspRover is still plugged and drives on the drone's map alone (as `lancer_overhead.py`
  without `--perception`), with a message saying so.
* `sim-rover` is merging_map's simulated rover; the MuJoCo rover is only seen, not driven.
