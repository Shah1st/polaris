# merging_map: drone + rover map fusion

Merges the obstacle tiles sent by the drone and the rover into one map, plans a path with A*
and decides what to do when the rover is blocked.

| File | What it is |
|---|---|
| `fusion_carte.py` | The map: tile storage, fusion, A*, blocking decisions |
| `cerveau.py` | The brain: the mission loop (decide, drive, look, handle link losses) |
| `robot.py` | The interface between the brain and a rover, plus shared odometry bookkeeping |
| `demo_fusion.py` | Simulated world and `Simu`, the simulated robot (needs `numpy` and `matplotlib`) |
| `rover_reel.py` | `RaspRoverReel`, the real RaspRover through its on-board server (**not yet tested on the robot**) |
| `demo_fusion.png` | Output of the demo |
| `test_*.py` | pytest tests (map, simulation, real-rover adapter against a fake rover server) |
| `conftest.py` | Lets pytest import the modules from this folder |
| `dashboard/` | Replay dashboard (see below) |

## Architecture: one brain, two robots

```
                 cerveau.mission(robot, carte, ...)        <- runs on the ground computer
                         |   robot.Robot interface
           +-------------+--------------+
           |                            |
   demo_fusion.Simu              rover_reel.RaspRoverReel
   (simulated world,             (steps sent to the rover's own server, which drives:
    tests, dashboard)             rover/rover_server.py, branch rover-teleop)
```

The brain only uses the `robot.Robot` interface: `position()`, `incertitude()`,
`deplacer(cible)`, `observer()`, `demander_zoom(tuile)`, `drone_disponible()`,
`rover_joignable()`, `attendre()`, `recaler()`… The same mission code therefore drives the
simulation and the real robot.

```python
from cerveau import mission
from fusion_carte import CarteFusion
from rover_reel import RaspRoverReel, client_http

carte = CarteFusion()
rover = RaspRoverReel(
    carte,
    client_http('http://<rover IP>:8080'),  # rover/rover_server.py on the rover (RDK X5)
    position_drone=...,  # -> (x, y, heading or None) of the rover seen from above, or None
    percevoir=...,  # -> (matrix -1/0/1, x0, y0): what the rover camera sees
    drone=...,  # object with disponible() and zoomer(tile) -> list of tiles
)
mission(rover, carte, depart=(3.0, 3.0), victime=(8.0, 20.0))
```

**Who does what.** The brain decides where to go; the rover's own server (RBDRB's
`rover/rover_server.py`, branch `rover-teleop`) decides how. Each step of the path (about
30 cm) goes to the server with `POST /path`, in the frame of its odometry (wheels + gyro);
the server follows it and stops. It needs the server of the branch `rover-teleop`:
`POST /path` came with commit 5a7277d and is not in `main` yet (the adapter says so if it
gets a 404).

* **Two frames.** The adapter keeps the transform map <- odometry (rotation + translation).
  It is corrected only when the rover has **stopped**, on an image taken after the stop;
  while the rover moves, the position comes from the odometry. A slow camera therefore no
  longer makes false bumps.
* **Heading.** From the camera if it gives one; otherwise measured from a motion seen by both
  the camera and the odometry (a 15 cm straight line before the first step, then refined at
  every stop).
* **Bumps.** `stalled` from the server (wheels not turning), or wheels that spun while the
  camera saw the rover stay in place.
* **Links.** Only a missing answer means "rover unreachable". An HTTP error (400, 404) raises
  `ServeurRoverError` instead of starting the recovery. When the link is lost the server
  finishes its current step and stops: there is no on-board return program, the brain waits
  for the link to come back.
* **Measured** against the real server in simulation (`rover_server.py --sim`, slope drift
  0.1 rad/s), with the camera in a frame rotated by 30 degrees, one image a second, 1 cm
  noise and no heading: arrived, no false bump, 1 to 3 cm position error at each stop, 1 to
  4 cm drift during a step. The previous adapter, which trusted every image, made 4 false
  bumps and 35 to 54 cm of error with the same slow camera (in its own simulation).
* Not yet run on the real robot.

## With the overhead camera (branch `overhead-camera`)

The overhead camera pipeline (`guidance_global_map.drone.OverheadDrone`) plays the drone.
`lancer_overhead.py` wires it to the rover and runs a mission:

```bash
PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose \
    python merging_map/lancer_overhead.py --image overhead.jpg --victime 1.2 0.9 \
    --rover http://<rover IP>:8080
```

* **The adapter is used as asked**: first survey from `camera.tiles()`, rover position from
  `camera.rover_pose` (with or without a heading), `drone=camera`, `zone=camera.zone()`.
  The start defaults to where the camera sees the rover.
* **Small-area settings** (`REGLAGES_PETITE_ZONE`): `CarteFusion(niveau_plan=4,
  rayon_robot=0.155)`, i.e. 5 cm planning cells, and `tolerance=0.15`. `rayon_robot` is the
  radius of the circle around the whole rover: hypot(0.20, 0.23) / 2 = 0.152 m for the
  0.20 x 0.23 m rover, rounded up. The 0.115 in the overhead camera's doc is the half-length,
  which let a corner of the rover hit an obstacle when turning on the spot.
  Provisional until a test drive gives real numbers.
* **Frame**: the camera's, origin at the image's bottom-left corner, x right, y up, metres.
  Start and goal outside the camera's view are refused with a message saying so; tiles from
  the rover camera must use the same frame.
* Needs the `overhead-camera` modules on `PYTHONPATH` and `opencv-contrib-python`. Tested here
  only with a fake camera that has the same methods (`test_lancer_overhead.py`): the real
  `OverheadDrone` has not been run with it yet.
* `--image` is a still photo, read again for every image: it does not see the rover move, so
  it is only good to try the chain. A real mission needs the live camera.

**ROS node** (`src/guidance_supervisor`, `merging_map_node`): subscribes to `/map` and
`/rover/pose_overhead` (and `/goal_pose`, or the `goal_x` / `goal_y` parameters), runs the
fusion's planner about once a second and publishes `/fusion/map` (merged `OccupancyGrid`),
`/fusion/plan` (`nav_msgs/Path`) and `/fusion/decision` (JSON). It plans; it does not drive
the rover. The conversion logic is in `fusion_bridge.py` (no ROS, tested with plain pytest);
the node finds `merging_map/` by itself in the repo, or through `MERGING_MAP_PATH`.

```bash
ros2 run guidance_supervisor merging_map_node --ros-args -p goal_x:=1.2 -p goal_y:=0.9
```

The node itself has not been run: it needs the ROS container.

```bash
cd merging_map
python demo_fusion.py
```

The simulated rover camera:

* sees up to 2 m, in a cone of `champ` degrees centred on the rover's heading
  (`Simu(..., champ=160)`; 360 = all around, e.g. by turning the pan-tilt);
* **cannot see behind obstacles** (ray casting, first hit on each ray): hidden cells are sent
  as `-1`, exactly as the real rover must do. `occlusion=False` brings back the unrealistic
  "X-ray" camera for comparison;
* if the rover bumps into an obstacle it did not see, it stops 5 cm before it, reports it as
  an obstacle (as the real robot would from wheel slip on the encoders) and replans.

Three simulated terrains: `monde_1()` ridge with a hidden log, `monde_2()` closed crater,
`monde_3()` forest where the drone sees nothing under the trees.

A replay dashboard (drone + rover = merged maps, decisions step by step, field-of-view slider
with a comparison table) is in `dashboard/`:

```bash
python merging_map/dashboard/export_trace.py      # re-run after changing the module
python -m http.server 8765 --directory merging_map/dashboard
```

![demo](demo_fusion.png)

## Instructions for the rover team: tile format

* **N×N** matrix (N = 32), finest cell = **5 cm** (level z = 4).
* Values:
   * `1` = obstacle
   * `0` = free, **actually seen**
   * `-1` = not seen
* ⚠️ **"not seen" ≠ "free"**. Use `-1` for cells:
   * **behind any detected obstacle** (the camera can't see through it);
   * **beyond the camera's reliable range** (about 2 m);
   * anywhere you didn't look.

  If a hidden cell is sent as `0`, it **erases obstacles detected by the drone**.
* Tiles must be **aligned to the global grid** (drone / ArUco frame), not to the robot's frame.
* **Row 0 = south** (not the top, as in an image). Row i → +Y, column j → +X.
* Message:

  ```json
  {"source": "rover", "t": <seconds>, "z": 4, "x": <tx>, "y": <ty>, "data": [[...]]}
  ```

  Tile `(z, x, y)` covers `x*T <= X < (x+1)*T` and `y*T <= Y < (y+1)*T`, with
  `T = 32 × cell size` (1.6 m at z = 4).
* To split a large matrix into tiles, use `decouper()` in `fusion_carte.py`:

  ```python
  from fusion_carte import CarteFusion, decouper

  carte = CarteFusion()   # N = 32, finest cell 5 cm, zmax = 4
  # local: numpy matrix of -1/0/1 at 5 cm, whose cell [0][0] is at (x0, y0) in the global frame
  for msg in decouper(local, x0, y0, 4, carte, 'rover', t=now):
      send(msg)           # or carte.recevoir(msg) when testing locally
  ```
* For the pan-tilt camera, first send `{"T":900,"main":1,"module":2}` to the ESP32. The OLED
  should then show "RaspRover PT".

## How the fusion works

* The drone and rover tiles are stored separately: nothing the drone sends is ever deleted.
* In the merged view, a cell **seen by the rover** (`0` or `1`) wins over the drone; elsewhere the
  drone value stands. `-1` never changes anything.
* Within one source, the finest level and the most recent tile win.

## Using the module

```python
from fusion_carte import CarteFusion

carte = CarteFusion(n=32, res_fine=0.05, zmax=4)
carte.recevoir(msg)            # every tile received from the drone or the rover
carte.signaler_obstacle(x, y)  # wheel slip / stuck / slope too steep, felt by the rover

dec = carte.decider(rover_position, victim_position)
# dec['action'] is one of:
#   'SUIVRE'           -> dec['chemin']: list of (x, y) in metres
#   'DEMANDER_DRONE'   -> dec['tuiles']: [(z, x, y), ...] for the drone to fly over lower
#   'OBJECTIF_DEGRADE' -> dec['point']: closest reachable point to the victim, plus dec['chemin']

# after each step: does the next 3 m of the path still avoid every known obstacle?
if not carte.chemin_valide(remaining_path, horizon=3.0):
    dec = carte.decider(rover_position, victim_position)
```

Planning runs where the drone has surveyed (`carte.zone_survolee()`), stretched to include the
rover and the victim, so the rover does not wander off the map. Pass `zone=(x0, y0, x1, y1)` to
`decider()` to choose another area.

## Path planning and scales

* **A\*** on the z3 grid (10 cm cells), 8 directions (straight 1, diagonal 1.414, no corner
  cutting), straight-line heuristic, so the path is the shortest one. Unknown cells cost 3×,
  obstacles are grown by the robot radius (15 cm). Arrival = any free cell within 50 cm of
  the victim.
* Tile levels, 32×32 cells each: z0 = 80 cm cells (drone, first survey), z2 = 20 cm (drone
  zoom), z3 = 10 cm (planning), z4 = 5 cm (rover). One first-survey drone cell covers
  16 × 16 rover cells.
* Typical timings on a laptop (25.6 m area): one A\* ≈ 0.12 s, a full decision ≈ 0.16 s,
  the path check after each step ≈ 1 ms. Expect a few times slower on a Raspberry Pi.

## When the drone link is lost

Everything runs on the ground computer, which sends orders to the drone and to the rover. If
the computer stops hearing from the drone (`decider(..., drone_disponible=False)`):

* **The rover keeps going, more carefully**: half speed, the drone map frozen as it was, its
  position from odometry only (wheels + IMU), with an uncertainty that grows with distance
  (2.5 cm per metre in the simulation). Drone obstacles get that much extra margin, except
  where the rover has looked closely itself (`carte.marge_drone`).
* **It does the drone's reconnaissance itself** (`action == 'INSPECTER'`): when no path exists,
  the module asks where a passage could hide under an obstacle only seen from far away, and
  sends the rover to look from about 1 m away (`cible`, `point_de_vue`). A 360° look with the
  pan-tilt follows; the sector is then marked inspected and the search moves on.
* **Too uncertain** (over 60 cm): if the victim is closer than the last point seen by the
  drone, it carries on and stays with the victim; otherwise it goes back along its own trace
  and waits for the drone. Without the drone, "no passage" is never final: it waits for it.
* **When the drone comes back**, it sees the rover's marker: the position is corrected, and
  everything the rover saw meanwhile is rewritten at the right place (error spread along the
  path), which removes the "ghost" obstacles written at the wrong position
  (`instantane_rover()` / `restaurer_rover()`).

In the simulation: `Simu(..., coupures=[(t0, t1)])` cuts the link (`t1=None`: for good);
`monde_4()` and `monde_5()` are the drone-loss scenarios.

## When the rover link is lost

If the computer can no longer talk to the rover (a radio dead zone), nobody sends it orders
any more, so the way back has to be done **by the rover itself**: a small safety program on
the rover's computer (RDK X5), not the whole module. The drone keeps seeing the rover meanwhile, so the computer
still knows where it is.

1. No order from the computer for 1 s: the rover records the exact moment and the **last point
   where the link still worked**, stops, and keeps what its camera sees on board.
2. It drives back along its own trace to 50 cm before that point; the link comes back.
3. It sends what it saw during the cut. The computer marks the area where the link was lost
   as a **no-go zone** (`carte.interdire_zone(x, y, r)`: a circle of 1.5 m just past that
   point). The A\* treats it like an obstacle, but it stays out of the obstacle tiles, so the
   drone does not try to zoom on it and it can be removed later (radio relay).
4. The module plans again with the global map and the drone. If the new path enters the dead
   zone somewhere else, the same thing happens and the zone is mapped bit by bit.

In the simulation: `Simu(..., zones_blanches=[(x, y, r)])`; `monde_6()` is the scenario.

**Still to do on the real robot:** the on-board safety program (a watchdog in the rover's
server, `rover/rover_server.py`: record the odometry trail and the link state, and drive back
on it when the computer goes silent). Today that server finishes its current step and stops,
and the brain waits for the link to come back (`attendre_retour_embarque`).

When no path exists, the module asks: "if the obstacles the drone only saw from far away were
actually passable, where would I go?" It then asks the drone to zoom on exactly those tiles. If
the drone confirms they are blocked, the rover goes to the closest reachable point and raises
the alert.

## Tests and lint

```bash
pytest merging_map                     # numpy only; matplotlib is needed only for the figure
PYTHONPATH=src/guidance_supervisor pytest src/guidance_supervisor/test/test_fusion_bridge.py
ruff check . && ruff format --check .
```

Outside the ROS container, run `pytest merging_map` rather than a bare `pytest` at the root,
which stops on the ROS packages.

Code comments and identifiers are in French.
