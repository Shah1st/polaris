# Local navigation with the rover's depth camera

Written 2026-11-16. Global navigation (overhead camera, `CarteFusion`, A*) decides the route;
this layer lets the rover see what is in front of it with its D405, stop by itself before an
obstacle the overhead camera missed, and drive without a stop at every step. Plain Python, no
ROS and no Nav2.

**Status: tests only.** Everything here runs against the real rover server with a simulated
ESP32 and a stand-in depth camera. Nothing has run on the real rover or on a real D405 picture
yet. All of it is **off by default**; the chain that works today is unchanged until the
switches below are turned on. "Before the first real run" lists what is still to do.

## How it works

```
ON THE ROVER (RDK X5)
  rover_mapper/d405_server.py                rover/rover_server.py (20 Hz loop)
    capture, 15 pictures a second              local_map.LocalMap: 2.5 x 2.5 m around the
    grid thread, 10 a second at most,          rover, 5 cm cells, odometry frame
    only while asked:                poll      Controller._local_guard, on every tick of a path:
      depth -> points above the  -- GET /grid -->  obstacle within 0.60 m on the path -> slower
      floor -> 5 cm cells                          obstacle within 0.10 m -> stop, status
      (-1 / 0 / 1)                                 "blocked" and where the obstacle is
                                               GET /local_grid, /state: local, ctrl.local
                    ---------------- Wi-Fi or tunnel ----------------
ON THE PC   sim/fusion_reelle.py --continu     about twice a second
  GET /state, GET /local_grid -> rover tiles -> CarteFusion
  decider() + A* -> stop one rover radius before unseen cells -> round the corners
  -> POST /path {"points", "id"} when the new path is better or the old one is barred;
     the rover takes it over at speed
  "blocked" -> obstacle marked on the map -> new route at once
```

The rover protects itself: the stop does not depend on the network. The planner only sends
paths. If the link is slow, the rover reaches the end of its path, which never enters cells
nobody has seen, and waits there.

## The parts

| Part | File | What it does |
|---|---|---|
| Depth to grid | `rover_mapper/d405_server.py`, `GridMaker` | `GET /grid`: 5 cm cells in front of the lens (x forward, y left), 1.25 m ahead, from the depth alone, with `rovermap/perception.py` (the geometry of the perception service's D405 mode). `POST /calibrate_floor` fits the floor once and keeps it in `d405_floor.json` |
| Rolling grid | `rover/local_map.py`, `LocalMap` | Places each grid with the pose the rover had when the picture was taken. A seen cell overwrites; an unseen one is forgotten after `local_hold` s, except within 0.45 m of the rover (the camera cannot see the floor under its nose) |
| Guard | `rover/rover_core.py`, `Controller._local_guard` | Looks `guard_slow_dist` along the path ahead, in a corridor `guard_half_width` wide each side. Caps the speed at `sqrt(2 * guard_decel * (distance - guard_stop_dist))`; ends the path as `blocked` at `guard_stop_dist`. Without a grid newer than `guard_max_age`: `guard_blind_speed` at most |
| Take-over | `rover/rover_core.py`, `Controller.run`, `_start` | A `POST /path` with an `id` that replaces a running path starts from the closest point of the new path, at the current speed, without the starting pivot |
| Rover grid on the maps | `merging_map/suivi_continu.py` `grille_locale`, `sim/fusion_reelle.py` `vue_locale` | `GET /local_grid` (odometry frame) turned into rover tiles; also while the rover drives |
| Path shaping | `merging_map/suivi_continu.py` | `lisser` cuts corners sharper than 40 degrees when the short cut is clear (the rover stops and pivots above 45); `a_remplacer` sends a new path only when it ends 0.2 m nearer the goal or is 0.2 m shorter; `prolonger` plans from a little ahead of the rover |
| Driver | `merging_map/avancer_esquive.py`, `RoverSegments.lancer`, `suivi`, `obstacle_vu`, `pivoter` | Non-blocking send and poll. Any path or stop that does not carry one of the mission's ids ends the mission as an operator stop |
| Mission loop | `sim/fusion_reelle.py`, `_boucle_continue` | The loop of the diagram. Looks around by pivoting the rover (+40, -80, +40 degrees) when nothing seen free is ahead |

## Switches and settings

Maps server (`sim/fusion_reelle.py`):

| Option | Default | Meaning |
|---|---|---|
| `--continu` | off | Drive without stopping (the loop above) |
| `--source-locale` | `service` | What feeds the local map: `service`, the perception service at each stop, as before; `rover`, the rover's own grid, also while driving |
| `--carte-locale` | off | As before. With `--continu` it also restricts driving to cells seen free |
| `--segment` | 0 | With `--continu`: longest path sent, m (2 m when 0) |
| `--periode-plan` | 0.5 s | Time between two plans |
| `--latence` | 0.4 s | The plan starts `vitesse * latence + 5 cm` ahead of the rover |
| `--duree-obstacle` | 30 s | Life of an obstacle reported by the guard when nothing sees it again |
| `--duree-max` | 600 s | Longest mission |

Rover settings (`POST /config` on the rover, saved in `rover_config.json`):

| Setting | Default | Meaning |
|---|---|---|
| `local_guard` | 0 | 1: the guard is on |
| `guard_half_width` | 0.12 m | Half the corridor checked (rover half width 0.10 + 2 cm; a 5 cm cell counts from its edge) |
| `guard_stop_dist` | 0.10 m | Path kept free before the obstacle |
| `guard_slow_dist` | 0.60 m | Path looked at |
| `guard_decel` | 0.3 m/s² | Braking the slow-down is computed for |
| `guard_blind_speed` | 0.08 m/s | Top speed without a recent grid |
| `guard_max_age` | 0.5 s | A grid older than this is not recent |
| `local_hold` | 10 s | Memory of an unseen cell |
| `d405_x`, `d405_y` | 0, 0 m | Lens ahead of, and left of, the rover's centre of rotation. **To measure** |

Capture server (`rover_mapper/d405_server.py`): `--grid-hz` 10 (0: no grid), `--cam-height`
0.15 m (lens centre above the floor, measured by the team), `--cam-height-tol` 0.03,
`--grid-stride` 2, `--grid-range` 1.25 m, `--floor-file`.

These values differ from the first plan (0.185 / 0.30 / 0.80 / 0.4): the corridor is the
rover's body, not the planner's keep-out, otherwise the guard would stop the rover on every
path the planner sends 15 cm from an obstacle.

## Interfaces added

- `d405_server`: `GET /grid` -> `{res, x0, y0, cells, floor, plane, seq, age_s, stats}`;
  `POST /calibrate_floor[?force=1]`; `GET /calibration`; `/health` has a `grid` block with
  `cpu_ms_per_grid`.
- `rover_server`: `GET /local_grid` -> `{res, x0, y0, cells, pose: [x, y, heading rad],
  age_s, frames, error}`, odometry frame, cell `[i][j]` starts at `(x0 + j*res, y0 + i*res)`.
  `/state` has `local` (`feed`, `error`, `grids`, `age_s`, `obstacle`, `free`) and
  `ctrl.local` (`state`: off, ok, slow, blind, blocked; `obstacle_dist`).
- `POST /path` takes `"id"` (1 to 40 characters), given back in `ctrl.current.id` and in the
  history entry; a replaced entry has `replaced_by`. New status `blocked`, with
  `obstacle: [x, y]` in the odometry frame.
- `status.json` of the maps server: `local` (`source`, `age_s`, `erreur`, `garde`,
  `obstacle_dist`) and `conduite` (`pas`, `segments`, `continu`).

## Running it

On the rover (the capture server needs `rover_mapper/rovermap/` next to it, numpy and OpenCV):

```bash
PYTHONPATH=~/polaris-d405/lib python3 d405_server.py --cam-height 0.15
curl -X POST http://127.0.0.1:8091/calibrate_floor      # rover on a flat, empty floor
curl -X POST http://127.0.0.1:8080/config -d '{"local_guard": 1, "d405_x": 0.0}'
```

`rover_server.py` needs `rover_core.py` and the new `local_map.py` from `main`.

On the PC, close to the rover on the network:

```bash
python sim/fusion_reelle.py --rover http://<rover>:8080 --port 8096 \
    --continu --carte-locale --source-locale rover --drone-source <overhead camera>
```

Then the dashboard as usual: click a goal, "Go". Each switch works alone: the guard with the
old step-by-step mission (a `blocked` step is not yet read by that loop: use `--continu`),
`--source-locale rover` without `--continu`, `--continu` without the local map.

## Before the first real run

1. **Compare the server on the rover with `rover/` on `main`**: `rover_core.py` changed here,
   and a note in the code says the robot's copy differs.
2. **Calibrate the floor** and read the result: `height_m` should be 0.15 within 3 cm, and
   `tilt_deg` is the lens's tilt, not known today.
3. **Measure `d405_x`, `d405_y`** with a ruler. With 0 the obstacles are placed too close to
   the rover's centre by that distance, so it stops later than set.
4. **Record real pictures** (`GET /frame`) on the floor: empty, a box at 0.3, 0.6 and 1 m, a
   chair leg, a low object, a wall, a shiny floor. Draw each grid over its picture
   (`rovermap.perception.draw`). The repo's lesson holds here: uneven light and wrong scale
   were found on photos, never by a test.
5. **Read `cpu_ms_per_grid`** in `/health` on the RDK X5 with the rover server running. Too
   heavy: `--grid-hz 5` or `--grid-stride 4`.
6. **Rover on blocks**, guard on: a box moved onto the path gives `slow` then `blocked`, and
   the wheels stop.
7. **On the floor, in this order**: the guard alone on a straight path towards a box;
   `--source-locale rover` (the box appears on the fused map); `--continu`.

## Known limits

- **Blind zone.** With the lens 15 cm up, the floor right in front of the rover is outside
  the picture (roughly the first 25 cm, depending on the tilt). What was seen there earlier
  is remembered; something put down inside it is not seen.
- **Low and high things.** An obstacle is a cell with points 4 to 40 cm above the floor.
  Lower things are floor; nothing checks what is above 40 cm.
- **Range.** 1.25 m ahead; the D405 is best under about 0.5 m, so far cells are noisier.
- **The guard only stops.** It does not steer round an obstacle; the planner does, on its
  next plan. Manual driving (`/cmd`) and `/move` are not guarded.
- **Odometry frame.** The grid and the maps of `fusion_reelle.py` live in the rover's
  odometry, which drifts; an obstacle seen long ago is where the odometry put it.
- **Obstacles that leave.** The rover forgets them after `local_hold`; the fused map keeps a
  rover tile until the rover sees that cell free again.
- **The link.** Through the VM's tunnel (1.3 to 4 s a request) the rover stays safe but ends
  each path before the next arrives. Smooth driving needs the planner on the room's Wi-Fi.
- **Two processes on the RDK X5**, which has no fan; the grid is computed only while asked.

## Tests

| File | Covers |
|---|---|
| `rover_mapper/test_d405_grid.py` | Grid from a synthetic floor 15 cm under the lens and a box; floor calibration kept, refused at a wrong height, still working facing a wall; `/grid` over HTTP with a fake pyrealsense2 |
| `rover/tests/test_local_map.py` | Placing a rotated grid without holes, memory and forgetting, first obstacle along a path |
| `rover/tests/test_path_guard.py` | Slow then `blocked` without contact, obstacle beside the corridor ignored, no pictures -> slow, guard off changes nothing; take-over at speed, bend followed as an arc, old behaviour without `id` |
| `rover/tests/test_server.py` | `/local_grid`, the feed, path ids over HTTP |
| `merging_map/test_suivi_continu.py` | Corner rounding, replacement rule, placing the rover grid |
| `merging_map/test_conduite_continue.py` | End to end against the real rover server (simulated ESP32, stand-in camera with a field of view and shadows): clear way driven without the speed dropping; a box only the rover sees driven round without contact; a wall across the zone ends `bloque` short of it. Real time, about 45 s |
| `sim/test_fusion_reelle.py` | Rover grid fused at the right place while moving; defaults off |

## Not done

- Dashboard: no display of the guard's state or a switch for `--continu`; `status.json`
  carries the fields.
- The rover's own web page has no box for `local_guard` (use `POST /config`).
- `merging_map/lancer_overhead.py` and the step and segment loops do not read `blocked`.
- Steering round an obstacle on the rover itself, and Nav2. A later Nav2 bridge would need
  the same rover interfaces (`/local_grid`, `id`, `blocked`).
