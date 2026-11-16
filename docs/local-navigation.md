# Local navigation with the rover's depth camera

Written 2026-11-16. Global navigation (overhead camera, `CarteFusion`, A*) decides the route;
this layer lets the rover see what is in front of it with its D405, stop by itself before an
obstacle the overhead camera missed, and drive without a stop at every step. Plain Python, no
ROS and no Nav2.

**Status: deployed, not driven.** The tests run against the real rover server with a
simulated ESP32 and a stand-in depth camera. Since 2026-11-16 the code is on the robot, the
floor is calibrated and `GET /local_grid` returns the real D405's grid ("On the robot,
2026-11-16" below), but the guard has never been on with the wheels turning and no path has
been driven with it. All of it is **off by default**; the chain that works today is unchanged
until the switches below are turned on. "Before the first real run" lists what is still to do.

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
| `--duree-obstacle` | 30 s | Life of an obstacle reported by the guard (`--continu` and `--segment`). In segments, a view of the perception service that says "free" there does not erase it before; the rover's own grid does |
| `--avance-camera` | 0 m | Lens ahead of the rover's centre, for the views of the perception service (`--source-locale service`). The rover's own grid has it already (`d405_x`) |
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
  `obstacle_dist`) and `conduite` (`pas`, `segments`, `continu`). `POST /conduite`
  `{"continu": bool, "source_locale": "rover"|"service"}` (either key) changes `--continu`
  and `--source-locale` while it runs; 409 during a mission.
- Dashboard (`teleop/server.py`): `POST /fusion/conduite` relays that; `POST /api/guard`
  `{"local_guard": bool}`; `/api/state` has `guard` (0, 1, `"absent"` on an older rover
  server), `robot.local` and `ctrl.local` from the rover's `/state`.

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

Then the dashboard as usual: click a goal, "Go".

**From the dashboard** (`teleop/`, "Drone and maps" tab), without restarting anything:

- **Drive without stops** and **Rover's depth grid**: the maps server's `--continu` and
  `--source-locale rover`, switched by its `POST /conduite` (refused during a mission).
  Hidden with the simulation's maps server.
- **Rover guard**: the rover's `local_guard` setting (`POST /api/guard` sends
  `POST /config {"local_guard": 1|0}`, nothing else). Hidden, and "not on this rover server"
  in the left table, when the rover's settings have no `local_guard` (an older
  `rover_server.py`).
- A line under the switches, and the "Rover guard" row on the left: the guard's state (way
  clear, slowing down, no recent depth picture, **STOPPED before an obstacle** in red) with the
  distance to the obstacle, the age and error of the rover's grid feed, where the rover map
  comes from, and how missions drive.
- The **speed slider** (keys) also sets the missions' speed once moved: the maps server's
  `POST /vitesse` (0.05 to 0.30 m/s; the slider stops at 0.20), from the next path sent to the
  rover, also during a mission.

Each switch works alone: the guard with missions in segments (`--segment`: a `blocked`
segment puts the obstacle on the map and a new route is planned; three refusals in a row
before the rover moves end the mission as blocked) but not yet with the step-by-step loop
(`--segment 0`), which does not read `blocked`; `--source-locale rover` without `--continu`;
`--continu` without the local map. With the guard on, prefer `--source-locale rover`: the
planner then sees what the guard sees. After a guard stop the rover stands about 10 cm
from the obstacle, which can still be inside the guard's corridor (12 cm each side) for any
new route: the mission then ends blocked and asks for the rover to be moved back by hand.
Backing off by itself, as the tilt guard does, is not done.

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

## On the robot, 2026-11-16

Steps 1, 2, 3 and 5 are done, 4 is started; 6 and 7 are not. The rover stood still on a flat
carpet the whole time; the guard stayed off.

- **Step 1, what the robot ran.** Not an older `main`: `rover_core.py` was the one of
  `experiment` (8a3846f) and `rover_server.py`, `web/index.html` those of `rover-teleop`
  (db24533): the command log (`GET /commands`, f4e92a6) and a speed carry-over for a path
  replacing a path (13 lines of 8a3846f), neither on `main`. `d405_server.py` was an older
  `main` (84e711b).
- **What it runs now.** `rover/`: `main` merged with `rover-teleop`, which is
  `git merge origin/rover-teleop` byte for byte (no conflict; 191 of the 192 rover tests pass
  on a Windows PC, the one that fails, the SIGTERM test, fails there on plain `main` too):
  the local navigation and the command log. The carry-over of `experiment` is gone: `main`'s take-over
  replaces it and needs a path `id`, so a guide that sends paths without one (`air_ground`)
  stops and pivots at each correction again. `~/polaris-d405`: `d405_server.py` and
  `rovermap/` of `main`. Backups on the robot: `~/rover/backup/0121-pre-localnav/`,
  `~/polaris-d405/backup/0114-pre-localnav/`. Before the swap, the merged server ran in
  `--sim` on the robot's Python 3.10 with the real D405 feeding its local map.
- **Step 2, the floor.** `POST /calibrate_floor`: `height_m` 0.1522 (0.15 expected),
  `tilt_deg` 1.46: the lens looks 1.4 degrees down and rolls 0.2 degrees. Kept in
  `~/polaris-d405/d405_floor.json`. Fitted picture by picture before that: 0.1533 m, 1.6 degrees.
- **Step 3, the lens offset.** Measured with a ruler: the lens is 9 cm ahead of the middle of
  the wheel axle and on the rover's centre line. Set on the robot (`POST /config`, kept in
  `~/rover/rover_config.json`): `d405_x` 0.09, `d405_y` 0.
- **Step 5, the cost.** RDK X5, 8 cores, rover server running, the grid thread at its 10
  grids a second for 20 s: `cpu_ms_per_grid` 56 to 58 ms, but the whole capture process took
  **2.3 cores** (8 % of one core at rest). `cpu_ms_per_grid` is the grid thread's own time;
  the rest is probably OpenCV and numpy working on other threads (OpenCV is set to 8 on this
  board), which it does not count: not checked. 62 to 64 degrees C during the measure. To
  try before driving, none measured yet: `--grid-hz 5`, `--grid-stride 4`, or one thread
  (`cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1`).
- **After the swap.** The rover server restarted, so its odometry is back at (0, 0, 0): a maps
  server started before must be reset (`POST /reset`, "Reset the maps") before a goal is
  clicked.
- **First missions with the guard on (segments, local map from the perception service).**
  The rover sometimes stood still while the mission went on. Its history said why: 39 of its
  last 50 paths ended `blocked` after 0 m, for the same obstacle of its own grid. The planner
  did not have that obstacle (it read the perception service, whose views are also drawn 9 cm
  too near without `--avance-camera`), the segments loop did not read `blocked`, and the same
  segment was sent again every 5 to 8 s, up to 23 times. Two identical `blocked` entries in
  `ctrl.history` then looked like a segment "disparu sans trace" and ended the mission on a
  false error. Fixed: the loop reads `blocked` (see "Running it"), each segment carries an
  `id`, and a mission that gets no view from the rover's camera now says that it ended.
  Reproduced on the bench before the fix (real rover server, simulated board: 12 times
  the same segment, the rover standing at the box); after it the rover goes round the
  box, or ends blocked when stopped too near to leave. Kept as tests with a stand-in
  rover (`sim/test_fusion_reelle.py`). On the real rover the next mission showed the new
  lines ("OBSTACLE seen by the rover ... new route", twice), then a second way to stand
  still: 15 cm segments to a degraded goal ended `done` after 0 m and 0 s, 17 times. The
  rover's path follower takes a path shorter than its look-ahead (0.2 m) whose end is
  behind its nose as "just passed" (`rover_core.py`, `PathFollower.update`). The segments
  loop now turns the rover to face such a segment, and ends the mission as blocked after
  three segments in a row that end `done` without moving. That part is tested with a
  stand-in rover only.
- **Step 4, pictures.** `rover_mapper/scripts/record_d405.py NAME` saves the frame, the
  rover's grid and a picture with the grid drawn over it, in `rover_mapper/recordings/`.
  Recorded: the empty carpet (floor free up to the range, no false obstacle; the grid
  recomputed from the frame and the rover's own agree within a few cells). Still to record:
  a box at 0.3, 0.6 and 1 m, a chair leg, a low object, a wall, a shiny floor.
- **On the real rover server:** `GET /local_grid` 0.03 s old, 441 free cells, no error;
  `local_guard` 0.

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
- **Obstacles that leave.** A cell the rover's camera sees free again is cleared at once, on
  the rover and on the fused map, with either source (`sim/test_fusion_reelle.py`). A cell it
  no longer sees is not: the rover forgets it after `local_hold`, the fused map keeps the
  rover tile until the rover sees that cell free. Three things do not clear by being seen:
  what the guard reported lasts `--duree-obstacle`; a bump (tilt, stall) is a forbidden zone
  until "Reset the maps", on purpose (the camera saw nothing where the rover tipped); and on
  the rover, an obstacle within 0.45 m is never forgotten and, inside the blind zone, cannot
  be seen free either: a box taken away right in front of a stopped rover stays in its memory
  until it has moved off, by another route or by hand.
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
| `sim/test_fusion_reelle.py` | Rover grid fused at the right place while moving; defaults off; an obstacle seen free again leaves the map with both sources, one no longer seen stays; the guard's obstacle expires and is not erased before; in segments a `blocked` segment changes the route, three before moving end the mission |

## Not done

- The rover's own web page has no box for `local_guard` (use `POST /config`, or the
  dashboard's "Rover guard" box).
- `merging_map/lancer_overhead.py` and the step-by-step loop do not read `blocked` (the
  segments and continuous loops do).
- Forgetting, on the rover, a near obstacle that was taken away (see "Obstacles that leave").
- Backing off after a guard stop, so that a new route can leave (see "Running it").
- On the rover, a short path whose end is behind the nose ends `done` at once
  (`PathFollower.update`: the "just past it" rule also holds before the rover has moved).
  The segments loop works round it by pivoting first; the continuous loop does not.
- Steering round an obstacle on the rover itself, and Nav2. A later Nav2 bridge would need
  the same rover interfaces (`/local_grid`, `id`, `blocked`).
