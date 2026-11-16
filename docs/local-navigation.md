# Local navigation with the rover's depth camera

Written 2026-11-16. Global navigation (overhead camera, `CarteFusion`, A*) decides the route;
this layer lets the rover see what is in front of it with its D405, stop by itself before an
obstacle the overhead camera missed, and drive without a stop at every step. Plain Python, no
ROS and no Nav2.

**Status: reworked on 2026-11-17 after the first drives; the rework is tests only.** The
first version was deployed and driven on 16 Nov with the guard and the rover's grid both on,
and was overprotective: the rover stood still in front of obstacles that were not there, and
the two maps together were worse than either alone ("What the first drives showed" below).
The rework below has not been on the robot. Everything is still **off by default**.

**Deploying the rework: the robot's saved settings keep the old values.**
`rover_config.json` on the robot holds every setting, so the new defaults do not apply by
themselves. After copying `rover/`, send once:

```bash
curl -X POST http://127.0.0.1:8080/config -d '{"guard_half_width": 0.10,
  "guard_stop_dist": 0.15, "guard_decel": 0.2, "guard_max_age": 1.0}'
```

## The idea

**Global decides the route. Local only protects, for a short time and a short way.**

- The drone map is the map. The rover's grid is a layer on top of it, near the rover only,
  rebuilt from what the rover knows at every plan and never stored.
- The rover's grid adds obstacles; it never erases the drone's.
- An obstacle counts once two pictures out of three have shown it.
- The guard's corridor is the rover's body, 5 cm inside the planner's keep-out.
- Everything local is forgotten by the clock, whether or not pictures still arrive.
- When one of the two maps is missing, the other is used alone, and the dashboard says so.

## How it works

```
ON THE ROVER (RDK X5)
  rover_mapper/d405_server.py                rover/rover_server.py (20 Hz loop)
    capture, 15 pictures a second              local_map.LocalMap: 2.5 x 2.5 m around the
    grid thread, 10 a second at most,          rover, 5 cm cells, odometry frame; a cell is an
    only while asked:                poll      obstacle after 2 pictures of 3, forgotten 10 s
      depth -> points above the  -- GET /grid -->  after its last look
      floor -> 5 cm cells                      Controller._local_guard, on every tick of a path:
      (-1 / 0 / 1), obstacles                    obstacle in the body's way within 0.60 m -> slower
      up to 0.8 m                                within 0.15 m -> stop, "blocked", back 10 cm
                                                 no picture for 1 s -> slow; for 3 s more ->
                                                 normal speed, UNPROTECTED, and said so
                                               GET /local_grid, /state: local, ctrl.local
                    ---------------- Wi-Fi or tunnel ----------------
ON THE PC   sim/fusion_reelle.py --continu     about twice a second
  GET /state, GET /local_grid
  which maps work? -> superposition | global | local          (status.json: fusion)
  rover layer rebuilt: cells within 1.5 m, without what the drone already has within 15 cm
  decider() + A* -> round the corners
  -> POST /path {"points", "id"} when the new path is better or the old one is barred;
     the rover takes it over at speed
  "blocked" -> one cell on the map for 10 s -> new route at once
```

The rover protects itself: the stop does not depend on the network. The planner only sends
paths. If the link is slow, the rover reaches the end of its path and waits there.

## The two maps together (`--fusion`)

| Asked | Drone map | Rover grid | Used | What the planner does |
|---|---|---|---|---|
| superposition (default) | placed | fresh | superposition | Drone map, plus the rover's obstacles near the rover |
| superposition | placed | missing | global | Drone map alone, as before local navigation |
| superposition | never placed | fresh | local | Rover's grid alone, only through cells seen free, looking around when needed |
| any | never placed | missing | aucune | With the rover's grid as source: the mission does not start, or ends, and says why |
| global | placed | any | global | Drone map alone |
| local | any | fresh | local | Rover's grid alone |
| local | placed | missing | global | Drone map instead, with the reason shown |

- **Rover grid missing**: no answer, an error, or older than `--age-max`, for 3 s in a row;
  back after two good grids. One late picture is not an outage. While it is missing its
  layer is empty, whether that lasts ten seconds or ten minutes.
- **Drone map**: the last placed picture stays valid when the camera stops sending. Only
  "never placed" counts as missing.
- **Superposition, in detail** (`FusionReelle.superposer`, `CarteFusion.grille` with
  `rover_efface_drone` False):
  - only rover cells within 1.5 m of the rover;
  - a rover obstacle within 15 cm of a drone obstacle is the same object and is left out
    (the rover's frame is its odometry, the drone's is the camera's: the same box lands a few
    centimetres apart, and twice on the map it closed the passage beside it);
  - a rover "free" fills what the drone has not seen and never replaces a drone obstacle.
- **What the rover bumped into** (stall, tilt) and the views of the perception service
  (`--source-locale service`) stay in the old `rover` source and are kept as before; with
  the local map on they too add to the drone map instead of replacing it.
- With the local map off nothing changed: the drone map, and the old rule.
- `status.json` `fusion`: `demandee`, `utilisee`, `raison`, `cellules_rover` (rover obstacle
  cells in the plan), `doublons` (left out as the drone's), `arrets_garde` (guard stops since
  the maps were reset). The dashboard shows them under the switches.

## The parts

| Part | File | What it does |
|---|---|---|
| Depth to grid | `rover_mapper/d405_server.py`, `GridMaker` | `GET /grid`: 5 cm cells in front of the lens (x forward, y left), floor seen to 1.25 m, obstacles reported to 0.8 m, with `rovermap/perception.py`. `POST /calibrate_floor` fits the floor once and keeps it in `d405_floor.json` |
| Rolling grid | `rover/local_map.py`, `LocalMap` | Places each grid with the pose the rover had when the picture was taken. Confirmation (`local_confirm`) and forgetting by the clock (`local_hold`, `local_hold_near`) |
| Guard | `rover/rover_core.py`, `Controller._local_guard` | Looks `guard_slow_dist` along the path ahead, `guard_half_width` each side to a cell's centre. Slows down, ends the path as `blocked` after two control ticks inside `guard_stop_dist`, then backs off `guard_backoff`. Not while pivoting on the spot |
| Take-over | `rover/rover_core.py`, `Controller.run`, `_start` | A `POST /path` with an `id` that replaces a running path starts from the closest point of the new path, at the current speed, without the starting pivot |
| Layers | `sim/fusion_reelle.py` `regler_fusion`, `superposer`; `merging_map/fusion_carte.py` `grille`, `couche` | Which maps are used, and the rover's layer (source `rover:d405`) rebuilt at every plan |
| Path shaping | `merging_map/suivi_continu.py` | `lisser` cuts corners sharper than 40 degrees when the short cut is clear; `a_remplacer` sends a new path only when it ends 0.2 m nearer the goal, is 0.2 m shorter, or reaches the goal; `prolonger` plans from a little ahead of the rover |
| Driver | `merging_map/avancer_esquive.py`, `RoverSegments.lancer`, `suivi`, `obstacle_vu`, `pivoter` | Non-blocking send and poll. Any path or stop that does not carry one of the mission's ids ends the mission as an operator stop |
| Mission loop | `sim/fusion_reelle.py`, `_boucle_continue` | The loop of the diagram. Looks around by pivoting the rover (+40, -80, +40 degrees) when nothing seen free is ahead |

## Which frame the floor calibration uses

- `POST /calibrate_floor` fits the plane in the **D405 depth camera's own optical frame**
  (not the colour camera, nothing aligned): origin at the depth lens, x right, y down, z
  forward, metres. `d405_floor.json` keeps `normal` (unit vector from the floor towards the
  lens) and `d` (lens to floor: 0.1522 m on the robot); `tilt_deg` (1.46) is the angle
  between that normal and the camera's "up".
- The grid is in the frame built from that plane (`perception.rover_axes`): origin on the
  floor under the lens, x = optical axis projected on the floor, y left. The mount's pitch
  and roll are absorbed.
- The rover (`Rover.feed_grid`) moves it to its centre of rotation with `d405_x` (0.09 on
  the robot) and `d405_y`, and turns it by the rover's heading plus `d405_yaw_deg`. The
  lens's yaw on the chassis is **not calibrated** (0 assumed): 3 degrees put an obstacle at
  1 m 5 cm to the side. To measure it: a straight edge on the floor along the rover's axis
  should come out along x in `GET /grid`.

## Switches and settings

Maps server (`sim/fusion_reelle.py`):

| Option | Default | Meaning |
|---|---|---|
| `--continu` | off | Drive without stopping (the loop above) |
| `--source-locale` | `service` | What feeds the local map: `service`, the perception service at each stop, as before; `rover`, the rover's own grid, also while driving |
| `--carte-locale` | off | As before |
| `--fusion` | `superposition` | With the local map on: `superposition`, `global`, `local` (table above). Also `POST /conduite {"fusion": ...}` |
| `--segment-vu` | off | Drive only through cells seen free. Also applied by itself when the rover's grid is the only map |
| `--segment` | 0 | With `--continu`: longest path sent, m (2 m when 0) |
| `--periode-plan` | 0.5 s | Time between two plans |
| `--latence` | 0.4 s | The plan starts `vitesse * latence + 5 cm` ahead of the rover |
| `--duree-obstacle` | 10 s | Life of what the guard stopped for, on the map: one cell (was a 20 cm disc for 30 s). With the perception service as source, a view that says "free" there does not erase it before |
| `--avance-camera` | 0 m | Lens ahead of the rover's centre, for the views of the perception service. The rover's own grid has it already (`d405_x`) |
| `--duree-max` | 600 s | Longest mission |

Rover settings (`POST /config` on the rover, saved in `rover_config.json`):

| Setting | Default | Meaning |
|---|---|---|
| `local_guard` | 0 | 1: the guard is on |
| `guard_half_width` | 0.10 m | Half the corridor checked along the path, to the centre of an obstacle cell: the rover's body. The planner keeps 0.15 m (was 0.12 + half a cell = 0.145: a path the planner sent tripped it) |
| `guard_stop_dist` | 0.15 m | Path between the rover's centre and the first point of it that touches an obstacle: closer, stop. Covers the nose (11.5 cm) and the braking (was 0.10) |
| `guard_slow_dist` | 0.60 m | Path looked at |
| `guard_decel` | 0.2 m/s² | Braking the slow-down is computed for (was 0.3) |
| `guard_backoff` | 0.10 m | Driven back after a `blocked` stop, so that a new path can start (0: stay) |
| `guard_blind_speed` | 0.08 m/s | Top speed while "blind" |
| `guard_max_age` | 1.0 s | Age of the newest depth picture beyond which the guard is blind: ten pictures missed in a row. Not the memory of obstacles (was 0.5) |
| `guard_blind_time` | 3 s | Blind for longer: the depth camera is given up, the rover drives at its speed, unprotected (`no_camera`) |
| `guard_blind_stop` | 0 | 1: end the path (`no depth camera`) instead of driving on unprotected |
| `local_hold` | 10 s | A cell is forgotten this long after its last look |
| `local_hold_near` | 30 s | The same for cells within 0.45 m (the camera cannot see the floor under its nose, so it cannot clear them); was "never" |
| `local_confirm` | 2 | Pictures out of the last three that make a cell an obstacle; two free ones clear it. 1: every picture at its word |
| `d405_x`, `d405_y` | 0, 0 m | Lens ahead of, and left of, the rover's centre of rotation (0.09, 0 on the robot) |
| `d405_yaw_deg` | 0 | Lens turned to the left of the rover's axis. Not measured |

Capture server (`rover_mapper/d405_server.py`): `--grid-hz` 10 (0: no grid), `--cam-height`
0.15 m, `--cam-height-tol` 0.03, `--grid-stride` 2, `--grid-range` 1.25 m, `--floor-file`,
and against phantoms (estimates, to adjust on recordings): `--grid-obstacle-range` 0.8 m,
`--grid-band-slope` 0.03 (the 4 cm obstacle height grows by 3 cm per metre of distance),
`--grid-min-points` 5, `--threads` 1.

## What the first drives showed (review of 2026-11-17)

Everything added obstacles, almost nothing removed them, and the layers did not agree on
where things were.

- **Depth grid.** One fixed floor plane, a rover that pitches when it brakes, a short-range
  camera read out to 1.25 m, and three points in one picture enough for an obstacle. In a
  synthetic scene a 3 degree pitch turns the floor beyond 0.8 m into obstacles; 2 degrees
  is at the limit at 1.25 m. Not measured on the robot.
- **The 10 s discard did not do its job.** Forgetting only ran when a new picture arrived;
  cells within 0.45 m were never forgotten; and the fused map never forgot at all: a cell
  the rover dropped came back as "unknown", which erases nothing there.
- **The guard.** Its corridor (0.145 m) was the planner's keep-out (0.15 m); one cell in
  one picture ended the path; each stop put a 20 cm disc on the map for 30 s, grown to
  50 cm by the planner; after a stop the rover stood inside its own corridor for any new
  path; it also blocked the pivot that would have turned it away; without a fresh picture
  it crept at 0.08 m/s for ever.
- **Fusion.** A known rover cell replaced the drone's, in a different frame: the same box
  twice, a few centimetres apart, each grown by 0.15 m; a rover "free" erasing a real drone
  obstacle; nothing of the rover's ever expiring.

## Interfaces added

- `d405_server`: `GET /grid` -> `{res, x0, y0, cells, floor, plane, seq, age_s, stats}`;
  `POST /calibrate_floor[?force=1]`; `GET /calibration`; `/health` has a `grid` block with
  `cpu_ms_per_grid`.
- `rover_server`: `GET /local_grid` -> `{res, x0, y0, cells, pose: [x, y, heading rad],
  age_s, frames, error}`, odometry frame, cell `[i][j]` starts at `(x0 + j*res, y0 + i*res)`.
  `/state` has `local` (`feed`, `error`, `grids`, `age_s`, `obstacle`, `free`) and
  `ctrl.local` (`state`: off, ok, slow, blind, no_camera, blocked; `obstacle_dist`).
  After a `blocked` stop `ctrl.guard.state` is `backoff` then `hold`, as for the tilt guard.
  New status `no depth camera` (only with `guard_blind_stop`).
- `POST /path` takes `"id"` (1 to 40 characters), given back in `ctrl.current.id` and in the
  history entry; a replaced entry has `replaced_by`. New status `blocked`, with
  `obstacle: [x, y]` in the odometry frame.
- `status.json` of the maps server: `local` (`source`, `age_s`, `erreur`, `garde`,
  `obstacle_dist`) and `conduite` (`pas`, `segments`, `continu`). `POST /conduite`
  `{"continu": bool, "source_locale": "rover"|"service", "fusion":
  "superposition"|"global"|"local"}` (any key) changes `--continu`, `--source-locale` and
  `--fusion` while it runs; 409 during a mission. `status.json` has `fusion` (above).
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

- **Maps**: which maps the planner uses (`--fusion`); the line under the switches says
  which are in use and why, how many rover obstacle cells are in the plan, how many were
  left out as the drone's, and the guard's stops. The guard shows "NO DEPTH PICTURE: driving
  UNPROTECTED" in red when it has given up on the depth camera. Not tried in a browser.

Each switch works alone: the guard with missions in segments (`--segment`: a `blocked`
segment puts the obstacle on the map and a new route is planned; three refusals in a row
before the rover moves end the mission as blocked) but not yet with the step-by-step loop
(`--segment 0`), which does not read `blocked`; `--source-locale rover` without `--continu`;
`--continu` without the local map. With the guard on, prefer `--source-locale rover`: the
planner then sees what the guard sees. After a guard stop the rover now moves 10 cm back
(`guard_backoff`) and the guard does not check a pivot on the spot, so a new route can
leave; the "three refusals" rule stays as a last resort.

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
- **Obstacles that leave.** A cell the rover's camera sees free in two pictures is cleared,
  on the rover and on the maps. A cell it no longer sees is forgotten `local_hold` (10 s)
  after its last look, 30 s within 0.45 m, and with `--source-locale rover` it leaves the
  maps at the next plan (the rover's layer is rebuilt). What still does not clear by being
  seen: what the guard reported lasts `--duree-obstacle` (10 s, one cell); a bump (tilt,
  stall) is a forbidden zone until "Reset the maps", on purpose; and the views of the
  perception service stay until that cell is seen free.
- **Unprotected without the depth camera.** After 4 s without a depth picture (1 s blind,
  3 s more) the rover drives at its speed with no guard. That is what it did before the
  guard existed, and it is shown in red, but it is a choice: `guard_blind_stop` 1 ends the
  path instead.
- **The same obstacle seen twice** is left out only within 15 cm of the drone's. A larger
  disagreement between odometry and camera frame shows up as two obstacles again. Aligning
  the rover's grid to the drone map would remove this at its root; not done.
- **The numbers of the depth grid** (0.8 m, 3 cm per metre, 5 points) are estimates.
- **The link.** Through the VM's tunnel (1.3 to 4 s a request) the rover stays safe but ends
  each path before the next arrives. Smooth driving needs the planner on the room's Wi-Fi.
- **Two processes on the RDK X5**, which has no fan; the grid is computed only while asked.

## Tests

| File | Covers |
|---|---|
| `rover_mapper/test_d405_grid.py` | Grid from a synthetic floor 15 cm under the lens and a box; floor calibration kept, refused at a wrong height, still working facing a wall; `/grid` over HTTP with a fake pyrealsense2 |
| `rover/tests/test_local_map.py` | Placing a rotated grid without holes; one picture decides nothing, two of three do; forgetting without pictures arriving; near cells kept longer, not for ever |
| `rover/tests/test_path_guard.py` | Slow, `blocked` with the nose 4 cm or more from the box, back-off; a path that turns away can start; a phantom in one picture does not slow the rover; a path 15 cm from obstacles is driven; blind then unprotected, or stopped; take-over at speed |
| `rover/tests/test_server.py` | `/local_grid`, the feed, path ids over HTTP |
| `merging_map/test_suivi_continu.py` | Corner rounding, replacement rule, placing the rover grid |
| `merging_map/test_conduite_continue.py` | End to end against the real rover server (simulated ESP32, stand-in camera with a field of view and shadows): clear way driven without the speed dropping; a box only the rover sees driven round without contact; a wall across the zone ends `bloque` short of it; phantoms in one picture of three beyond 0.6 m: no stop, arrives; the depth camera dies on the way: slow, then unprotected at speed, maps go to `global`, arrives. Real time, about 80 s |
| `sim/test_fusion_reelle.py` | Rover grid fused at the right place while moving; defaults off; an obstacle seen free again leaves the map with both sources, one the rover forgot leaves it too; the guard's obstacle is one cell, expires and is not erased before; superposition (the rover adds, erases nothing of the drone; the same box seen twice is one; only near the rover); one map missing (late grid, 3 s outage, no drone map, neither); the mode asked for and switched; in segments a `blocked` segment changes the route, three before moving end the mission |

## Not done

- The rover's own web page has no box for `local_guard` (use `POST /config`, or the
  dashboard's "Rover guard" box).
- `merging_map/lancer_overhead.py` and the step-by-step loop do not read `blocked` (the
  segments and continuous loops do).
- The rework of 17 Nov on the robot: none of it has been driven. The dashboard's new
  selector and line have not been opened in a browser, and `teleop`'s tests do not run in
  the dev container (no `requests`).
- Measuring the depth grid's numbers and `d405_yaw_deg` on recordings.
- Aligning the rover's grid to the drone map; pitch from the IMU into the floor plane.
- On the rover, a short path whose end is behind the nose ends `done` at once
  (`PathFollower.update`: the "just past it" rule also holds before the rover has moved).
  The segments loop works round it by pivoting first; the continuous loop does not.
- Steering round an obstacle on the rover itself, and Nav2. A later Nav2 bridge would need
  the same rover interfaces (`/local_grid`, `id`, `blocked`).
