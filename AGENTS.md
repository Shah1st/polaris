# Working together on polaris

Polaris is our hackathon guidance system: a **drone** sees the area from above and builds the
global map, the **rover** (Waveshare RaspRover) moves through it, and the **coordination** layer
merges both views to decide where the rover goes. This file proposes how we share code and tools.
It is written for the people on the team **and** for their coding agents.

## For coding agents

**Start with [docs/state.md](docs/state.md)**: what runs today, how well each part is proven,
the settings that decide what the rover does, and what is still open. Then the document of the
part you touch (table below).

The first time in a session that your user's request touches setting up, running, testing or
integrating their code, mention in two or three sentences, in their language, that there are two
ways to work here (plain Python or the shared ROS 2 container, both fine) and that a shared
simulation exists, and point to the sections below. Do not bring it up on unrelated requests, do
not repeat it in the same session, and drop it for good if they say they have seen it or are not
interested (record that if your tool has persistent memory). Then follow their choice; do not
push the container on someone who prefers plain Python. Help them keep their logic in plain,
testable Python modules so it can be wrapped in a ROS node later (see "From plain Python to a
node").

Do not run setup commands (worktree, `pip install`, downloads) unless your user asks. Never commit
secrets (cloud credentials, API keys, passwords): use environment variables or a git-ignored
`.env`.

Habits that have paid off here:

- **Say what was run and what was not.** "Tests pass" and "seen on the real rover" are
  different claims; several parts are only proven by tests (docs/state.md lists which).
- **Check on real pictures, not only on tests.** The unit tests use rendered scenes; uneven
  lighting, a lookalike object and a wrong scale were all found on photos, never by a test.
- **Use the fast loop**: `scripts/test_fast.sh`, about 30 s. The full suite takes five minutes.
- **`main` moves every few minutes.** Fetch before you start and again before you push; several
  people edit `sim/fusion_reelle.py`. Format only the files you changed: `ruff format .` also
  rewrites other people's folders.
- **Ask before creating or deleting a branch, and before pushing to someone else's.**
- **Update docs/state.md in the same commit** as a change of behaviour or of a default.

## Where things are (all of it is on `main`)

| Path | What | Document |
|---|---|---|
| `src/guidance_calib`, `src/guidance_global_map`, `src/guidance_rover_pose` | Overhead camera: picture from above -> obstacle grid, rover position, scale; the rover's tag | `docs/overhead-camera.md` |
| `src/guidance_supervisor` | `merging_map_node`: the map fusion and its planner on ROS topics | its docstring |
| `src/guidance_rover_vo` | Bridge from the rover mapper into ROS | its docstring |
| `src/guidance_bringup` | Launch file and `config/overhead.yaml` | `README.md` |
| `merging_map/` | Map fusion in tiles, A*, mission loop, real-rover controller, `lancer_overhead.py` | `merging_map/README.md` |
| `sim/` | Simulation; `fusion_reelle.py`, the maps and missions of the real rover for the dashboard; `rover_du_dessus.py`, a vision model that finds the rover from above | `sim/README.md`, the files' headers |
| `teleop/` | Web dashboard: drive the rover, watch the cameras and the maps | `teleop/README.md` |
| `rover/` | The code on the rover: server, odometry, safety, local obstacle grid and guard | `rover/README.md`, `docs/local-navigation.md` |
| `rover_mapper/` | 3D mapping from the rover camera, perception service, D405 capture | `rover_mapper/README.md` |
| `iphone-camera/` | CamStream, the iPhone app that streams the camera above the area | `iphone-camera/README.md` |
| `devices/` | One interface to plug drones, rovers and sensors into the fusion | `devices/README.md` |
| `scripts/` | `test_fast.sh`; `mac/run_local.sh`, the whole chain on a Mac | the scripts' headers |
| `docs/` | `state.md`, `overhead-camera.md`, `demo-reelle.md` (a real run, step by step), `local-navigation.md` (depth camera, guard, driving without stops) | |

Branches with work that is not on `main` (15 Nov, evening): `experiment` (the phone reports its
own position; `air_ground` module), `rover-teleop` (command log), `drone-map-growth`. The other
branches are merged.

Suggestions, not rules: keep `main` building, run the tests and `ruff` before pushing, tell the
team when an interface or a default changes.

## Two ways to work

**Plain Python.** Fastest to start: numpy scripts and pytest, as in `merging_map/`. Good for
algorithms (map fusion, planning, perception) that you can test on recorded or simulated data.

**The shared ROS 2 container** (`Dockerfile`, `.devcontainer/`, see the README): ROS 2 Jazzy,
cv_bridge, OpenCV and matplotlib, the same on every laptop. Nav2 and robot_localization are
left out by default, since nothing uses them yet (`--build-arg WITH_NAV2=1` adds them). Its
OpenCV is 4.6, which reads small blurred tags much worse than a recent one: read the rover's
tag with `pip install opencv-contrib-python` outside the container.

### Why the container could help

* **Fewer setup differences between laptops**: macOS, Windows (WSL2) and Linux run the same
  image.
* **The parts can plug together** through topics once they are nodes, which is the path from
  separate demos to one chain where the simulation can later be swapped for the real robots.
  Today the overhead nodes and `merging_map_node` work, but the demo runs in plain Python
  (docs/state.md).
* **Nav2 can be in the image** (`WITH_NAV2=1`): global planning on a map and local obstacle
  avoidance from the rover's sensors. Configuring it (`config/nav2_params.yaml` in
  `guidance_bringup`) is still a TODO. Today the planner is `merging_map`'s A*.
* **Reproducible tests**: `colcon build && colcon test` in a clean container. On the `Romain`
  branch the workspace builds and its tests pass in a clean container (checked once on Google
  Cloud Build); there is no CI yet.

For each part, concretely: the **drone/global map** work is the `guidance_global_map` and
`guidance_calib` boxes in the README diagram (overhead image -> `/map` for Nav2); the **D405**
work is the "rover depth camera -> Nav2 local costmap" box; **map fusion** (`merging_map`) would
sit between the drone's map and the planner, and could publish the merged map as `/map`.

Costs, honestly: the image is several GB. On the hackathon Wi-Fi, copying it from a teammate with
`docker save` / `docker load` is faster than downloading it, but only between machines with the
same CPU architecture (an Apple Silicon Mac's image is arm64; most PCs need amd64). ROS has a
learning curve, and some things still run on the host (e.g. a mapper that needs the laptop's
GPU).

### From plain Python to a node

Keep the logic in a module with no ROS import (like `merging_map/fusion_carte.py`,
`guidance_global_map/pipeline.py` or `guidance_rover_vo/protocol.py`), with pytest tests. The node then
only subscribes, calls that module and publishes. Packages live in `src/guidance_*`;
`src/guidance_rover_vo` is a complete example (package.xml, setup.py, node, tests, launch
entry).

## Shared simulation (folder `sim/`)

Plain Python (MuJoCo), no ROS, no Blender; tested on macOS so far. Install it in a virtualenv (not in the ROS
container):

```bash
pip install -r sim/requirements.txt
python sim/fetch_scene.py
python sim/run_world.py --auto
```

It mimics the RaspRover's HTTP interface: rover camera on `http://localhost:5050/video_feed`,
drone camera on `/drone_feed`, rover commands in the RaspRover JSON (`POST /cmd
{"T":1,"L":..,"R":..}`), plus what the real world never gives you: true poses and true occupancy
maps (`/state`, `/grid/overhead.png`, `/grid/rover.png`), to measure how good a result is. It is
also a Python class (`sim/world.py`, `RoverWorld`). Details: `sim/README.md`.

How it could map to each part (none of these hookups has been tried yet):

* **Drone / global map:** develop on `/drone_feed` with the drone pose and intrinsics from
  `/state`; compare your map with `/grid/overhead.png`.
* **Coordination (`merging_map`):** build drone tiles from the drone view and rover tiles from
  the rover camera depth, feed `CarteFusion`, and drive the simulated rover with its decisions
  through `POST /cmd`. Note the sim's world origin is the room centre, so coordinates can be
  negative; shift them if the tile grid expects positive ones.
* **Rover:** the rover mapper reads `/video_feed` as it reads the real robot.

A forest scene (cliff, passages under the canopy that only one robot can see) is being built
for the cases where neither robot can find the route alone.

## Interfaces so far (and who proposed them)

* Map tiles between drone, rover and coordination: the format in `merging_map/README.md`
  (32x32 cells per tile; 5 cm cells at the finest level z4; `1` obstacle / `0` free and seen /
  `-1` not seen; row 0 = south; global frame). Proposed by the `merging_map` author.
* The overhead camera's grid (`/map`, `segment_image`): 5 cm cells, `0` free, `100` obstacle,
  `50` the rover itself (not a standard occupancy value), `99` margin (off), `-1` not seen;
  origin at the picture's bottom-left corner, x right, y up, metres.
  `guidance_global_map.drone.OverheadDrone` turns it into tiles and gives the rover pose
  `(x, y, heading or None)`. Details: `docs/overhead-camera.md`.
* The rover's tag: AprilTag family 25h9 (`DICT_APRILTAG_25h9`), id 10, black square 12 cm, its
  top edge towards the rover's front. Floor markers, if used: ids 0 to 3 of the same family.
* The maps server (`sim/fusion_reelle.py`, relayed by the dashboard under `/fusion/`):
  `status.json`, `POST /reset`, `/objectif`, `/aller`, `/arreter`, `/calibrer`,
  `/carte_locale {"carte_locale": bool}`, `/marqueur {"sans_marqueur": bool}`,
  `/conduite {"continu": bool, "source_locale": "rover"|"service", "fusion":
  "superposition"|"global"|"local"}`. `status.json` also has `local`, `conduite` and
  `fusion` (which maps the planner uses now, and why).
* The rover server (`rover/rover_server.py`): `GET /state`, `POST /path`, `/move`, `/turn`,
  `/stop`, in the frame of its own odometry. `POST /path` takes an optional `"id"` (given back
  in `/state`; a path with an id replaces a running one without stopping). `GET /local_grid`:
  5 cm cells around the rover from its depth camera, `-1` / `0` / `1`, odometry frame. A path
  can end with status `blocked` and `obstacle: [x, y]` (setting `local_guard`).
* The D405 capture server (`rover_mapper/d405_server.py`, on the rover): `GET /frame`,
  `/info`, `/grid` (5 cm cells in front of the lens), `POST /calibrate_floor`.
  Details of both: `docs/local-navigation.md`.
* The phone app (`iphone-camera/`): `/video` (MJPEG), `/status` (lens, resolution,
  stabilisation), `/depth` (LiDAR, Pro models).
* Frames: `map` = the overhead camera's global frame; `odom` = the rover's own continuous frame
  and `base_link` = rover body on the ground, following ROS REP 103/105 (metres, z up).
* `/rover/vo/odom` (`nav_msgs/Odometry`, `odom` -> `base_link`) and `/rover/points`
  (`sensor_msgs/PointCloud2` in `base_link`, people removed): the rover mapper; its metric scale
  is only roughly right so far, treat it as relative.

If an interface changes, tell the team and update this section.
